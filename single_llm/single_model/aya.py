import torch
import datetime
import jsonlines
import os
import argparse
from transformers import pipeline
from huggingface_hub.hf_api import HfFolder
from prompt import prompts
from utils import *


def extract_answer(text):
    if "<|CHATBOT_TOKEN|>" in text:
        answer_part = text.split("<|CHATBOT_TOKEN|>")[-1]
        return answer_part.split("<|END_OF_TURN_TOKEN|>")[0].strip()
    return text


model_id = "CohereForAI/aya-23-8B"


own_cache_dir = ""
os.environ["HF_HOME"] = own_cache_dir
os.environ["HF_DATASETS"] = own_cache_dir

def main():
    start_time = datetime.datetime.now()

    hf_token = ""
    HfFolder.save_token(hf_token)

    # =========================================== Parameter Setup ===========================================
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--type", type=str, default="without_rot")
    args = parser.parse_args()
    
    pipe = pipeline(
        "text-generation", 
        model=model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )

    # =========================================== Load Dataset =========================================== 
    generations = []
    with jsonlines.open(args.output_path, mode="w") as outfile:
        with jsonlines.open(args.input_path) as file:
            for line in file.iter():
                country_small = line['Country']
                country = country_capitalized_mapping[country_small]
                story = line['Story']
                rot = line["Rule-of-Thumb"]
                
                if args.type == "without_rot":
                    en_prompt = prompts["en"]
                    prompt = en_prompt.replace("{{country}}", country).replace("{{story}}", story)
                elif args.type == "with_rot":
                    en_prompt = prompts["en_rot"]
                    prompt = en_prompt.replace("{{country}}", country).replace("{{story}}", story).replace("{{rot}}", rot)
                else:
                    raise ValueError("type argument should be either 'without_rot' or 'with_rot'.")
                print(prompt)

                messages = [{"role": "user", "content": prompt}]
                outputs = pipe(messages, max_new_tokens=1024, do_sample=False)
                generated_text = outputs[0]["generated_text"]

                generated_text = generated_text[1]['content']

                answer_start = "Answer (Yes, No or Neither):"
                if answer_start in generated_text:
                    try:
                        generation = generated_text.split(answer_start)[-1].strip().split()[0]
                    except: 
                        generation = generated_text
                else:
                    generation = generated_text
                
                generation = extract_answer(generation)
                
                print(f"> {generation}")
                print("\n======================================================\n")
                generations.append(generation)

                line[f"aya"] = generation
                outfile.write(line)
    
    end_time = datetime.datetime.now()
    print(f"Time elapsed: {end_time - start_time}")


if __name__ == "__main__":
    main()