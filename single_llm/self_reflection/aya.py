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
    args = parser.parse_args()
    
    pipe = pipeline(
        "text-generation", 
        model=model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )

    # =========================================== Load Dataset =========================================== 
    with jsonlines.open(args.output_path, mode="w") as outfile:
        with jsonlines.open(args.input_path) as file:
            for line in file.iter():
                country_small = line['Country']
                country = country_capitalized_mapping[country_small]
                story = line['Story']
                rot = line["Rule-of-Thumb"]

                print("[1] Make initial decision")
                prompt1 = prompts["prompt_1"].replace("{{country}}", country).replace("{{story}}", story).replace("{{rot}}", rot)
                print(prompt1)
                messages = [{"role": "user", "content": prompt1}]
                outputs = pipe(messages, max_new_tokens=1024)
                generated_text = outputs[0]["generated_text"]
                generated_text = generated_text[1]['content']
                generation1 = extract_answer(generated_text)
                print(f">>> {generation1}")
                print("-----------------------------------------------------")

                print("[2] Generate self-reflection")
                prompt2 = prompts["prompt_2"].replace("{{country}}", country).replace("{{story}}", story).replace("{{rot}}", rot).replace("{{response}}", str(generation1))
                print(prompt2)
                messages = [{"role": "user", "content": prompt2}]
                outputs = pipe(messages, max_new_tokens=1024)
                generated_text = outputs[0]["generated_text"]
                generated_text = generated_text[1]['content']
                generation2 = extract_answer(generated_text)
                print(f">>> {generation2}")
                print("-----------------------------------------------------")

                print("[3] Make final decision")
                prompt3 = prompts["prompt_3"].replace("{{country}}", country).replace("{{story}}", story).replace("{{rot}}", rot).replace("{{response}}", str(generation1)).replace("{{reflection}}", str(generation2))
                print(prompt3)
                messages = [{"role": "user", "content": prompt3}]
                outputs = pipe(messages, max_new_tokens=1024)
                generated_text = outputs[0]["generated_text"]
                generated_text = generated_text[1]['content']
                generation3 = extract_answer(generated_text)
                print(f">>> {generation3}")
                print("-----------------------------------------------------")
                
                line[f"aya_1"] = generation1
                line[f"aya_2"] = generation2
                line[f"aya_final"] = generation3
                outfile.write(line)
    
    end_time = datetime.datetime.now()
    print(f"Time elapsed: {end_time - start_time}")


if __name__ == "__main__":
    main()
