import torch
import datetime
import jsonlines
import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub.hf_api import HfFolder
from prompt import prompts
from utils import country_capitalized_mapping


def extract_first_word_or_phrase(text):
    delimiters = [".", ",", "\n", ", "]
    for delimiter in delimiters:
        if delimiter in text:
            return text.split(delimiter)[0].strip()
    return text.strip()

model_id = "internlm/internlm2_5-7b"


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

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=own_cache_dir,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=own_cache_dir, 
        torch_dtype=torch.float16, 
        trust_remote_code=True
    ).eval()

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

                inputs = tokenizer([prompt], return_tensors="pt")
                for k,v in inputs.items():
                    inputs[k] = v.cuda()
                gen_kwargs = {"max_length": 1024, "temperature": 0.0, "do_sample": False}
                output = model.generate(**inputs, **gen_kwargs)
                generated_text = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)

                answer_start = "Answer (Yes, No or Neither):"
                if answer_start in generated_text:
                    try:
                        generation = generated_text.split(answer_start)[-1].strip().split()[0]
                    except: 
                        generation = generated_text
                else:
                    generation = generated_text
                
                print(f"> {generation}")
                print("\n======================================================\n")
                generations.append(generation)

                line[f"internlm"] = generation
                outfile.write(line)
    
    end_time = datetime.datetime.now()
    print(f"Time elapsed: {end_time - start_time}")


if __name__ == "__main__":
    main()