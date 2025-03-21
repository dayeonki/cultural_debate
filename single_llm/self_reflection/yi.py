import torch
import datetime
import jsonlines
import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub.hf_api import HfFolder
from prompt import prompts
from utils import *


model2_id = '01-ai/Yi-1.5-9B-Chat'


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
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype="auto"
    ).eval()

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
                input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, return_tensors='pt')
                output_ids = model.generate(input_ids.to('cuda'), eos_token_id=tokenizer.eos_token_id, max_length=1024)
                generation1 = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
                print(f">>> {generation1}")
                print("-----------------------------------------------------")

                print("[2] Generate self-reflection")
                prompt2 = prompts["prompt_2"].replace("{{country}}", country).replace("{{story}}", story).replace("{{rot}}", rot).replace("{{response}}", str(generation1))
                print(prompt2)
                messages = [{"role": "user", "content": prompt2}]
                input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, return_tensors='pt')
                output_ids = model.generate(input_ids.to('cuda'), eos_token_id=tokenizer.eos_token_id, max_length=1024)
                generation2 = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
                print(f">>> {generation2}")
                print("-----------------------------------------------------")

                print("[3] Make final decision")
                prompt3 = prompts["prompt_3"].replace("{{country}}", country).replace("{{story}}", story).replace("{{rot}}", rot).replace("{{response}}", str(generation1)).replace("{{reflection}}", str(generation2))
                print(prompt3)
                messages = [{"role": "user", "content": prompt3}]
                input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, return_tensors='pt')
                output_ids = model.generate(input_ids.to('cuda'), eos_token_id=tokenizer.eos_token_id, max_length=1024)
                generation3 = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
                print(f">>> {generation3}")
                print("-----------------------------------------------------")
                
                line[f"yi_1"] = generation1
                line[f"yi_2"] = generation2
                line[f"yi_final"] = generation3
                outfile.write(line)
    
    end_time = datetime.datetime.now()
    print(f"Time elapsed: {end_time - start_time}")


if __name__ == "__main__":
    main()
