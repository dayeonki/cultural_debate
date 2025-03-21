import torch
import datetime
import jsonlines
import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub.hf_api import HfFolder
from prompt import prompts
from utils import *


model_id = "SeaLLMs/SeaLLMs-v3-7B-Chat"


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
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16, 
        cache_dir=own_cache_dir,
        device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=own_cache_dir
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
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                model_inputs = tokenizer([text], return_tensors="pt").to('cuda')

                generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=1024, eos_token_id=tokenizer.eos_token_id)
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                generation1 = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                print(f">>> {generation1}")
                print("-----------------------------------------------------")

                print("[2] Generate self-reflection")
                prompt2 = prompts["prompt_2"].replace("{{country}}", country).replace("{{story}}", story).replace("{{rot}}", rot).replace("{{response}}", str(generation1))
                print(prompt2)
                messages = [{"role": "user", "content": prompt2}]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                model_inputs = tokenizer([text], return_tensors="pt").to('cuda')

                generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=1024, eos_token_id=tokenizer.eos_token_id)
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                generation2 = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                print(f">>> {generation2}")
                print("-----------------------------------------------------")

                print("[3] Make final decision")
                prompt3 = prompts["prompt_3"].replace("{{country}}", country).replace("{{story}}", story).replace("{{rot}}", rot).replace("{{response}}", str(generation1)).replace("{{reflection}}", str(generation2))
                print(prompt3)
                messages = [{"role": "user", "content": prompt3}]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                model_inputs = tokenizer([text], return_tensors="pt").to('cuda')

                generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=1024, eos_token_id=tokenizer.eos_token_id)
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                generation3 = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                print(f">>> {generation3}")
                print("-----------------------------------------------------")
                
                line[f"seallm_1"] = generation1
                line[f"seallm_2"] = generation2
                line[f"seallm_final"] = generation3
                outfile.write(line)
    
    end_time = datetime.datetime.now()
    print(f"Time elapsed: {end_time - start_time}")


if __name__ == "__main__":
    main()
