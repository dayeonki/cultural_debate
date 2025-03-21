import torch
import datetime
import jsonlines
import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub.hf_api import HfFolder
from prompt import prompts
from utils import *


def extract_answer(text, prefix):
    if prefix in text:
        answer = text.split(prefix)[-1].strip()
        return answer.split()[0]
    elif "[|assistant|]" in text:
        answer = text.split("[|assistant|]")[-1].strip()
        return answer.split()[0]  # Return only the first word (Yes, No, Neither)
    return text.strip()


model_id = "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"


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
        trust_remote_code=True,
        device_map="auto"
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
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt"
                )
                output = model.generate(
                    input_ids.to("cuda"),
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=1024,
                )
                generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
                generation1 = extract_answer(generated_text, "Answer:")
                print(f">>> {generation1}")
                print("-----------------------------------------------------")

                print("[2] Generate self-reflection")
                prompt2 = prompts["prompt_2"].replace("{{country}}", country).replace("{{story}}", story).replace("{{rot}}", rot).replace("{{response}}", str(generation1))
                print(prompt2)
                messages = [{"role": "user", "content": prompt2}]
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt"
                )
                output = model.generate(
                    input_ids.to("cuda"),
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=1024,
                )
                generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
                generation2 = extract_answer(generated_text, "Reflection:")
                print(f">>> {generation2}")
                print("-----------------------------------------------------")

                print("[3] Make final decision")
                prompt3 = prompts["prompt_3"].replace("{{country}}", country).replace("{{story}}", story).replace("{{rot}}", rot).replace("{{response}}", str(generation1)).replace("{{reflection}}", str(generation2))
                print(prompt3)
                messages = [{"role": "user", "content": prompt3}]
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt"
                )
                output = model.generate(
                    input_ids.to("cuda"),
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=1024,
                )
                generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
                generation3 = extract_answer(generated_text, "Answer (Yes, No or Neither):")
                print(f">>> {generation3}")
                print("-----------------------------------------------------")
                
                line[f"exaone_1"] = generation1
                line[f"exaone_2"] = generation2
                line[f"exaone_final"] = generation3
                outfile.write(line)
    
    end_time = datetime.datetime.now()
    print(f"Time elapsed: {end_time - start_time}")


if __name__ == "__main__":
    main()
