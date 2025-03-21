import torch
import datetime
import jsonlines
import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub.hf_api import HfFolder
from prompt import prompts
from utils import *


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
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=own_cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        cache_dir=own_cache_dir,
        device_map="auto",
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
                inputs = tokenizer([prompt1], return_tensors="pt")
                for k,v in inputs.items():
                    inputs[k] = v.cuda()
                gen_kwargs = {"max_length": 1024}
                output = model.generate(**inputs, **gen_kwargs)
                generation1 = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
                generation1 = extract_first_word_or_phrase(generation1)
                print(f">>> {generation1}")
                print("-----------------------------------------------------")

                print("[2] Generate self-reflection")
                prompt2 = prompts["prompt_2"].replace("{{country}}", country).replace("{{story}}", story).replace("{{rot}}", rot).replace("{{response}}", str(generation1))
                print(prompt2)
                inputs = tokenizer([prompt2], return_tensors="pt")
                for k,v in inputs.items():
                    inputs[k] = v.cuda()
                gen_kwargs = {"max_length": 1024}
                output = model.generate(**inputs, **gen_kwargs)
                generation2 = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
                generation2 = extract_first_word_or_phrase(generation2)
                print(f">>> {generation2}")
                print("-----------------------------------------------------")

                print("[3] Make final decision")
                prompt3 = prompts["prompt_3"].replace("{{country}}", country).replace("{{story}}", story).replace("{{rot}}", rot).replace("{{response}}", str(generation1)).replace("{{reflection}}", str(generation2))
                print(prompt3)
                inputs = tokenizer([prompt3], return_tensors="pt")
                for k,v in inputs.items():
                    inputs[k] = v.cuda()
                gen_kwargs = {"max_length": 1024}
                output = model.generate(**inputs, **gen_kwargs)
                generation3 = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
                generation3 = extract_first_word_or_phrase(generation3)
                print(f">>> {generation3}")
                print("-----------------------------------------------------")
                
                line[f"internlm_1"] = generation1
                line[f"internlm_2"] = generation2
                line[f"internlm_final"] = generation3
                outfile.write(line)
    
    end_time = datetime.datetime.now()
    print(f"Time elapsed: {end_time - start_time}")


if __name__ == "__main__":
    main()
