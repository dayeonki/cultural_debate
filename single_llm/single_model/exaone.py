import torch
import datetime
import jsonlines
import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub.hf_api import HfFolder
from prompt import prompts
from utils import country_capitalized_mapping


def extract_answer(text):
    if "Answer (Yes, No or Neither):" in text:
        answer = text.split("Answer (Yes, No or Neither):")[-1].strip()
        return answer.split()[0]  # Return only the first word (Yes, No, Neither)
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
    parser.add_argument("--type", type=str, default="without_rot")
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

                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ]
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt"
                )

                output = model.generate(
                    input_ids.to("cuda"),
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=512
                )
                generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

                generation = extract_answer(generated_text)

                print(f"> {generation}")
                print("\n======================================================\n")
                generations.append(generation)

                line[f"exaone"] = generation
                outfile.write(line)
    
    end_time = datetime.datetime.now()
    print(f"Time elapsed: {end_time - start_time}")


if __name__ == "__main__":
    main()