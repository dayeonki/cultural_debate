import datetime
import jsonlines
import os
import argparse
from huggingface_hub.hf_api import HfFolder
from prompt import prompts
from utils import country_capitalized_mapping
from transformers import AutoModelForCausalLM, AutoTokenizer


model_id = '01-ai/Yi-1.5-9B-Chat'

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

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype="auto"
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

                messages = [
                    {"role": "user", "content": prompt}
                ]

                input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, return_tensors='pt')
                output_ids = model.generate(input_ids.to('cuda'), eos_token_id=tokenizer.eos_token_id, max_length=256)
                generated_text = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

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

                line[f"yi"] = generation
                outfile.write(line)
    
    end_time = datetime.datetime.now()
    print(f"Time elapsed: {end_time - start_time}")


if __name__ == "__main__":
    main()