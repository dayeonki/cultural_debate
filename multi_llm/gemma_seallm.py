import torch
import datetime
import jsonlines
import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub.hf_api import HfFolder
from prompt import prompts
from utils import country_capitalized_mapping, parse_final_answer, parse_response


own_cache_dir = ""
os.environ["HF_HOME"] = own_cache_dir
os.environ["HF_DATASETS"] = own_cache_dir

model1_id = "google/gemma-2-9b-it"
model2_id = "SeaLLMs/SeaLLMs-v3-7B-Chat"


def main():
    start_time = datetime.datetime.now()

    hf_token = ""
    HfFolder.save_token(hf_token)

    # =========================================== Parameter Setup ===========================================
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)

    args = parser.parse_args()

    tokenizer1 = AutoTokenizer.from_pretrained(model1_id, cache_dir=own_cache_dir)
    model1 = AutoModelForCausalLM.from_pretrained(
        model1_id,
        torch_dtype=torch.bfloat16,
        cache_dir=own_cache_dir,
        device_map="auto",
    )

    model2 = AutoModelForCausalLM.from_pretrained(
        model2_id,
        torch_dtype=torch.bfloat16, 
        cache_dir=own_cache_dir,
        device_map="auto"
    )
    tokenizer2 = AutoTokenizer.from_pretrained(model2_id, cache_dir=own_cache_dir)

    # =========================================== Load Existing IDs from Output ===========================================
    processed_ids = set()
    if os.path.exists(args.output_path):
        with jsonlines.open(args.output_path) as outfile:
            for line in outfile.iter():
                processed_ids.add(line['ID'])

    # =========================================== Load Dataset ===========================================    
    generations1, generations2 = [], []

    with jsonlines.open(args.output_path, mode="a") as outfile:
        with jsonlines.open(args.input_path) as file:
            for line in file.iter():
                current_id = line['ID']

                # Skip if already processed
                if current_id in processed_ids:
                    print(f"Skipping already processed ID: {current_id}")
                    continue
                
                country_small = line['Country']
                country = country_capitalized_mapping[country_small]
                story = line['Story']
                rule = line['Rule-of-Thumb']
                gold_label = line["Gold Label"]

                # [1] Model1 (rot+story) - stance (socially acceptable/not acceptable) + reason
                print("[1] Model1 - Make initial decision")
                prompt1_1 = prompts["prompt_1"].replace("{{country}}", country).replace("{{story}}", story).replace("{{rule}}", rule)
                print(prompt1_1)
                input_ids = tokenizer1(prompt1_1, return_tensors="pt").to("cuda")
                outputs = model1.generate(**input_ids, max_new_tokens=1024, temperature=0.0)
                generation1_1 = tokenizer1.decode(outputs[0])
                generation1_1 = parse_response(generation1_1, "Answer:")
                print(f">>> {generation1_1}")
                print("-----------------------------------------------------")

                print("[2] Model2 - Make initial decision")
                prompt2_1 = prompts["prompt_1"].replace("{{country}}", country).replace("{{story}}", story).replace("{{rule}}", rule)
                print(prompt2_1)
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt2_1}
                ]
                text = tokenizer2.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                model_inputs = tokenizer2([text], return_tensors="pt").to('cuda')

                generated_ids = model2.generate(model_inputs.input_ids, max_new_tokens=1024, do_sample=False, temperature=0.0, eos_token_id=tokenizer2.eos_token_id)
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                generation = tokenizer2.batch_decode(generated_ids, skip_special_tokens=True)[0]
                generation2_1 = parse_response(generation)
                print(f">>> {generation2_1}")
                print("-----------------------------------------------------")

                print("[3] Model1 - Give feedback to 2")
                prompt1_2 = prompts["prompt_2"].replace("{{country}}", country).replace("{{story}}", story).replace("{{rule}}", rule).replace("{{other_response}}", generation2_1).replace("{{your_response}}", generation1_1)
                print(prompt1_2)
                input_ids = tokenizer1(prompt1_2, return_tensors="pt").to("cuda")
                outputs = model1.generate(**input_ids, max_new_tokens=1024, temperature=0.0)
                generation1_2 = tokenizer1.decode(outputs[0])
                generation1_2 = parse_response(generation1_2, "Response:")
                print(f">>> {generation1_2}")
                print("-----------------------------------------------------")

                print("[4] Model2 - Give feedback to 1")
                prompt2_2 = prompts["prompt_2"].replace("{{country}}", country).replace("{{story}}", story).replace("{{rule}}", rule).replace("{{other_response}}", generation1_1).replace("{{your_response}}", generation1_2)
                print(prompt2_2)
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt2_2}
                ]
                text = tokenizer2.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                model_inputs = tokenizer2([text], return_tensors="pt").to('cuda')

                generated_ids = model2.generate(model_inputs.input_ids, max_new_tokens=1024, do_sample=False, temperature=0.0, eos_token_id=tokenizer2.eos_token_id)
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                generation = tokenizer2.batch_decode(generated_ids, skip_special_tokens=True)[0]
                generation2_2 = parse_response(generation)
                print(f">>> {generation2_2}")
                print("-----------------------------------------------------")

                print("[5] Model1 - Make final decision")
                prompt1_3 = prompts["prompt_3"].replace("{{country}}", country).replace("{{story}}", story).replace("{{rule}}", rule).replace("{{feedback}}", generation2_2).replace("{{your_feedback}}", generation1_2).replace("{{your_response}}", generation1_1).replace("{{other_response}}", generation2_1)
                print(prompt1_3)
                input_ids = tokenizer1(prompt1_3, return_tensors="pt").to("cuda")
                outputs = model1.generate(**input_ids, max_new_tokens=1024, temperature=0.0)
                generation1_3 = tokenizer1.decode(outputs[0])
                generation1_3 = parse_final_answer(generation1_3)
                print(f">>> {generation1_3}")
                print("-----------------------------------------------------")

                print("[6] Model2 - Make final decision")
                prompt2_3 = prompts["prompt_3"].replace("{{country}}", country).replace("{{story}}", story).replace("{{rule}}", rule).replace("{{feedback}}", generation1_2).replace("{{your_feedback}}", generation2_2).replace("{{your_response}}", generation2_1).replace("{{other_response}}", generation1_1)
                print(prompt2_3)
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt2_3}
                ]
                text = tokenizer2.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                model_inputs = tokenizer2([text], return_tensors="pt").to('cuda')

                generated_ids = model2.generate(model_inputs.input_ids, max_new_tokens=1024, do_sample=False, temperature=0.0, eos_token_id=tokenizer2.eos_token_id)
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                generation = tokenizer2.batch_decode(generated_ids, skip_special_tokens=True)[0]
                generation2_3 = parse_final_answer(generation)
                print(f"> {generation2_3}")
                print("-----------------------------------------------------")

                print(f"Gold > {gold_label}")
                print(f"Gemma > {generation1_3}")
                print(f"SeaLLM > {generation2_3}")
                print("\n======================================================\n")
                generations1.append(generation1_3)
                generations2.append(generation2_3)

                line[f"gemma_1"] = generation1_1
                line[f"seallm_1"] = generation2_1
                line[f"gemma_2"] = generation1_2
                line[f"seallm_2"] = generation2_2
                line[f"gemma_final"] = generation1_3
                line[f"seallm_final"] = generation2_3
                outfile.write(line)
    
    end_time = datetime.datetime.now()
    print(f"Time elapsed: {end_time - start_time}")


if __name__ == "__main__":
    main()


