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

model1_id = "meta-llama/Meta-Llama-3-8B-Instruct"
model2_id = "CohereForAI/aya-23-8B"

def extract_answer(text):
    if "<|CHATBOT_TOKEN|>" in text:
        answer_part = text.split("<|CHATBOT_TOKEN|>")[-1]
        return answer_part.split("<|END_OF_TURN_TOKEN|>")[0].strip()
    return text


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

    pipe = pipeline("text-generation", model=model2_id, torch_dtype=torch.bfloat16, device_map="auto")

    # =========================================== Load Dataset ===========================================    
    generations1, generations2 = [], []

    with jsonlines.open(args.output_path, mode="w") as outfile:
        with jsonlines.open(args.input_path) as file:
            for line in file.iter():
                country_small = line['Country']
                country = country_capitalized_mapping[country_small]
                story = line['Story']
                rule = line['Rule-of-Thumb']
                gold_label = line["Gold Label"]

                # [1] Model1 (rot+story) - stance (socially acceptable/not acceptable) + reason
                print("[1] Model1 - Make initial decision")
                prompt1_1 = prompts["prompt_1"].replace("{{country}}", country).replace("{{story}}", story).replace("{{rule}}", rule)
                print(prompt1_1)
                messages = [{"role": "user", "content": prompt1_1}]
                input_ids = tokenizer1.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model1.device)
                terminators = [tokenizer1.eos_token_id, tokenizer1.convert_tokens_to_ids("<|eot_id|>")]
                outputs = model1.generate(input_ids, max_new_tokens=1024, eos_token_id=terminators, do_sample=False, temperature=0.0,)
                response = outputs[0][input_ids.shape[-1]:]
                generation1_1 = tokenizer1.decode(response, skip_special_tokens=True)
                generation1_1 = parse_response(generation1_1, "Answer:")
                print(f">>> {generation1_1}")
                print("-----------------------------------------------------")

                print("[2] Model2 - Make initial decision")
                prompt2_1 = prompts["prompt_1"].replace("{{country}}", country).replace("{{story}}", story).replace("{{rule}}", rule)
                print(prompt2_1)
                messages = [{"role": "user", "content": prompt2_1}]
                outputs = pipe(messages, max_new_tokens=1024, do_sample=False, temperature=0.0)
                generated_text = outputs[0]["generated_text"]
                generated_text = generated_text[1]['content']
                generation2_1 = extract_answer(generated_text)
                generation2_1 = parse_response(generation2_1, "Answer:")
                print(f">>> {generation2_1}")
                print("-----------------------------------------------------")

                print("[3] Model1 - Give feedback to 2")
                prompt1_2 = prompts["prompt_2"].replace("{{country}}", country).replace("{{story}}", story).replace("{{rule}}", rule).replace("{{other_response}}", generation2_1).replace("{{your_response}}", generation1_1)
                print(prompt1_2)
                messages = [{"role": "user", "content": prompt1_2}]
                input_ids = tokenizer1.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model1.device)
                terminators = [tokenizer1.eos_token_id, tokenizer1.convert_tokens_to_ids("<|eot_id|>")]
                outputs = model1.generate(input_ids, max_new_tokens=1024, eos_token_id=terminators, do_sample=False, temperature=0.0,)
                response = outputs[0][input_ids.shape[-1]:]
                generation1_2 = tokenizer1.decode(response, skip_special_tokens=True)
                generation1_2 = parse_response(generation1_2, "Response:")
                print(f">>> {generation1_2}")
                print("-----------------------------------------------------")

                print("[4] Model2 - Give feedback to 1")
                prompt2_2 = prompts["prompt_2"].replace("{{country}}", country).replace("{{story}}", story).replace("{{rule}}", rule).replace("{{other_response}}", generation1_1).replace("{{your_response}}", generation1_2)
                print(prompt2_2)
                messages = [{"role": "user", "content": prompt2_2}]
                outputs = pipe(messages, max_new_tokens=1024, do_sample=False, temperature=0.0)
                generated_text = outputs[0]["generated_text"]
                generated_text = generated_text[1]['content']
                generation2_2 = extract_answer(generated_text)
                generation2_2 = parse_response(generation2_2)
                print(f">>> {generation2_2}")
                print("-----------------------------------------------------")

                print("[5] Model1 - Make final decision")
                prompt1_3 = prompts["prompt_3"].replace("{{country}}", country).replace("{{story}}", story).replace("{{rule}}", rule).replace("{{feedback}}", generation2_2).replace("{{your_feedback}}", generation1_2).replace("{{your_response}}", generation1_1).replace("{{other_response}}", generation2_1)
                print(prompt1_3)
                messages = [{"role": "user", "content": prompt1_3}]
                input_ids = tokenizer1.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model1.device)
                terminators = [tokenizer1.eos_token_id, tokenizer1.convert_tokens_to_ids("<|eot_id|>")]
                outputs = model1.generate(input_ids, max_new_tokens=1024, eos_token_id=terminators, do_sample=False, temperature=0.0,)
                response = outputs[0][input_ids.shape[-1]:]
                generation1_3 = tokenizer1.decode(response, skip_special_tokens=True)
                generation1_3 = parse_final_answer(generation1_3)
                print(f">>> {generation1_3}")
                print("-----------------------------------------------------")

                print("[6] Model2 - Make final decision")
                prompt2_3 = prompts["prompt_3"].replace("{{country}}", country).replace("{{story}}", story).replace("{{rule}}", rule).replace("{{feedback}}", generation1_2).replace("{{your_feedback}}", generation2_2).replace("{{your_response}}", generation2_1).replace("{{other_response}}", generation1_1)
                print(prompt2_3)
                messages = [{"role": "user", "content": prompt2_3}]
                outputs = pipe(messages, max_new_tokens=1024, do_sample=False, temperature=0.0)
                generated_text = outputs[0]["generated_text"]
                generated_text = generated_text[1]['content']
                generation2_3 = extract_answer(generated_text)
                generation2_3 = parse_final_answer(generation2_3)
                print(f"> {generation2_3}")
                print("-----------------------------------------------------")

                print(f"Gold > {gold_label}")
                print(f"LLaMA-3 > {generation1_3}")
                print(f"Aya > {generation2_3}")
                print("\n======================================================\n")
                generations1.append(generation1_3)
                generations2.append(generation2_3)

                line[f"llama3_1"] = generation1_1
                line[f"aya_1"] = generation2_1
                line[f"llama3_2"] = generation1_2
                line[f"aya_2"] = generation2_2
                line[f"llama3_final"] = generation1_3
                line[f"aya_final"] = generation2_3
                outfile.write(line)
    
    end_time = datetime.datetime.now()
    print(f"Time elapsed: {end_time - start_time}")


if __name__ == "__main__":
    main()


