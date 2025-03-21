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

model1_id = "internlm/internlm2_5-7b"
model2_id = "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"


def extract_answer(text):
    if "Answer:" in text:
        answer = text.split("Answer:")[-1].strip()  # Get the part after "Answer:"
        return answer
    elif "[|assistant|]" in text:
        answer = text.split("[|assistant|]")[-1].strip()
        return answer
    return text.strip()


def main():
    start_time = datetime.datetime.now()

    hf_token = ""
    HfFolder.save_token(hf_token)

    # =========================================== Parameter Setup ===========================================
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)

    args = parser.parse_args()

    tokenizer1 = AutoTokenizer.from_pretrained(model1_id, cache_dir=own_cache_dir, trust_remote_code=True)
    model1 = AutoModelForCausalLM.from_pretrained(model1_id, cache_dir=own_cache_dir, torch_dtype=torch.float16, trust_remote_code=True).cuda()
    model1 = model1.eval()

    model2 = AutoModelForCausalLM.from_pretrained(
        model2_id,
        torch_dtype=torch.bfloat16,
        cache_dir=own_cache_dir,
        trust_remote_code=True,
        device_map="auto"
    )
    tokenizer2 = AutoTokenizer.from_pretrained(model2_id, cache_dir=own_cache_dir)

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
                inputs = tokenizer1([prompt1_1], return_tensors="pt")
                for k,v in inputs.items():
                    inputs[k] = v.cuda()
                gen_kwargs = {"max_length": 1024, "temperature": 0.3}
                output = model1.generate(**inputs, **gen_kwargs)
                generation1_1 = tokenizer1.decode(output[0].tolist(), skip_special_tokens=True)
                generation1_1 = parse_response(generation1_1, "Answer:")
                print(f">>> {generation1_1}")
                print("-----------------------------------------------------")

                # [2] model1 - stance + reason
                print("[2] model1 - Speak own opinion")
                prompt2_1 = prompts["prompt_1"].replace("{{country}}", country).replace("{{story}}", story).replace("{{rule}}", rule)
                print(prompt2_1)
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt2_1},
                ]
                input_ids = tokenizer2.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt"
                )
                output = model2.generate(
                    input_ids.to("cuda"),
                    eos_token_id=tokenizer2.eos_token_id,
                    max_new_tokens=1024,
                    do_sample=False,
                    temperature=0.0,
                )
                generated_text = tokenizer2.decode(output[0], skip_special_tokens=True)
                generation = extract_answer(generated_text)
                generation2_1 = generation
                generation2_1 = parse_response(generation2_1, "Answer:")
                print(f">>> {generation2_1}")
                print("-----------------------------------------------------")

                # [3] Model1 - give feedback to model1 stance + reason
                print("[3] Model1 - Give feedback to 2")
                prompt1_2 = prompts["prompt_2"].replace("{{country}}", country).replace("{{story}}", story).replace("{{rule}}", rule).replace("{{other_response}}", generation2_1).replace("{{your_response}}", generation1_1)
                print(prompt1_2)
                inputs = tokenizer1([prompt1_2], return_tensors="pt")
                for k,v in inputs.items():
                    inputs[k] = v.cuda()
                gen_kwargs = {"max_length": 1024, "temperature": 0.3}
                output = model1.generate(**inputs, **gen_kwargs)
                generation1_2 = tokenizer1.decode(output[0].tolist(), skip_special_tokens=True)
                generation1_2 = parse_response(generation1_2, "Response:")
                print(f">>> {generation1_2}")
                print("-----------------------------------------------------")

                # [4] model1 - give feedback to Model1 stance + reason
                print("[4] model1 - Give feedback to 1")
                prompt2_2 = prompts["prompt_2"].replace("{{country}}", country).replace("{{story}}", story).replace("{{rule}}", rule).replace("{{other_response}}", generation1_1).replace("{{your_response}}", generation1_2)
                print(prompt2_2)
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt2_2},
                ]
                input_ids = tokenizer2.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt"
                )
                output = model2.generate(
                    input_ids.to("cuda"),
                    eos_token_id=tokenizer2.eos_token_id,
                    max_new_tokens=1024,
                    do_sample=False,
                    temperature=0.0,
                )
                generated_text = tokenizer2.decode(output[0], skip_special_tokens=True)
                generation = extract_answer(generated_text)
                generation2_2 = generation
                generation2_2 = parse_response(generation2_2, "Response:")
                print(f">>> {generation2_2}")
                print("-----------------------------------------------------")

                print("[5] Model1 - Make final decision")
                prompt1_3 = prompts["prompt_3"].replace("{{country}}", country).replace("{{story}}", story).replace("{{rule}}", rule).replace("{{feedback}}", generation2_2).replace("{{your_feedback}}", generation1_2).replace("{{your_response}}", generation1_1).replace("{{other_response}}", generation2_1)
                print(prompt1_3)
                inputs = tokenizer1([prompt1_3], return_tensors="pt")
                for k,v in inputs.items():
                    inputs[k] = v.cuda()
                gen_kwargs = {"max_length": 1024, "temperature": 0.3}
                output = model1.generate(**inputs, **gen_kwargs)
                generation1_3 = tokenizer1.decode(output[0].tolist(), skip_special_tokens=True)
                generation1_3 = parse_final_answer(generation1_3)
                print(f">>> {generation1_3}")
                print("-----------------------------------------------------")

                print("[6] model1 - Make final decision")
                prompt2_3 = prompts["prompt_3"].replace("{{country}}", country).replace("{{story}}", story).replace("{{rule}}", rule).replace("{{feedback}}", generation1_2).replace("{{your_feedback}}", generation2_2).replace("{{your_response}}", generation2_1).replace("{{other_response}}", generation1_1)
                print(prompt2_3)
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt2_3},
                ]
                input_ids = tokenizer2.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt"
                )
                output = model2.generate(
                    input_ids.to("cuda"),
                    eos_token_id=tokenizer2.eos_token_id,
                    max_new_tokens=1024,
                    do_sample=False,
                    temperature=0.0,
                )
                generated_text = tokenizer2.decode(output[0], skip_special_tokens=True)
                generation = extract_answer(generated_text)
                generation2_3 = generation
                generation2_3 = parse_final_answer(generation2_3)
                print(f"> {generation2_3}")
                print("-----------------------------------------------------")

                print(f"Gold > {gold_label}")
                print(f"Internlm > {generation1_3}")
                print(f"Internlm > {generation2_3}")
                print("\n======================================================\n")
                generations1.append(generation1_3)
                generations2.append(generation2_3)

                line[f"internlm_1"] = generation1_1
                line[f"second_internlm_1"] = generation2_1
                line[f"internlm_2"] = generation1_2
                line[f"second_internlm_2"] = generation2_2
                line[f"internlm_final"] = generation1_3
                line[f"second_internlm_final"] = generation2_3
                outfile.write(line)
    
    end_time = datetime.datetime.now()
    print(f"Time elapsed: {end_time - start_time}")


if __name__ == "__main__":
    main()


