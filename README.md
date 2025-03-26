# Multiple LLM Agents Debate for Equitable Cultural Alignment
This repository contains the code and dataset for our paper **Multiple LLM Agents Debate for Equitable Cultural Alignment**.


<div align="center">
<img src="https://github.com/user-attachments/assets/b3415a65-ccac-4468-a291-07602cb95509" style="width: 15px;" alt="code"> <b><a href=https://github.com/dayeonki/askqe>Code</a></b> | <img src="https://github.com/user-attachments/assets/2bd9af9b-2182-4aef-83cd-6e9ca6189a39" style="width: 15px;" alt="data">
 <b><a>Dataset</a></b> | <img src="https://github.com/user-attachments/assets/fc2ca3c2-3e78-4ca4-a208-448c0a6c7068" style="width: 15px;" alt="paper"> <b><a>Paper</a></b>
</div>



## Abstract
Large Language Models (LLMs) need to adapt their predictions to diverse cultural contexts to benefit diverse communities across the world. While previous efforts have focused on single-LLM, single-turn approaches, we propose to exploit the complementary strengths of multiple LLMs to promote cultural adaptability. We introduce a Multi-Agent Debate framework, where two LLM-based agents debate over a cultural scenario and collaboratively reach a final decision. We propose two variants: one where either LLM agents exclusively debate and another where they dynamically choose between self-reflection and debate during their turns. We evaluate these approaches on 7 open-weight LLMs (and 21 LLM combinations) using the NormAd-ETI benchmark for social etiquette norms in 75 countries. Experiments show that debate improves both overall accuracy and cultural group parity over single-LLM baselines. Notably, multi-agent debate enables relatively small LLMs (7-9B) to achieve accuracies comparable to that of a much larger model (27B parameters).

<p align="center">
  <img src="https://github.com/user-attachments/assets/74b6e33b-2eea-4aed-8db5-d6f89945c1b6" width="800">
</p>



## Quick Links
- [Overview](#overview)
- [Data](#data)
- [Single LLM](#single-llm)
- [Multiple LLM](#multiple-llm)
- [Evaluate](#evaluate)


## Overview
How can multiple LLMs collaborate toward equitable alignment across cultures? We investigate a common form of multi-LLM collaboration: _debate_. We propose a **Multi-Agent Debate framework**, where two LLM agents debate over the given scenario and collaboratively arrive at a final decision with a judge LLM. We introduce two key variants as illustrated in the above figure:
1. **Debate-Only:** multiple LLM agents exclusively engage in debate with a discussant
2. **Self-Reflect+Debate:** each LLM agent dynamically choose between self-reflection and debating during its turn

For more comprehensive comparison study, we investigate two additional strategies based on single-LLM:

3. **Single Model:** a single LLM generates outputs
4. **Self-Reflection:** an LLM generates verbal self-reflections on its own outputs and incorporate them in subsequent iterations


## Data
We use [NORMAD-ETI](https://github.com/Akhila-Yerukola/NormAd) dataset for evaluation, a benchmark designed to assess the cultural adaptability of LLMs. The dataset contains 2.6K stories reflecting social and cultural norms from 75 countries, derived from the social-etiquette norms outlined in the [Cultural Atlas](https://culturalatlas.sbs.com.au/). Each story is associated with a country, a rule-of-thumb, and a ternary ground truth label in {Yes, No, Neither} as shown in the figure above. We categorize a total of 75 countries according to the Inglehart-Welzel cultural map and show the label and country distribution for each bin.

- Raw data: `data/normad_raw.csv`
- Country distribution: `data/normad_country_dist.csv`
- Refined data: `data/normad.jsonl`

<p align="center">
<img width="650" alt="Screenshot 2025-03-21 at 2 51 32 PM" src="https://github.com/user-attachments/assets/fee2b7a7-b58e-423e-8d2a-dbe4c0810f62" />
</p>

## Single LLM
### [1] Single Model
We first investigate the effect of adding relevant cultural context in enhancing cultural alignment of LLMs. We test two variants: without and with the rule-of-thumb (RoT) information in the prompts. (`single_llm/single_model/`)

For running without RoT prompting,

```bash
python -u sinlge_llm/single_model/{$LLM}.py \
  --input_path $PATH_TO_INPUT_FILE \
  --output_path $PATH_TO_OUTPUT_FILE \
  --type without_rot
```

For running with RoT prompting,

```bash
python -u sinlge_llm/single_model/{$LLM}.py \
  --input_path $PATH_TO_INPUT_FILE \
  --output_path $PATH_TO_OUTPUT_FILE \
  --type with_rot
```

Arguments for the prompting code are as follows:
  - `$LLM`: Name of the LLM (specific names can be found in the directory).
  - `--input_path`: Path to input data file (`data/normad.jsonl`).
  - `--output_path`: Save path of output file.
  - `--type`: Without or with RoT information.


### [2] Self-Reflection
Also, building on previous works that showed that LLMs can evaluate their outputs and learn from their own feedback, we explore self-reflection for each LLM. (`single_llm/self_reflection/`)

```bash
python -u sinlge_llm/self_reflection/{$LLM}.py \
  --input_path $PATH_TO_INPUT_FILE \
  --output_path $PATH_TO_OUTPUT_FILE \
```

Arguments for the prompting code are as follows:
  - `$LLM`: Name of the LLM (specific names can be found in the directory).
  - `--input_path`: Path to input data file (`data/normad.jsonl`).
  - `--output_path`: Save path of output file.



## Multiple LLM
LLMs often exhibit varying knowledge coverage, with the potential to complement each other due to differences in training data distributions and alignment processes. We tap into this _knowledge complementarity_ through multi-LLM collaboration, debate, where two LLM-based agents debate and collaboratively evaluate the given scenario.

```bash
python -u multi_llm/{$FIRST_LLM}_$SECOND_LLM.py \
  --input_path $PATH_TO_INPUT_FILE \
  --output_path $PATH_TO_OUTPUT_FILE \
```

Arguments for the prompting code are as follows:
  - `$FIRST_LLM`: Name of the first participant LLM (specific names can be found in the directory).
  - `$SECOND_LLM`: Name of the second participant LLM (specific names can be found in the directory).
  - `--input_path`: Path to input data file (`data/normad.jsonl`).
  - `--output_path`: Save path of output file.


## Evaluate
For evaluating single LLM baselines, use `evaluate/accuracy_single.py`. Add the model names to test in the `MODEL_NAMES` variable and run the code: `python evaluate/accuracy_single.py`.

For evaluating multi LLM baselines, use `evaluate/accuracy_multi.py`. Add the name of the first model as `FIRST_MODEL` and the name of the second model as `SECOND_MODEL` variables and run the code: `python evaluate/accuracy_multi.py`.


## Citation
```
TBD
```
