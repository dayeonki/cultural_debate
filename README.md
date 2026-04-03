<div align="center">

 # Multiple LLM Agents Debate for Equitable <br> Cultural Alignment

<p align="center">
  <img src="https://github.com/user-attachments/assets/74b6e33b-2eea-4aed-8db5-d6f89945c1b6" width="800">
</p>


<a href=https://dayeonki.github.io/>Dayeon Ki</a><sup>1</sup>, <a href=https://rudinger.github.io/>Rachel Rudinger</a><sup>1</sup>, <a href=https://tianyizhou.github.io/>Tianyi Zhou</a><sup>2</sup>, <a href=https://www.cs.umd.edu/~marine/>Marine Carpuat<a><sup>1</sup> <br>
<sup>1</sup>University of Maryland, <sup>2</sup>MBUZAI
<br>

This repository contains the code and dataset for our ACL 2025 Main paper <br> **Multiple LLM Agents Debate for Equitable Cultural Alignment**.

<p>
  <a href="https://aclanthology.org/2025.acl-long.1210/" target="_blank" style="text-decoration:none">
    <img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?style=flat&logo=arxiv" alt="arXiv">
  </a>
</p>

</div>

---

## 👾 TL;DR
While previous efforts in cultural alignment have focused on single-model, single-turn approaches, we propose to exploit the complementary strengths of multiple LLMs to promote cultural adaptability. We introduce a Multi-Agent Debate framework, where two LLM-based agents debate over a cultural scenario and collaboratively reach a final decision, which improves both (i) overall accuracy and (ii) cultural group parity over single-model baselines.

## 📰 News
- **`2025-07-10`** Our paper has been selected for an **oral** presentation — top 8% of accepted papers!
- **`2025-05-15`** Our paper is accepted to **ACL 2025**! See you in Vienna!


## ✏️ Content
- [🗺️ Overview](#overview)
- [🚀 Quick Start](#quick_start)
  - [Data Preparation](#data-preparation)
  - [Single LLM](#single-llm)
  - [Multiple LLM](#multiple-llm)
  - [Evaluation](#evaluation)
- [🤲 Citation](#citation)
- [📧 Contact](#contact)

---

<a id="overview"></a>
## 🗺️ Overview

How can multiple LLMs collaborate toward equitable alignment across cultures? We investigate a common form of multi-LLM collaboration: _debate_. We propose a **Multi-Agent Debate framework**, where two LLM agents debate over the given scenario and collaboratively arrive at a final decision with a judge LLM. We introduce two key variants as illustrated in the above figure:
1. **Debate-Only:** multiple LLM agents exclusively engage in debate with a discussant
2. **Self-Reflect+Debate:** each LLM agent dynamically choose between self-reflection and debating during its turn

For more comprehensive comparison study, we investigate two additional strategies based on single-LLM:

3. **Single Model:** a single LLM generates outputs
4. **Self-Reflection:** an LLM generates verbal self-reflections on its own outputs and incorporate them in subsequent iterations

### Results

<div align="center">
<img width="900" height="279" alt="Screenshot 2026-04-03 at 1 44 40 PM" src="https://github.com/user-attachments/assets/602d2b6b-59db-4715-aa8f-2895311f6f9c" />
</div>


<a id="quick_start"></a>
## 🚀 Quick Start

### Data Preparation

<div align="center">
<img width="850" height="489" alt="Screenshot 2026-04-03 at 1 20 57 PM" src="https://github.com/user-attachments/assets/4ece461f-3875-42ac-9571-cdb6eb3882df" />
</div>


We use [NORMAD-ETI](https://github.com/Akhila-Yerukola/NormAd) dataset for evaluation, a benchmark designed to assess the cultural adaptability of LLMs. The dataset contains 2.6K stories reflecting social and cultural norms from 75 countries, derived from the social-etiquette norms outlined in the [Cultural Atlas](https://culturalatlas.sbs.com.au/). Each story is associated with a country, a rule-of-thumb, and a ternary ground truth label in {Yes, No, Neither} as shown in the figure above. We categorize a total of 75 countries according to the Inglehart-Welzel cultural map and show the label and country distribution for each bin.

- Raw data: `data/normad_raw.csv`
- Country distribution: `data/normad_country_dist.csv`
- Refined data: `data/normad.jsonl`


### Single LLM
#### (1) Single Model
We first investigate the effect of adding relevant cultural context in enhancing cultural alignment of LLMs. We test two variants: without and with the rule-of-thumb (RoT) information in the prompts. (`single_llm/single_model/`)

For running **without** RoT prompting,

```bash
python -u sinlge_llm/single_model/{$LLM}.py \
  --input_path $PATH_TO_INPUT_FILE \
  --output_path $PATH_TO_OUTPUT_FILE \
  --type without_rot
```

For running **with** RoT prompting,

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


#### (2) Self-Reflection
Building on previous works that showed that LLMs can evaluate their outputs and learn from their own feedback, we explore self-reflection for each LLM. (`single_llm/self_reflection/`)

```bash
python -u sinlge_llm/self_reflection/{$LLM}.py \
  --input_path $PATH_TO_INPUT_FILE \
  --output_path $PATH_TO_OUTPUT_FILE \
```

Arguments for the prompting code are as follows:
  - `$LLM`: Name of the LLM (specific names can be found in the directory).
  - `--input_path`: Path to input data file (`data/normad.jsonl`).
  - `--output_path`: Save path of output file.



### Multiple LLM
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


### Evaluation

- For evaluating single LLM baselines, use `evaluate/accuracy_single.py`. Add the model names to test in the `MODEL_NAMES` variable and run the code: `python evaluate/accuracy_single.py`.

- For evaluating multi LLM baselines, use `evaluate/accuracy_multi.py`. Add the name of the first model as `FIRST_MODEL` and the name of the second model as `SECOND_MODEL` variables and run the code: `python evaluate/accuracy_multi.py`.


<a id="citation"></a>
## 🤲 Citation
If you find our work useful in your research, please consider citing our work:
```
@inproceedings{ki-etal-2025-multiple,
    title = "Multiple {LLM} Agents Debate for Equitable Cultural Alignment",
    author = "Ki, Dayeon  and
      Rudinger, Rachel  and
      Zhou, Tianyi  and
      Carpuat, Marine",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.1210/",
    doi = "10.18653/v1/2025.acl-long.1210",
    pages = "24841--24877",
    ISBN = "979-8-89176-251-0",
}
```

<a id="contact"></a>
## 📧 Contact
For questions, issues, or collaborations, please reach out to [dayeonki@umd.edu](mailto:dayeonki@umd.edu).
