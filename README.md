# Cross-Representation Benchmarking of EHR Data for Clinical Outcome Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)

This repository contains the official code and benchmark results for the paper **"Cross-Representation Benchmarking of EHR Data for Clinical Outcome Prediction"**. 

It provides a unified pipeline to evaluate Electronic Health Records (EHR) across three distinct representation formats: **Multivariate Time-series**, **Event Streams**, and **Textual Event Streams** (for LLMs). The benchmark covers two major clinical settings: ICU (MIMIC-IV) and longitudinal care (EHRSHOT).

## 📂 Repository Structure

The repository is organized into specific modules to support the full pipeline from data processing to model evaluation:

### 1. `ehrshot-benchmark/`
This directory contains the updated source code for the EHRSHOT benchmark. 
* **Label Generation:** Updated logic for generating prediction labels.
* **Task Definitions:** Source code defining the specific clinical prediction tasks used in the event stream representation generation.

### 2. Event Stream Notebooks (MIMIC-IV & EHRSHOT)
Located in their respective folders, these Jupyter notebooks handle the end-to-end workflow for event stream representations:
* **Conversion:** Scripts to convert datasets from **OMOP** format into the **Event Stream** representation.
* **Training & Eval:** Pipelines to train and evaluate event stream models (e.g., CLMBR, Count-based models).

### 3. Multivariate Time-Series Evaluation
Scripts and pipelines dedicated to evaluating standard time-series models (Transformer, LSTM, RETAIN, MLP) on the benchmark tasks.

## 🛑 Prerequisites & Data Access

**Important:** Due to patient privacy requirements, this repository **does not** contain raw EHR data. You must acquire the data and set up the environment as follows:

1.  **Data Access:**
    * **MIMIC-IV:** Request access via [PhysioNet](https://physionet.org/content/mimiciv/).
    * **EHRSHOT:** Request access via the [official EHRSHOT portal](https://ehrshot.stanford.edu/) (requires STARR dataset access).

2.  **Data Standardization (MEDS):**
    * The pipeline expects data to be converted into the **Medical Event Data Standard (MEDS)** format.
    * Ensure your raw data is first mapped to the OMOP CDM, and then converted to MEDS before running the notebooks in this repository.

## 📊 Benchmark Results

Below is the performance comparison across the three EHR data representations on MIMIC-IV and EHRSHOT datasets.

* **Metric:** F1 scores for LLMs are calculated from 1000 bootstraps.
* **Averaging:** ICU Phenotyping metrics are macro-averaged across 25 classes.
* **Settings:** Few-shot models are trained on 16 samples.
* **Best:** Bold font marks the best result per column.

### (a) MIMIC-IV Results (ICU Setting)

| Representation | Model | ICU Mortality (AUROC) | ICU Mortality (AUPRC) | ICU Mortality (F1) | ICU Phenotyping (AUROC) | ICU Phenotyping (AUPRC) | ICU Phenotyping (F1) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Multivariate Time-Series** | Transformer | 0.806 | 0.264 | 0.244 | 0.700 | 0.370 | 0.401 |
| | MLP | 0.806 | 0.289 | 0.248 | 0.680 | 0.342 | 0.387 |
| | LSTM | 0.794 | 0.291 | 0.246 | 0.691 | 0.356 | 0.398 |
| | RETAIN | 0.790 | 0.302 | 0.251 | 0.686 | 0.349 | 0.392 |
| **Event Stream** | Count (few-shot) | 0.530 | 0.074 | 0.112 | 0.553 | 0.248 | 0.250 |
| | CLMBR (few-shot)| 0.598 | 0.099 | 0.140 | 0.549 | 0.235 | 0.307 |
| | Count | 0.830 | 0.273 | 0.217 | **0.848** | **0.640** | **0.600** |
| | CLMBR | **0.857** | **0.330** | 0.241 | 0.782 | 0.504 | 0.493 |
| **Textual Event Stream** | GPT-OSS-20B | -- | -- | **0.254** | -- | -- | 0.256 |
| | Qwen3-8B-Thinking| -- | -- | 0.218 | -- | -- | 0.182 |
| | Llama3-8B | -- | -- | 0.137 | -- | -- | 0.184 |
| | DeepSeek-R1-8B | -- | -- | 0.135 | -- | -- | 0.218 |

### (b) EHRSHOT Results (Longitudinal Care)

| Representation | Model | 30-day Readmit (AUROC) | 30-day Readmit (AUPRC) | 30-day Readmit (F1) | 1-yr Pancreatic Cancer (AUROC) | 1-yr Pancreatic Cancer (AUPRC) | 1-yr Pancreatic Cancer (F1) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Multivariate Time-Series** | Transformer | 0.666 | 0.222 | 0.296 | 0.610 | 0.055 | 0.049 |
| | MLP | 0.611 | 0.164 | 0.233 | 0.614 | 0.080 | 0.061 |
| | LSTM | 0.616 | 0.164 | 0.239 | 0.622 | 0.058 | 0.065 |
| | RETAIN | 0.617 | 0.162 | 0.238 | 0.636 | 0.061 | 0.069 |
| **Event Stream** | Count (few-shot) | 0.679 | 0.208 | 0.266 | 0.674 | 0.077 | 0.074 |
| | CLMBR (few-shot)| **0.736** | **0.273** | **0.335** | 0.703 | 0.117 | 0.085 |
| | Count | 0.685 | 0.249 | 0.298 | **0.781** | **0.213** | **0.314** |
| | CLMBR | 0.691 | 0.225 | 0.278 | 0.706 | 0.103 | 0.163 |
| **Textual Event Stream** | GPT-OSS-20B | -- | -- | 0.232 | -- | -- | 0.179 |
| | Qwen3-8B-Thinking| -- | -- | 0.226 | -- | -- | 0.114 |
| | Llama3-8B | -- | -- | 0.209 | -- | -- | 0.045 |
| | DeepSeek-R1-8B | -- | -- | 0.211 | -- | -- | 0.128 |

## 🚀 Quick Start

1.  Clone the repository:
    ```bash
    git clone [https://github.com/BrandonC8310/EHR-Cross-Representation-Benchmarks.git](https://github.com/BrandonC8310/EHR-Cross-Representation-Benchmarks.git)
    cd EHR-Cross-Representation-Benchmarks
    ```

2.  Ensure you have obtained the necessary data permissions (MIMIC-IV / EHRSHOT) and converted your data to the MEDS format.

3.  Navigate to the specific folder for your desired representation (e.g., `ehrshot-benchmark/` for event streams or the multivariate time-series folder) and follow the specific instructions within.

## 📝 Citation

If you use this code or benchmark in your research, please cite:

```bibtex
@inproceedings{chen2026crossrepresentation,
  title={Cross-Representation Benchmarking in Time-Series Electronic Health Records for Clinical Outcome Prediction},
  author={Chen, Tianyi and Zhu, Mingcheng and Luo, Zhiyao and Zhu, Tingting},
  booktitle={2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2026},
  organization={IEEE}
}
