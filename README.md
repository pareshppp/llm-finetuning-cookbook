# LLM Finetuning Cookbook series.

The plan is designed to be modular and incremental. Each part builds on the last, introducing new concepts, frameworks, and complexities in a logical order.

### Overarching Philosophy

1.  **Start Simple, Build Complexity:** The series begins with the most common task (SFT with LoRA) using the most common library (TRL) and progressively introduces more advanced techniques.
2.  **Consistency for Comparison:** For the initial set of notebooks (Parts 1-3), use the same base model (e.g., `mistralai/Mistral-7B-Instruct-v0.2` or a smaller model like `TinyLlama/TinyLlama-1.1B-Chat-v1.0`) and the same dataset (e.g., a subset of `HuggingFaceH4/ultrachat_200k` or `mlabonne/guanaco-llama2-1k`) to make performance and resource differences between frameworks and techniques clear.
3.  **Explain the "Why":** Each notebook should have extensive markdown cells explaining *why* certain choices are made (e.g., "Why use LoRA?", "What is a preference dataset?", "Why is QAT different from PTQ?").
4.  **Practical and Runnable:** Every notebook should be a self-contained, end-to-end example, from data loading to a simple inference test.

---

### Planned Series Structure & Notebook Plan

**Repository Structure:**
```
/llm-finetuning-cookbook
|-- Part_0_Introduction_and_Setup
|   |-- 00_Introduction_and_Setup.ipynb
|   |-- 00b_Data_Curation_and_Formatting.ipynb
|-- Part_1_Supervised_Finetuning_SFT
|   |-- 01_SFT_with_TRL_and_LoRA.ipynb
|   |-- 01b_SFT_for_Structured_Output_JSON.ipynb
|   |-- 02_Full_Parameter_SFT_with_TRL.ipynb
|   |-- 03_Faster_SFT_with_Unsloth.ipynb
|-- Part_2_Preference_Alignment
|   |-- 04_Aligning_with_DPO_using_TRL.ipynb
|   |-- 05_Simplified_Alignment_with_ORPO.ipynb
|   |-- 06_Advanced_RL_Alignment_with_GRPO.ipynb
|-- Part_3_Advanced_Finetuning_with_Axolotl
|   |-- 07_Reproducible_Finetuning_with_Axolotl.ipynb
|   |-- 08_Continued_Pretraining_with_Axolotl.ipynb
|-- Part_4_Quantization
|   |-- 09_Post_Training_Quantization_for_GGUF_llama_cpp.ipynb
|   |-- 10_Advanced_PTQ_with_AutoRound.ipynb
|   |-- 11_Quantization_Aware_Training_QAT.ipynb
|-- Part_5_Deployment_with_vLLM
|   |-- 12_Deploying_a_Merged_Model_with_vLLM.ipynb
|   |-- 13_Deploying_with_LoRA_Adapters_vLLM.ipynb
|   |-- 14_Deploying_Quantized_Models_vLLM.ipynb
|-- Part_6_Evaluation
|   |-- 15_Benchmarking_with_LM_Evaluation_Harness.ipynb
|-- Part_7_Advanced_Architectures
|   |-- 16_Finetuning_a_Mixture_of_Experts_MoE.ipynb
|   |-- 17_Finetuning_a_Vision_Language_Model_VLM.ipynb
|   |-- 18_Merging_LoRA_Adapters.ipynb
|-- Appendix
|   |-- Debugging.md
|   |-- Common_Pitfalls.ipynb
|-- /scripts
|-- /data
|-- requirements.txt
|-- README.md
```

---

### Detailed Notebook Content

### **Part 0: Introdcution and Setup**
*Get Started.*

#### `00_Introduction_and_Setup.ipynb`
*   **Objective:** Set the stage for the series.
*   **Content:**
    *   Overview of the cookbook structure.
    *   Explanation of key concepts: Finetuning, LoRA, Quantization, SFT, DPO, etc.
    *   Instructions for setting up the environment: NVIDIA drivers, CUDA, `conda`/`venv`.
    *   How to install core libraries (`transformers`, `torch`, `accelerate`, `bitsandbytes`).
    *   Setting up Hugging Face Hub credentials.

---

### **Part 1: Supervised Finetuning (SFT)**
*The foundation of adapting an LLM to a specific style or task.*

#### `01_SFT_with_TRL_and_LoRA.ipynb`
*   **Objective:** Your first finetune. Learn the most common and resource-efficient method.
*   **Key Concepts:** LoRA, PEFT, `SFTTrainer`, ChatML/Instruction formatting.
*   **Steps:**
    1.  Load a dataset (e.g., `mlabonne/guanaco-llama2-1k`).
    2.  Format the dataset into a consistent prompt structure.
    3.  Load a base model and tokenizer in 4-bit using `bitsandbytes`.
    4.  Define a `LoraConfig` from PEFT. Explain parameters like `r`, `lora_alpha`, `target_modules`.
    5.  Instantiate the `SFTTrainer` from TRL.
    6.  Run the training.
    7.  Perform inference with the adapter.
    8.  **Crucially:** Merge the LoRA adapter with the base model and save the final, merged model.

#### `02_Full_Parameter_SFT_with_TRL.ipynb`
*   **Objective:** Understand the trade-offs of full finetuning.
*   **Key Concepts:** Full finetuning vs. LoRA, resource requirements (VRAM, time).
*   **Steps:**
    1.  Use the same dataset and base model as Notebook 01.
    2.  This time, do not use a `LoraConfig`.
    3.  Highlight the massive increase in trainable parameters.
    4.  Discuss the hardware needed (e.g., multiple A100s). Use DeepSpeed/FSDP for memory optimization.
    5.  Run the training (or a few steps of it) and compare results/resource usage with the LoRA version.

#### `03_Faster_SFT_with_Unsloth.ipynb`
*   **Objective:** Introduce a highly optimized framework for massive speed and memory improvements.
*   **Key Concepts:** Unsloth's custom kernels, Triton language.
*   **Steps:**
    1.  Replicate the SFT + LoRA training from Notebook 01 using Unsloth's `FastLanguageModel`.
    2.  Show how the code is almost identical but requires minimal changes.
    3.  Benchmark and explicitly compare the training time and VRAM usage against the TRL version. Explain *why* it's faster.

---

### **Part 2: Preference Alignment**
*Teaching the model what humans prefer, moving beyond simple instruction-following.*

#### `04_Aligning_with_DPO_using_TRL.ipynb`
*   **Objective:** Fine-tune a model based on preference pairs (chosen vs. rejected).
*   **Key Concepts:** Preference tuning, DPO algorithm, preference dataset format (`prompt`, `chosen`, `rejected`).
*   **Steps:**
    1.  Start with the SFT model from Part 1.
    2.  Load a preference dataset (e.g., `HuggingFaceH4/ultrafeedback_binarized`).
    3.  Explain the structure of the dataset.
    4.  Use TRL's `DPOTrainer`.
    5.  Run the DPO training (can also be done with LoRA).
    6.  Show qualitative examples of how the model's responses change to be more helpful/less harmful.

#### `05_Simplified_Alignment_with_ORPO.ipynb`
*   **Objective:** Showcase a newer, simpler alignment technique that combines SFT and preference tuning.
*   **Key Concepts:** ORPO, combining instruction and preference loss.
*   **Steps:**
    1.  Explain how ORPO differs from SFT+DPO by using a single dataset and a single training stage.
    2.  Use a dataset formatted for ORPO (containing both the instruction and a single "chosen" response).
    3.  Use the `ORPO` trainer from TRL.
    4.  Compare the simplicity and results with the two-stage SFT->DPO process.

#### `06_Advanced_RL_Alignment_with_GRPO.ipynb`
*   **Objective:** Introduce a cutting-edge method that can handle K-wise comparisons.
*   **Key Concepts:** Reinforcement Learning with Verbalized Rewards (RLVR), k-wise preferences.
*   **Steps:**
    1.  Explain the limitation of DPO (pairwise) and how RLVR/GRPO handles ranked lists of responses.
    2.  Prepare a dataset with more than two preferences.
    3.  Use the `GRP-trainer` library to perform the alignment.
    4.  Frame this as an advanced, research-oriented technique.

---

### **Part 3: Advanced Finetuning with Axolotl**
*A powerful, configuration-driven framework for complex and reproducible pipelines.*

#### `07_Reproducible_Finetuning_with_Axolotl.ipynb`
*   **Objective:** Demonstrate how to manage complex finetuning pipelines using a single YAML config file.
*   **Key Concepts:** YAML configuration, reproducibility, multi-dataset handling.
*   **Steps:**
    1.  Introduce Axolotl and its philosophy.
    2.  Create a YAML config file that replicates the SFT + LoRA training from Notebook 01.
    3.  Create another YAML file that replicates the DPO training from Notebook 04.
    4.  Show how to mix datasets, use different LoRA types (like DoRA), and manage all hyperparameters from one place.

#### `08_Continued_Pretraining_with_Axolotl.ipynb`
*   **Objective:** Teach the model new knowledge by continuing the pre-training phase on new text data.
*   **Key Concepts:** Continued pre-training vs. finetuning, causal language modeling objective.
*   **Steps:**
    1.  Prepare a large, domain-specific text corpus (e.g., medical texts, legal documents).
    2.  Use an Axolotl YAML config for a pre-training run.
    3.  Explain the differences in configuration (dataset type, no prompt formatting).
    4.  Run the training and show how the model gains knowledge in the new domain.

---

### **Part 4: Quantization**
*Making models smaller, faster, and more efficient for deployment.*

#### `09_Post_Training_Quantization_for_GGUF_llama_cpp.ipynb`
*   **Objective:** Convert a trained model to the popular GGUF format for CPU and efficient GPU inference.
*   **Key Concepts:** PTQ, GGUF, `llama.cpp`.
*   **Steps:**
    1.  Take a merged, finetuned model from a previous notebook.
    2.  Use the `llama.cpp` Python bindings or command-line tools to convert the Hugging Face model to GGUF.
    3.  Show how to quantize to different levels (e.g., Q4_K_M, Q8_0) and explain the trade-offs.
    4.  Run inference with the quantized GGUF model using `llama-cpp-python`.

#### `10_Advanced_PTQ_with_AutoRound.ipynb`
*   **Objective:** Introduce a more sophisticated PTQ algorithm for better performance preservation.
*   **Key Concepts:** AutoRound algorithm, preserving perplexity.
*   **Steps:**
    1.  Explain the limitations of simple rounding-based quantization.
    2.  Introduce AutoRound as a method that optimizes the quantization process.
    3.  Use the `AutoRound` library to quantize a finetuned model to 4-bit.
    4.  Compare its performance (e.g., using perplexity) against a model quantized with a simpler method.

#### `11_Quantization_Aware_Training_QAT.ipynb`
*   **Objective:** Train a model that is "aware" it will be quantized, leading to higher accuracy.
*   **Key Concepts:** QAT vs. PTQ, fake quantization during training.
*   **Steps:**
    1.  Explain the concept of simulating quantization effects during the training loop.
    2.  Use a library that supports QAT (e.g., `quanto` or specific PEFT configurations).
    3.  Perform a LoRA finetune with QAT enabled.
    4.  Compare the final quantized model's performance to a PTQ'd model.

---

### **Part 5: Deployment with vLLM**
*Serving models for high-throughput, low-latency inference.*

#### `12_Deploying_a_Merged_Model_with_vLLM.ipynb`
*   **Objective:** Serve a standard, full-precision finetuned model using vLLM.
*   **Key Concepts:** PagedAttention, continuous batching, OpenAI-compatible API server.
*   **Steps:**
    1.  Take a merged model from Part 1 or 2.
    2.  Write a simple Python script to load the model with vLLM.
    3.  Launch the vLLM API server.
    4.  Use `curl` or the `openai` Python library to send requests and benchmark throughput.

#### `13_Deploying_with_LoRA_Adapters_vLLM.ipynb`
*   **Objective:** Serve multiple LoRA adapters on a single base model for efficient multi-tenant deployment.
*   **Key Concepts:** LoRARequest, multi-tenant serving.
*   **Steps:**
    1.  Assume you have multiple LoRA adapters (e.g., one for "coding assistant", one for "creative writer").
    2.  Launch the vLLM server with the *base model*.
    3.  Show how to make API requests, specifying which LoRA adapter to use for each request.
    4.  Explain the massive efficiency gain of this approach.

#### `14_Deploying_Quantized_Models_vLLM.ipynb`
*   **Objective:** Serve quantized models (e.g., AWQ, GPTQ) for even better performance.
*   **Key Concepts:** vLLM's quantization support.
*   **Steps:**
    1.  First, create a quantized version of a model using a supported format like AWQ or GPTQ.
    2.  Launch the vLLM server, specifying the `quantization` parameter.
    3.  Benchmark the latency, throughput, and VRAM usage compared to the full-precision model.

---

### **Part 6: Evaluation**
*Quantitatively measuring if your finetuning was successful.*

#### `15_Benchmarking_with_LM_Evaluation_Harness.ipynb`
*   **Objective:** Evaluate your models on standard academic benchmarks.
*   **Key Concepts:** MMLU, Arc, HellaSwag, TruthfulQA, evaluation metrics.
*   **Steps:**
    1.  Introduce the `lm-evaluation-harness` library.
    2.  Run an evaluation on the *base model* to get a baseline score.
    3.  Run the same evaluation on your SFT model and your DPO model.
    4.  Compare the scores and discuss how different finetuning stages affect different benchmarks.

---

### **Part 7: Advanced Architectures**
*Applying the cookbook's techniques to more complex models.*

#### `16_Finetuning_a_Mixture_of_Experts_MoE.ipynb`
*   **Objective:** Adapt the finetuning process for MoE models like Mixtral.
*   **Key Concepts:** MoE architecture, experts, routers, router loss.
*   **Steps:**
    1.  Explain the basics of an MoE model.
    2.  Perform an SFT with LoRA on a model like `mistralai/Mixtral-8x7B-Instruct-v0.1`.
    3.  Discuss MoE-specific considerations:
        *   Which LoRA modules to target (gates vs. experts).
        *   The importance of monitoring the router loss to prevent expert specialization collapse.
    4.  Use a framework like Unsloth or Axolotl, which have excellent support for MoE.

#### `17_Finetuning_a_Vision_Language_Model_VLM.ipynb`
*   **Objective:** Finetune a model like LLaVA to understand images and text.
*   **Key Concepts:** VLM architecture, vision encoder, projector, multi-modal data.
*   **Steps:**
    1.  Explain the components of a VLM (e.g., CLIP vision tower, projection module, LLM base).
    2.  Prepare a multi-modal instruction dataset (image + text pairs).
    3.  Demonstrate the typical two-stage finetuning process:
        *   **Stage 1:** Freeze the vision encoder and LLM, and train only the projector module.
        *   **Stage 2:** Unfreeze the LLM (or apply LoRA) and do an end-to-end SFT on vision-language instructions.
    4.  Use a library like LLaVA's official repo or a TRL example for VLM finetuning.
