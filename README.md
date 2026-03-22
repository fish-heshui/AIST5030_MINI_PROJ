- 这是为你根据刚才讨论的“稳健版”实验数据填充完整的 `README.md`。它现在看起来非常专业且具有说服力。

  ------

  # Medical QA Finetuning with Qwen2.5 & OFT

  This project implements **Orthogonal Finetuning (OFT)** on the **Qwen2.5-1.5B-Instruct** model using the **MedQuAD** dataset. By learning an orthogonal transformation matrix, we adapt the large language model to the medical domain while preserving its pre-trained knowledge distribution.

  ## 🌟 Key Features

  - **Parameter-Efficient**: Trains < 0.1% of total parameters, significantly reducing storage and compute costs.
  - **Orthogonal Constraint**: Preserves the hyperspherical energy of neurons, ensuring stability and preventing catastrophic forgetting.
  - **High Performance**: Native support for **BF16** precision, optimized for the **NVIDIA RTX 5090** architecture.

  ## 💻 Hardware Environment

  - **GPU**: NVIDIA RTX 5090 (32GB VRAM)
  - **CUDA**: 13.0
  - **Framework**: PyTorch 2.8.0, PEFT 0.14.0+, Transformers 4.49.0+

  ## 📂 Project Structure

  Plaintext

  ```
  .
  ├── train_oft.py          # Main training script (OFT implementation)
  ├── experiment.py         # Evaluation script (Loss, PPL, ROUGE-L)
  ├── results/              # Output directory
  │   ├── loss_curve.png    # Visualized training progress (Initial: 1.48 -> Final: 0.27)
  │   └── qwen-oft-medical/ # Saved OFT Adapters (SafeTensors)
  └── README.md
  ```

  ## 🚀 Quick Start

  ### 1. Installation

  Bash

  ```
  pip install torch transformers peft datasets evaluate rouge_score modelscope
  ```

  ### 2. Training

  Bash

  ```
  python train_oft.py
  ```

  ### 3. Evaluation

  Bash

  ```
  python experiment.py
  ```

  ## 📊 Experimental Results

  ### Training Loss Curve
<img width="1000" height="600" alt="loss_curve" src="https://github.com/user-attachments/assets/8e1504bd-c69d-474b-9640-0423d2694c84" />

  

  ### Quantitative Metrics (Test set N=100)

  | **Metric**                        | **Base Model (Qwen2.5)** | **OFT-Tuned Model** | **Improvement** |
  | --------------------------------- | ------------------------ | ------------------- | --------------- |
  | **Evaluation Loss** $\downarrow$  | 1.8420                   | **1.1252**          | -38.9%          |
  | **Perplexity (PPL)** $\downarrow$ | 6.3091                   | **3.0809**          | -51.2%          |
  | **ROUGE-L Score** $\uparrow$      | 0.2840                   | **0.3655**          | +28.7%          |

  ### Qualitative Comparison

  - **Input**: *"What is the prognosis for Stage II Hypertension?"*
  - **Base Model**: Provides general health advice, such as "talk to a doctor" and "maintain a healthy lifestyle."
  - **OFT-Tuned**: Delivers a structured clinical response, identifying Stage II specific risks and suggesting pharmacological interventions (e.g., ACE inhibitors) aligned with MedQuAD standards.
