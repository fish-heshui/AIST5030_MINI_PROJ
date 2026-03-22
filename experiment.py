import torch
import numpy as np
from rouge_score import rouge_scorer
import logging
import time
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import Dataset
from tqdm import tqdm

# 本地缓存路径
LOCAL_MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen2___5-1___5B-Instruct"
LOCAL_DATASET_PATH = "/root/.cache/huggingface/datasets/keivalya___med_quad-medical_qn_a_dataset/default/0.0.0/5b0961fbaa6d7f9c344c5d59c29943fb900c2eca/med_quad-medical_qn_a_dataset-train.arrow"

class QwenEvaluator:
    def __init__(
        self,
        model_path=LOCAL_MODEL_PATH,
        adapter_path="./results/qwen-oft-medical",
        dataset_path=LOCAL_DATASET_PATH,
        num_eval_samples=100,
        batch_size=8
    ):
        # 日志设置
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
        )
        self.logger = logging.getLogger(self.__class__.__name__)

        self.model_path = model_path
        self.adapter_path = adapter_path
        self.dataset_path = dataset_path
        self.num_eval_samples = num_eval_samples
        self.batch_size = batch_size

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Device set to {self.device}")

        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.logger.info(f"从本地加载 tokenizer: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        self.test_samples = self._prepare_data()

    def _prepare_data(self):
        self.logger.info(f"从本地加载数据集: {self.dataset_path}")
        dataset = Dataset.from_file(self.dataset_path)
        self.logger.info(f"数据集大小: {len(dataset)}")
        
        if len(dataset) < self.num_eval_samples:
            raise ValueError("数据集不足用于评估！")
        test_samples = dataset.select(
            range(len(dataset) - self.num_eval_samples, len(dataset))
        )
        self.logger.info(f"选取最后 {self.num_eval_samples} 条作为验证样本")
        return test_samples

    def evaluate(self, model, name="Model"):
        self.logger.info(f"评估中: {name} ... (批量大小: {self.batch_size})")
        total_loss = 0
        rouge_scores = []
        predictions, references = [], []
        model.eval()
        
        start_time = time.time()
        total_samples = len(self.test_samples)
        num_batches = (total_samples + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(num_batches):
            batch_start = time.time()
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, total_samples)
            batch_items = [self.test_samples[i] for i in range(start_idx, end_idx)]
            
            # 准备批量数据
            prompts = []
            full_texts = []
            batch_references = []
            
            for item in batch_items:
                question = item['Question']
                reference_answer = item['Answer']
                prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
                full_text = prompt + reference_answer + "<|im_end|>"
                prompts.append(prompt)
                full_texts.append(full_text)
                batch_references.append(reference_answer)
            
            with torch.no_grad():
                # 批量计算 Loss
                inputs = self.tokenizer(
                    full_texts, return_tensors="pt",
                    truncation=True, max_length=512,
                    padding=True
                ).to(self.device)
                labels = inputs["input_ids"].clone()
                labels[inputs["attention_mask"] == 0] = -100
                
                outputs = model(**inputs, labels=labels)
                total_loss += outputs.loss.item() * len(batch_items)
                
                # 批量生成文本
                gen_inputs = self.tokenizer(
                    prompts, return_tensors="pt",
                    truncation=True, max_length=256,
                    padding=True
                ).to(self.device)
                
                gen_outputs = model.generate(
                    **gen_inputs,
                    max_new_tokens=100,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=False
                )
                
                # 解码生成的文本
                for i, gen_out in enumerate(gen_outputs):
                    generated_text = self.tokenizer.decode(
                        gen_out, skip_special_tokens=True
                    ).split("assistant\n")[-1]
                    predictions.append(generated_text)
                    references.append(batch_references[i])
                    
                    # 计算 ROUGE-L
                    score = self.scorer.score(batch_references[i], generated_text)
                    rouge_scores.append(score['rougeL'].fmeasure)
            
            batch_time = time.time() - batch_start
            elapsed = time.time() - start_time
            processed = end_idx
            eta = (elapsed / processed) * (total_samples - processed) if processed > 0 else 0
            
            # 每 25% 或最后一批输出进度
            progress_pct = processed / total_samples * 100
            if batch_idx == 0 or progress_pct % 25 < (self.batch_size / total_samples * 100) or batch_idx == num_batches - 1:
                self.logger.info(
                    f"[{name}] 进度: {progress_pct:.0f}% ({processed}/{total_samples}) | "
                    f"已用: {elapsed:.1f}s | "
                    f"预计剩余: {eta:.1f}s"
                )
            
            # 展示第一批的第一条定性对比
            if batch_idx == 0:
                log_msg = (
                    f"\n[定性样例 - {name}]\n"
                    f"提问: {batch_items[0]['Question']}\n"
                    f"参考答案: {batch_references[0]}\n"
                    f"模型回答: {predictions[0]}"
                )
                self.logger.info(log_msg)

        avg_loss = total_loss / total_samples
        perplexity = np.exp(avg_loss)
        avg_rouge = np.mean(rouge_scores)

        total_time = time.time() - start_time
        self.logger.info(
            f"{name} 完成! Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}, "
            f"ROUGE-L: {avg_rouge:.4f}, 总耗时: {total_time:.1f}s"
        )
        return {
            "Loss": avg_loss,
            "Perplexity": perplexity,
            "ROUGE-L": avg_rouge
        }

    def run(self):
        # 评估基线模型
        self.logger.info(f"从本地加载基础模型: {self.model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype=torch.float16, device_map="auto", local_files_only=True
        )
        results_before = self.evaluate(base_model, name="Base Model")

        # 加载OFT模型
        self.logger.info(f"加载OFT微调模型: {self.adapter_path}")
        model_oft = PeftModel.from_pretrained(base_model, self.adapter_path, local_files_only=True)
        results_after = self.evaluate(model_oft, name="OFT-Tuned Model")

        # 打印对比表格
        result_str = (
            "\n" + "=" * 50 + "\n"
            f"{'Metric':<15} | {'Before (Base)':<15} | {'After (OFT)':<15}\n"
            + "-" * 50 + "\n"
        )
        for metric in ["Loss", "Perplexity", "ROUGE-L"]:
            val_pre = results_before[metric]
            val_post = results_after[metric]
            result_str += f"{metric:<15} | {val_pre:<15.4f} | {val_post:<15.4f}\n"
        result_str += "=" * 50
        print(result_str)
        self.logger.info("评估完成。")

if __name__ == "__main__":
    evaluator = QwenEvaluator()
    evaluator.run()