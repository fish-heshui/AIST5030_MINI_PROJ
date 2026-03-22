import torch
import numpy as np
from rouge_score import rouge_scorer
import logging
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm

HF_TOKEN = "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"

class QwenEvaluator:
    def __init__(
        self,
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        adapter_path="./results/qwen-oft-medical",
        dataset_id="keivalya/MedQuad-MedicalQnADataset",
        num_eval_samples=10
    ):
        # 日志设置
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
        )
        self.logger = logging.getLogger(self.__class__.__name__)

        self.model_id = model_id
        self.adapter_path = adapter_path
        self.dataset_id = dataset_id
        self.num_eval_samples = num_eval_samples

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Device set to {self.device}")

        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.test_samples = self._prepare_data()

    def _prepare_data(self):
        self.logger.info("准备测试数据集 ...")
        dataset = load_dataset(self.dataset_id, split="train", token=HF_TOKEN)
        if len(dataset) < self.num_eval_samples:
            raise ValueError("数据集不足用于评估！")
        test_samples = dataset.select(
            range(len(dataset) - self.num_eval_samples, len(dataset))
        )
        self.logger.info(f"选取最后{self.num_eval_samples}条作为验证样本")
        return test_samples

    def evaluate(self, model, name="Model"):
        self.logger.info(f"评估中: {name} ...")
        total_loss = 0
        rouge_scores = []
        predictions, references = [], []
        model.eval()
        
        start_time = time.time()
        total_samples = len(self.test_samples)
        
        for i, item in enumerate(self.test_samples):
            sample_start = time.time()
            question = item['Question']
            reference_answer = item['Answer']

            # 构造输入
            prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
            full_text = prompt + reference_answer + "<|im_end|>"

            inputs = self.tokenizer(
                full_text, return_tensors="pt",
                truncation=True, max_length=512
            ).to(self.device)
            labels = inputs["input_ids"].clone()

            with torch.no_grad():
                # Loss & Perplexity
                outputs = model(**inputs, labels=labels)
                total_loss += outputs.loss.item()

                # 文本生成（用于ROUGE）
                input_ids = self.tokenizer(
                    prompt, return_tensors="pt"
                ).to(self.device)
                gen_out = model.generate(
                    **input_ids, max_new_tokens=150,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                generated_text = self.tokenizer.decode(
                    gen_out[0], skip_special_tokens=True
                ).split("assistant\n")[-1]
                predictions.append(generated_text)
                references.append(reference_answer)
                
                # 计算单条 ROUGE-L
                score = self.scorer.score(reference_answer, generated_text)
                rouge_scores.append(score['rougeL'].fmeasure)

            sample_time = time.time() - sample_start
            elapsed = time.time() - start_time
            eta = (elapsed / (i + 1)) * (total_samples - i - 1)
            
            # 打印进度
            self.logger.info(
                f"[{name}] 进度: {i+1}/{total_samples} | "
                f"本条耗时: {sample_time:.1f}s | "
                f"已用: {elapsed:.1f}s | "
                f"预计剩余: {eta:.1f}s"
            )

            # 展示第一条定性对比
            if i == 0:
                log_msg = (
                    f"\n[定性样例 - {name}]\n"
                    f"提问: {question}\n"
                    f"回答: {generated_text[:200]}..."
                )
                self.logger.info(log_msg)

        avg_loss = total_loss / len(self.test_samples)
        perplexity = np.exp(avg_loss)
        avg_rouge = np.mean(rouge_scores)

        self.logger.info(f"{name} Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}, ROUGE-L: {avg_rouge:.4f}")
        return {
            "Loss": avg_loss,
            "Perplexity": perplexity,
            "ROUGE-L": avg_rouge
        }

    def run(self):
        # 评估基线模型
        self.logger.info(f"加载基础模型: {self.model_id}")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=torch.float16, device_map="auto"
        )
        results_before = self.evaluate(base_model, name="Base Model")

        # 加载OFT模型
        self.logger.info(f"加载OFT微调模型: {self.adapter_path}")
        model_oft = PeftModel.from_pretrained(base_model, self.adapter_path)
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