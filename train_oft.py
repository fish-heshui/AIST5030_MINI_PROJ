import torch
import matplotlib.pyplot as plt
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import OFTConfig, get_peft_model
from modelscope import snapshot_download

HF_TOKEN = ""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

class OFTFineTuner:
    def __init__(
        self, 
        model_id="Qwen/Qwen2.5-1.5B-Instruct", 
        dataset_id="keivalya/MedQuad-MedicalQnADataset", 
        output_dir="./results/qwen-oft-medical", 
        max_steps=300,
        seed=42,
        data_size=1000,
        loss_curve_path="./results/loss_curve.png"  # 新增曲线图片路径到配置
    ):
        self.model_id = model_id
        self.dataset_id = dataset_id
        self.output_dir = output_dir
        self.max_steps = max_steps
        self.seed = seed
        self.data_size = data_size
        self.loss_curve_path = loss_curve_path  # 保存Loss曲线的路径

        self.tokenizer = None
        self.model = None
        self.tokenized_dataset = None
        self.trainer = None

    def load_tokenizer_and_model(self):
        logger.info("正在通过 ModelScope 极速下载模型...")
        # 指定 model_id 为 'qwen/Qwen2.5-1.5B-Instruct' (注意 ModelScope 的 ID 格式)
        # 如果 self.model_id 是 "Qwen/Qwen2.5-1.5B-Instruct"，ModelScope 也能识别
        model_dir = snapshot_download(self.model_id, revision='master')
        
        logger.info(f"模型下载成功，路径: {model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto"
        )
        logger.info("model loaded")

    def format_dataset(self, example):
        prompt = f"<|im_start|>user\n{example['Question']}<|im_end|>\n<|im_start|>assistant\n"
        answer = f"{example['Answer']}<|im_end|>"
        full_text = prompt + answer
        return self.tokenizer(full_text, truncation=True, max_length=512)

    def prepare_data(self):
        logger.info("Loading and processing dataset: %s", self.dataset_id)
        dataset = load_dataset(self.dataset_id, split="train", token=HF_TOKEN).shuffle(seed=self.seed).select(range(self.data_size))
        self.tokenized_dataset = dataset.map(self.format_dataset, remove_columns=dataset.column_names)
        logger.info("Dataset loaded and tokenized. Size: %d", self.data_size)

    def configure_oft(self):
        logger.info("Configuring OFT...")
        oft_config = OFTConfig(
            r=8,
            oft_block_size=0,
            eps=6e-5,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, oft_config)
        self.model.print_trainable_parameters()
        logger.info("OFT configured and trainable parameters printed.")

    def setup_trainer(self):
        logger.info("Setting up Trainer...")
        self.model.gradient_checkpointing_enable()
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            warmup_steps=20,
            max_steps=self.max_steps,
            learning_rate=5e-4,
            fp16=False,
            bf16=True,
            logging_steps=10,
            save_strategy="no",
            report_to="none",
            dataloader_num_workers=4
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dataset,
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        )
        logger.info("Trainer is set up.")

    def train_and_plot(self):
        logger.info("🚀 启动 OFT 训练...")
        train_result = self.trainer.train()

        # 绘制 Loss 曲线
        loss_history = [log['loss'] for log in self.trainer.state.log_history if 'loss' in log]
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history, label='Training Loss')
        plt.title('OFT Finetuning Loss Curve')
        plt.xlabel('Logging Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(self.loss_curve_path)  # 使用配置中的路径
        logger.info("✅ Loss 曲线已保存为 %s", self.loss_curve_path)

    def save_model(self):
        self.model.save_pretrained(self.output_dir)
        logger.info("模型权重已保存到 %s", self.output_dir)

    def run(self):
        self.load_tokenizer_and_model()
        self.prepare_data()
        self.configure_oft()
        self.setup_trainer()
        self.train_and_plot()
        self.save_model()

if __name__ == "__main__":
    finetuner = OFTFineTuner()
    finetuner.run()