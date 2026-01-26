import os
import torch
from PIL import Image
from torch.utils.data import Dataset as TorchDataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments

# ================= CONFIG =================
TRAIN_ROOT = "/home/sandyalzabadani/train"
MODEL_NAME = "microsoft/trocr-base-handwritten"
OUTPUT_DIR = "./trocr-finetuned-lecture"
EPOCHS = 3
BATCH_SIZE = 4
GRAD_ACCUM = 4
LR = 3e-5
MAX_LABEL_LENGTH = 128

# ================= DEVICE =================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ================= LOAD MODEL =================
processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size
model.to(device)

# ================= LAZY LOADING =================
class OCRDataset(TorchDataset):
    def __init__(self, train_root):
        self.samples = []
        for parent in os.listdir(train_root):
            parent_path = os.path.join(train_root, parent)
            if not os.path.isdir(parent_path):
                continue
            label_file = os.path.join(parent_path, "Id-Label.txt")
            if not os.path.exists(label_file):
                continue
            with open(label_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("Crops/train/"):
                        line = line.replace("Crops/train/", "")
                    img_rel_path, text = line.split(" ", 1)
                    img_path = os.path.join(train_root, img_rel_path)
                    if os.path.exists(img_path):
                        self.samples.append((img_path, text))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, text = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        labels = processor.tokenizer(text, padding="max_length", truncation=True, max_length=MAX_LABEL_LENGTH).input_ids
        labels = [(token if token != processor.tokenizer.pad_token_id else -100) for token in labels]
        return {"pixel_values": pixel_values, "labels": torch.tensor(labels)}

dataset = OCRDataset(TRAIN_ROOT)
print("Total samples:", len(dataset))

# ================= TRAINING ARGS =================
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    fp16=True,
    save_strategy="steps",
    save_steps=500,   # save every 500 steps
    logging_steps=100,
    save_total_limit=2,
    report_to="none",
    dataloader_num_workers=2,
    predict_with_generate=False,
    remove_unused_columns=False
)

# ================= TRAINER =================
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# ================= TRAIN =================
trainer.train()

# ================= SAVE =================
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

print("Training complete. Model saved to:", OUTPUT_DIR)

