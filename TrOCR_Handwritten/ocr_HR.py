import os
from PIL import Image
from datasets import Dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments

# ================= CONFIG =================
DATA_DIR = r"C:\Users\twing\OneDrive\Desktop\OCR_train\images"  # folder with nested subfolders
LABEL_FILE = r"C:\Users\twing\OneDrive\Desktop\OCR_train\labels\Id-Label.txt"
MODEL_NAME = "microsoft/trocr-base-handwritten"
OUTPUT_DIR = "./trocr-finetuned-lecture2"
EPOCHS = 2 #to accomodate CPU will change for GPU
BATCH_SIZE = 1 #to accomodate CPU will chnage for GPU
LR = 5e-5 #low learning rate for better tuning results

# ================= LOAD MODEL =================
processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

# Fix special tokens
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

# ================= READ LABEL FILE =================
# Format per line:
#   path/to/image.png  text label here
label_dict = {}

with open(LABEL_FILE, "r", encoding="utf-8") as f:
    for line in f:
        path, text = line.strip().split(" ", 1)
        filename = os.path.basename(path)  # only use last part
        label_dict[filename] = text

# ================= CREATE DATASET (Access each image in database) =================
dataset_items = []

for root, dirs, files in os.walk(DATA_DIR):
    for img_file in files:
        if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
            if img_file in label_dict:
                full_path = os.path.join(root, img_file)
                dataset_items.append({
                    "image": Image.open(full_path).convert("RGB"),
                    "text": label_dict[img_file]
                })

dataset = Dataset.from_list(dataset_items)


def preprocess_images(examples):
    # Convert images directly using the processor
    pixel_values = processor(
        images=examples["image"], 
        return_tensors="pt"
    ).pixel_values

    labels = processor.tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True
    ).input_ids

    return {"pixel_values": pixel_values, "labels": labels}

processed_dataset = dataset.map(preprocess_images, batched=True)

# ================= TRAINING ARGUMENTS =================
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    save_total_limit=2,
    logging_dir="./logs",
    predict_with_generate=True
)

# ================= TRAINER =================
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset
)

# ================= START TRAINING =================
trainer.train()

# ================= SAVE FINAL MODEL =================
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

print(f"\n Fine-tuned lecture handwritten model saved to {OUTPUT_DIR}")