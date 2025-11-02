from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
import json
import evaluate
import numpy as np

# 1️⃣ Load and flatten SQuAD files
def load_and_flatten_squad(path):
    with open(path, "r") as f:
        squad_dict = json.load(f)

    contexts, questions, answers = [], [], []
    for article in squad_dict["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                question = qa["question"]
                if len(qa["answers"]) > 0:
                    ans = qa["answers"][0]
                    answers.append({"text": ans["text"], "answer_start": ans["answer_start"]})
                else:
                    answers.append({"text": "", "answer_start": 0})
                contexts.append(context)
                questions.append(question)
    return Dataset.from_dict({"context": contexts, "question": questions, "answers": answers})

train_dataset = load_and_flatten_squad("../data/train-v1.1.json")
validation_dataset = load_and_flatten_squad("../data/dev-v1.1.json")
dataset = DatasetDict({"train": train_dataset, "validation": validation_dataset})

print("✅ Loaded dataset successfully. Example:")
print(dataset["train"][0])

# 2️⃣ Initialize tokenizer and model **before** using them
model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
import torch
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
print(f"✅ Using device: {device}")

# 3️⃣ Preprocess function (uses tokenizer now safely)
def preprocess(examples):
    questions = [q.strip() for q in examples["question"]]

    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = inputs.pop("overflow_to_sample_mapping")
    offset_mapping = inputs.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = inputs["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = inputs.sequence_ids(i)
        sample_index = sample_mapping[i]
        answer = examples["answers"][sample_index]

        # If no answer, CLS token
        if answer["text"] == "" or answer["answer_start"] == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        start_char = answer["answer_start"]
        end_char = start_char + len(answer["text"])

        # Find start and end token indices in the context
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1
        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        # If answer is not inside context
        if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            # Move token_start_index to the first token start >= start_char
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            start_positions.append(token_start_index - 1)
            # Move token_end_index to the last token end <= end_char
            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            end_positions.append(token_end_index + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


tokenized_datasets = dataset.map(preprocess, batched=True, remove_columns=dataset["train"].column_names)

# 4️⃣ Define training parameters
training_args = TrainingArguments(
    output_dir="../models/roberta_finetuned",
    do_eval=True,
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="./logs",
    save_total_limit=2,
)

# 5️⃣ Define metrics
metric = evaluate.load("squad")

def compute_metrics(p):
    return metric.compute(predictions=p.predictions, references=p.label_ids)

# 6️⃣ Train model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"].select(range(2000)),  # small subset for testing
    eval_dataset=tokenized_datasets["validation"].select(range(500)),
    tokenizer=tokenizer,
)

trainer.train()

# 7️⃣ Save the model
trainer.save_model("../models/roberta_finetuned")
print("✅ Fine-tuning complete! Model saved in models/roberta_finetuned/")
