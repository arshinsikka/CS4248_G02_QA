from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import json

# Paths
model_path = "../models/roberta_finetuned"
dev_file = "../data/dev-v1.1.json"
predictions_file = "../predictions.json"

# Load model
print("🔄 Loading fine-tuned model...")
qa_pipeline = pipeline("question-answering", model=model_path, tokenizer=model_path, device=-1)
print("✅ Model loaded successfully!")

# Load dev set
print("📚 Loading SQuAD dev dataset...")
with open(dev_file, "r") as f:
    squad_dict = json.load(f)

predictions = {}
for article in squad_dict["data"]:  # test with 5 articles first
    for paragraph in article["paragraphs"]:
        context = paragraph["context"]
        for qa in paragraph["qas"]:
            question = qa["question"]
            qid = qa["id"]

            result = qa_pipeline({"context": context, "question": question})
            predictions[qid] = result["answer"]

print(f"✅ Generated {len(predictions)} predictions.")
with open(predictions_file, "w") as f:
    json.dump(predictions, f, indent=2)
print(f"💾 Saved predictions to {predictions_file}")
