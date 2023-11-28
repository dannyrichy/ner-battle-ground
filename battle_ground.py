from datasets import load_dataset
from transformers import TrainingArguments

from span_marker import SpanMarkerModel, Trainer

if __name__ == '__main__':
    dataset = "Babelscape/multinerd"
    train_dataset = load_dataset(dataset, split="train")
    train_dataset = train_dataset.filter(lambda x:x['lang'] == 'en')    
    eval_dataset = load_dataset(dataset, split="validation")
    eval_dataset = eval_dataset.filter(lambda x:x['lang'] == 'en').shuffle().select(range(3000))
    labels = [
        "O",
        "B-PER",
        "I-PER",
        "B-ORG",
        "I-ORG",
        "B-LOC",
        "I-LOC",
        "B-ANIM",
        "I-ANIM",
        "B-BIO",
        "I-BIO",
        "B-CEL",
        "I-CEL",
        "B-DIS",
        "I-DIS",
        "B-EVE",
        "I-EVE",
        "B-FOOD",
        "I-FOOD",
        "B-INST",
        "I-INST",
        "B-MEDIA",
        "I-MEDIA",
        "B-MYTH",
        "I-MYTH",
        "B-PLANT",
        "I-PLANT",
        "B-TIME",
        "I-TIME",
        "B-VEHI",
        "I-VEHI",
    ]

    # Initialize a SpanMarker model using a pretrained BERT-style encoder
    model_name = "bert-base-multilingual-cased"
    model = SpanMarkerModel.from_pretrained(
        model_name,
        labels=labels,
        # SpanMarker hyperparameters:
        model_max_length=256,
        marker_max_length=128,
        entity_max_length=8,
    )

    # Prepare the 🤗 transformers training arguments
    args = TrainingArguments(
        output_dir="models/model1",
        # Training Hyperparameters:
        learning_rate=5e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        # gradient_accumulation_steps=2,
        num_train_epochs=1,
        weight_decay=0.01,
        warmup_ratio=0.1,
        bf16=True,  # Replace `bf16` with `fp16` if your hardware can't use bf16.
        # Other Training parameters
        logging_first_step=True,
        logging_steps=50,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=1000,
        save_total_limit=2,
        dataloader_num_workers=2,
    )

    # Initialize the trainer using our model, training args & dataset, and train
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    trainer.save_model("models/span_marker_mbert_base_multinerd/checkpoint-final")

    test_dataset = load_dataset(dataset, split="test")
    test_dataset = test_dataset.filter(lambda x:x['lang'] == 'en')
    # Compute & save the metrics on the test set
    metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
    trainer.save_metrics("test", metrics)

    trainer.create_model_card(language="english", license="apache-2.0")