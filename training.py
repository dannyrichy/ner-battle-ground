from datasets import load_dataset
from transformers import TrainingArguments

from span_marker import SpanMarkerModel, Trainer
import argparse
from utils import LABELS_SYS_A, LABELS_SYS_B, DATASET, filter_subclasses

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_type", help="Model type", choices=["a", "b"], default="a"
    )
    args = parser.parse_args()

    train_dataset = load_dataset(DATASET, split="train")
    train_dataset = train_dataset.filter(lambda x: x["lang"] == "en")
    eval_dataset = load_dataset(DATASET, split="validation")
    eval_dataset = (
        eval_dataset.filter(lambda x: x["lang"] == "en").shuffle().select(range(3000))
    )
    if args.model_type == "b":
        train_dataset = train_dataset.map(filter_subclasses)
        eval_dataset = eval_dataset.map(filter_subclasses)

    # Initialize a SpanMarker model using a pretrained BERT-style encoder
    model_name = "bert-large-cased"
    model = SpanMarkerModel.from_pretrained(
        model_name,
        labels=LABELS_SYS_A if args.model_type == "a" else LABELS_SYS_B,
        # SpanMarker hyperparameters:
        model_max_length=256,
        marker_max_length=128,
        entity_max_length=8,
    )

    # Prepare the transformers training arguments
    args = TrainingArguments(
        output_dir=f"models/output/sys_{args.model_type}",
        # Training Hyperparameters:
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        # gradient_accumulation_steps=2,
        num_train_epochs=1,
        weight_decay=0.01,
        warmup_ratio=0.1,
        fp16=True,  # Replace `bf16` with `fp16` if your hardware can't use bf16.
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
    trainer.save_model(f"models/sys_{args.model_type}/checkpoint-final")

    test_dataset = load_dataset(DATASET, split="test")
    test_dataset = test_dataset.filter(lambda x: x["lang"] == "en")
    if args.model_type == 'b':
        test_dataset = test_dataset.map(filter_subclasses)
    # Compute & save the metrics on the test set
    metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
    trainer.save_metrics("test", metrics)

    trainer.create_model_card(language="english", license="apache-2.0")
