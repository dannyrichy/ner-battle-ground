import argparse
from span_marker import SpanMarkerModel, Trainer
from transformers import TrainingArguments
from datasets import load_dataset
from utils import filter_subclasses, DATASET


def _inf(model, dataset):
    training_args = TrainingArguments("test_trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
    )

    # The metrics returned are Precision, Recall and F1 Score that is evaluated using the seqeval method from huggingface
    metrics = trainer.evaluate()
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_type", help="Model type", choices=['a','b'], default='a')
    args=parser.parse_args()

    test_dataset = load_dataset(DATASET, split="test")
    test_dataset = test_dataset.filter(lambda x: x["lang"] == "en").select(range(5))
    # test_dataset = test_dataset.map(filter_subclasses)

    # Load the model from local
    model_a = SpanMarkerModel.from_pretrained(f"models/sys_{args.model_type}/checkpoint-final")
    metrics = _inf(model_a, test_dataset if args.model_type == 'b' else test_dataset.map(filter_subclasses))
    print(metrics)
