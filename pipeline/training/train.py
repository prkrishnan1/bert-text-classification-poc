"""
Training script for Hugging Face SageMaker Estimator
"""
import logging
import sys
import argparse
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments, EvalPrediction
import torch
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--tokenizer_name", type=str)
    parser.add_argument("--learning_rate", type=str, default=2e-5)
    # parser.add_argument("--num_labels", type=int)

    # Data, model, and output directories
    parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--eval_dir", type=str, default=os.environ["SM_CHANNEL_EVAL"])
    args, _ = parser.parse_known_args()

    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # load datasets
    train_dataset = load_from_disk(args.training_dir)
    eval_dataset = load_from_disk(args.eval_dir)
    
    num_labels = len(train_dataset.unique("labels"))
    
    logger.info("loaded train_dataset length is: %s", len(train_dataset))
    logger.info("loaded test_dataset length is: %s", len(eval_dataset))

    def compute_metrics(p: EvalPrediction):
        predictions = p.predictions[0] if isinstance(p.predictions,
                                               tuple) else p.predictions
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(predictions))
        y_pred = torch.argmax(probs, dim=1)
        y_true = p.label_ids
        f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
        accuracy = accuracy_score(y_true, y_pred)
        # return as dictionary
        metrics = {'eval_f1': f1_micro_average,
                   'accuracy': accuracy}
        return metrics

    # download model and tokenizer from model hub
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    logger.info("Model and Tokenizer have been loaded!")
    # define training args
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=args.warmup_steps,
        evaluation_strategy="epoch",
        logging_dir=f"{args.output_data_dir}/logs",
        learning_rate=float(args.learning_rate),
    )

    # create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    # train model
    trainer.train()

    # evaluate model
    eval_result = trainer.evaluate(eval_dataset=eval_dataset)

    # writes eval result to file which can be accessed later in s3 output
    with open(os.path.join(args.output_data_dir, "eval_results.txt"), "w") as writer:
        logger.info("***** Eval results *****")
        for key, value in sorted(eval_result.items()):
            writer.write(f"{key} = {value}\n")

    # Saves the model to s3
    trainer.save_model(args.model_dir)
    logger.info("Model has been saved!")