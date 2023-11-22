import tarfile
import pathlib
import json
import logging
from transformers import Trainer, AutoModelForSequenceClassification, EvalPrediction
import torch
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    
    model_path = f"/opt/ml/processing/model/model.tar.gz"
    
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")
    
    model = AutoModelForSequenceClassification.from_pretrained(".")
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    
    def compute_metrics(p: EvalPrediction):
        predictions = p.predictions[0] if isinstance(p.predictions,
                                               tuple) else p.predictions
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(predictions))
        y_pred = torch.argmax(probs, dim=1)
        y_true = p.label_ids
        precision, recall, f1, _ = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, average="weighted")
        accuracy = accuracy_score(y_true, y_pred)
        # return as dictionary
        metrics = {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}
        return metrics
    
    logger.info("Model and Tokenizer have been loaded!")

    trainer = Trainer(
        model=model,
        compute_metrics=compute_metrics,
    )
    
    test_dataset = load_from_disk(f"/opt/ml/processing/test")
    
    eval_result = trainer.evaluate(eval_dataset=test_dataset)
    
    report_dict = {
        "multiclass_classification_metrics": {
            "f1": {
                "value": eval_result['eval_f1'],
                "standard_deviation": 'NaN'
            },
            "accuracy": {
                "value": eval_result['eval_accuracy'],
                "standard_deviation": 'NaN'
            },
            "precision": {
                "value": eval_result['eval_precision'],
                "standard_deviation": 'NaN'
            },
            "recall": {
                "value": eval_result['eval_recall'],
                "standard_deviation": 'NaN'
            },
        },
    }
    
    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    evaluation_path = f"{output_dir}/evaluation.json"
    
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))