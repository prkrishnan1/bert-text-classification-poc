# BERT Model for Text Classification

Example notebooks and training pipeline for building a fine-tuned BERT LLM for Text Classification

# Example Notebooks

### BERT_Model_Training_and_Deployment.ipynb

A notebook which shows an example of data cleaning, processing, tokenization. Trains a HuggingFace AutoModelForSequenceClassification ('bert-base-uncased')

Deploys model on SageMaker endpoint and runs evaluation set, measuring F1-score

### BERT_Text_Classification

A notebook which shows data cleaning, processing, tokenization. 
Trains a HuggingFace AutoModelForSequenceClassification ('bert-base-uncased')
Loads model from checkpoint and runs evaluation set, measuring F-1 score


### AutoML_Text_Classification.ipynb

Trains SageMaker AutoMLModel
Choose best candidate model
Create SageMaker Predictor and Endpoint objects
Measure accuracy across Confidence Threshold and Dataset Coverage

# Pipeline

### pipeline.py

```
.
├── config
│   └── config.json

├── docker
│   ├── Dockerfile
│   └── requirements.txt

├── evaluation
│   └── evaluate.py

├── inference
│   ├── inference.py
│   └── requirements.txt

├── invoke_endpoint.sh

├── pipeline.py
	* Head file: Defines pipeline structure using sagemaker.pipelines library

├── processing
│   └── preprocess.py

└── training
    └── train.py

```
