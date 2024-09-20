# BERT Model for Text Classification

Example notebooks and training pipeline for building a fine-tuned BERT LLM for Text Classification

# Example Notebooks

### BERT_Text_Classification.ipynb

A notebook which shows an example of data cleaning, processing, tokenization. Trains a HuggingFace AutoModelForSequenceClassification ('bert-base-uncased')

Deploys model on SageMaker endpoint and runs evaluation set, measuring F1-score

### 
A facet is column or feature that will be used to measure bias against. A facet can have value(s) that designates that sample as "***sensitive***".

### Label
The label is a column or feature which is the target for training a machine learning model. The label can have value(s) that designates that sample as having a "***positive***" outcome.

### Bias measure
A bias measure is a function that returns a bias metric.

### Bias metric
A bias metric is a numerical value indicating the level of bias detected as determined by a particular bias measure.

### Bias report
A collection of bias metrics for a given dataset or a combination of a dataset and model.

# Development

It's recommended that you setup a virtualenv.

```
virtualenv -p(which python3) venv
source venv/bin/activate.fish
pip install -e .[test]
cd src/
../devtool all
```

For running unit tests, do `pytest --pspec`. If you are using PyCharm, and cannot see the green run button next to the tests, open `Preferences` -> `Tools` -> `Python Integrated tools`, and set default test runner to `pytest`.

For Internal contributors, run ```../devtool integ_tests``` after creating virtualenv with the above steps to run the integration tests.
