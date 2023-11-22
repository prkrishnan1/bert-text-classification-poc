import os
import sys
import time
import boto3
from botocore.exceptions import ClientError
import json
import logging
from time import strftime
import numpy as np
import pandas as pd
import sagemaker
from sagemaker import get_execution_role
from sagemaker.session import Session
from sagemaker.processing import (
    ScriptProcessor,
    ProcessingInput,
    ProcessingOutput,
)
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.workflow.condition_step import (
    ConditionStep,
    JsonGet,
)
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.huggingface.model import HuggingFaceModel
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.fail_step import FailStep
from sagemaker.huggingface import HuggingFace
from sagemaker.inputs import TrainingInput

logger = logging.getLogger(__name__)

def write_env_variables():
    """
    Write all configuration variables to environment variables for pipeline execution
    """
    with open("./config/config.json", "r", encoding="utf-8") as file:
        config = json.load(file)
        for key in config:
            os.environ[key] = config[key]

def upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: S3 location if file was successfully uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    
    return True

def upload_data(local_data_path: str):
    """
    Upload data in local directory to S3
    """
    #TODO: Upload data and code
    container_base_path = os.getenv("PROCESSING_CONTAINER_BASE_PATH")
    s3_prefix = "snowtextclassification"
    bucket = os.getenv("BUCKET_NAME")
    
    print(local_data_path, bucket, s3_prefix)
    
    filename = f"incidents-{strftime('%Y-%m-%d-%H-%M-%S')}.csv"
    data_upload = upload_file(local_data_path, bucket, f"{s3_prefix}/data/input/{filename}")
    
    if not data_upload:
        logger.exception(f"Data: {filename} did not upload to S3")
        return
    
    return filename

def define_preprocessing_step(data_file_name):
    """
    Defines a Processing Step in pipeline for data preprocessing
    
    :param: input path: S3 Path where raw data is located
    :param: output path: Destination S3 path where processed data will be located
    :param: data file name: file name of data being processed
    :return: ProcessingStep object
    """
    container_base_path = os.getenv("PROCESSING_CONTAINER_BASE_PATH")
    bucket = os.getenv("BUCKET_NAME")
    s3_prefix = "snowtextclassification"
    
    s3_input_location = f"s3://{bucket}/{s3_prefix}/data/input/{data_file_name}"
    s3_output_location = f"s3://{bucket}/{s3_prefix}/data/output"
    
    processor = ScriptProcessor(
        role=os.getenv("ROLE"),
        image_uri=os.getenv("PROCESSING_IMAGE_URI"),
        instance_count=1,
        instance_type="ml.m5.xlarge",
        command=["python3"],
    )
    
    step_process = ProcessingStep(
            name="PreprocessSNOWIncidentData",
            code="./processing/preprocess.py",
            processor=processor,
            inputs=[
                ProcessingInput(
                    source=s3_input_location,
                    # preprocess.py needs to use destination path for data
                    destination=f"{container_base_path}/data/",
                )
            ],
            outputs=[
                ProcessingOutput(
                    # preprocess.py needs to write to the source path here.
                    source=f"{container_base_path}/output/train",
                    destination=f"{s3_output_location}/train",
                    output_name="train",
                ),
                ProcessingOutput(
                    source=f"{container_base_path}/output/test",
                    destination=f"{s3_output_location}/test",
                    output_name="test",
                ),
                ProcessingOutput(
                    source=f"{container_base_path}/output/validation",
                    destination=f"{s3_output_location}/validation",
                    output_name="validation",
                ),
                ProcessingOutput(
                    source=f"{container_base_path}/output/id2label",
                    destination=f"{s3_output_location}/id2label",
                    output_name="id2label"
                )
            ],
            job_arguments = ["--container-base-path", container_base_path,
                             "--data-file-name", data_file_name,
                             "--model-name", "bert-base-uncased",
                             "--train-ratio", "0.7",
                             "--val-ratio", "0.2"],
        )
    
    return step_process

def define_training_step(step_process: ProcessingStep):
    """
    Define model training step of pipeline
    
    :param: step_process: ProcessingStep object used to pipe artifacts from processing stage
    :return: TrainingStep object defining Training Job, and estimator object
    """
    model_name = "bert-base-uncased"
    training_job_name = f"bert-base-uncased-snow-tc-{strftime('%Y-%m-%d-%H-%M-%S')}"
    
    bucket = os.getenv("BUCKET_NAME")
    s3_prefix = "snowtextclassification"
    model_artifact_path = f"s3://{bucket}/{s3_prefix}/model/"

    hyperparameters={'epochs': 4,
                     'train_batch_size': 16,
                     'model_name': model_name,
                     'tokenizer_name': model_name,
                     'output_dir':'/opt/ml/checkpoints',
                     }
    
    metric_definitions=[
        {'Name': 'loss', 'Regex': "'loss': ([0-9]+(.|e\-)[0-9]+),?"},
        {'Name': 'learning_rate', 'Regex': "'learning_rate': ([0-9]+(.|e\-)[0-9]+),?"},
        {'Name': 'eval_loss', 'Regex': "'eval_loss': ([0-9]+(.|e\-)[0-9]+),?"},
        {'Name': 'eval_accuracy', 'Regex': "'eval_accuracy': ([0-9]+(.|e\-)[0-9]+),?"},
        {'Name': 'eval_f1', 'Regex': "'eval_f1': ([0-9]+(.|e\-)[0-9]+),?"},
        {'Name': 'eval_precision', 'Regex': "'eval_precision': ([0-9]+(.|e\-)[0-9]+),?"},
        {'Name': 'eval_recall', 'Regex': "'eval_recall': ([0-9]+(.|e\-)[0-9]+),?"},
        {'Name': 'eval_runtime', 'Regex': "'eval_runtime': ([0-9]+(.|e\-)[0-9]+),?"},
        {'Name': 'eval_samples_per_second', 'Regex': "'eval_samples_per_second': ([0-9]+(.|e\-)[0-9]+),?"},
        {'Name': 'epoch', 'Regex': "'epoch': ([0-9]+(.|e\-)[0-9]+),?"}]

    huggingface_estimator = HuggingFace(entry_point='train.py',
                                source_dir='./training',
                                instance_type='ml.g4dn.2xlarge',
                                instance_count=1,
                                role=os.getenv("ROLE"),
                                transformers_version=os.getenv("TRANSFORMERS_VERSION"), 
                                pytorch_version=os.getenv("PYTORCH_VERSION"),
                                py_version=os.getenv("PY_VERSION"),
                                hyperparameters = hyperparameters,
                                metric_definitions=metric_definitions,
                                output_path=model_artifact_path,
                                max_run=36000, # expected max run in seconds
                            )
    
    step_train = TrainingStep(
        name="TrainHuggingFaceBERTModel",
        estimator=huggingface_estimator,
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri
            ),
            "eval": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri
            ),
        },
    )
    
    return step_train, huggingface_estimator

def define_evaluation_step(step_train, step_process):
    """
    Define evaluation step of pipeline
    
    :param: step_train: TrainingStep definition to access model artifacts
    :param: step_process: ProcessingStep definition to access test dataset
    :return: ProcessingStep for evaluating model, and Evaluation Report for future pipeline steps
    """
    
    processor = ScriptProcessor(
        role=os.getenv("ROLE"),
        image_uri=os.getenv("PROCESSING_IMAGE_URI"),
        instance_count=1,
        instance_type="ml.m5.4xlarge",
        command=["python3"],
    )

    evaluation_report = PropertyFile(
        name="BERTSNOWEvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )
    
    bucket = os.getenv("BUCKET_NAME")
    s3_prefix = os.getenv("S3_PREFIX")
    
    s3_output_location = f"s3://{bucket}/{s3_prefix}/model/output/evaluation/"
    
    step_eval = ProcessingStep(
        name="EvaluateBERTSNOWModel",
        code="./evaluation/evaluate.py",
        processor=processor,
            inputs=[
                ProcessingInput(
                    source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                    destination="/opt/ml/processing/model",
                ),
                ProcessingInput(
                    source=step_process.properties.ProcessingOutputConfig.Outputs[
                        "test"
                    ].S3Output.S3Uri,
                    destination="/opt/ml/processing/test",
                ),
            ],
            outputs=[
                ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation", destination=s3_output_location),
            ],
        property_files=[evaluation_report],
    )
    
    return step_eval, evaluation_report

def define_condition_step(step_train, step_eval, pipeline_session, evaluation_report):
    base_framework_version = "pytorch{}".format(os.getenv("PYTORCH_VERSION"))
    
    inference_image_uri = sagemaker.image_uris.retrieve(
        framework="huggingface",
        region="us-east-1",
        version=os.getenv("TRANSFORMERS_VERSION"),
        instance_type='ml.m5.2xlarge',
        image_scope='inference',
        base_framework_version=base_framework_version
    )
    
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=step_eval.properties.ProcessingOutputConfig.Outputs[0].S3Output.S3Uri,
            content_type="application/json"
        )
    )
    
    huggingface_model = HuggingFaceModel(
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,       # path to your model and script
        role=os.getenv("ROLE"),                    # iam role with permissions to create an Endpoint
        source_dir="./inference",
        entry_point="inference.py",
        name="HuggingFaceBERTSNOWModel",
        sagemaker_session=pipeline_session,
        transformers_version=os.getenv("TRANSFORMERS_VERSION"),  # transformers version used
        pytorch_version=os.getenv("PYTORCH_VERSION"),        # pytorch version used
        py_version=os.getenv("PY_VERSION"),            # python version used
    )

    step_model_create = ModelStep(
       name="BERTSNOWModelCreationStep",
       step_args=huggingface_model.create(instance_type="ml.m5.2xlarge"),
    )
    
    step_model_register = ModelStep(
        name="BERTSNOWModelRegisterStep",
        step_args=huggingface_model.register(
            content_types=["application/json"],
            response_types=["application/json"],
            inference_instances=["ml.m5.xlarge","ml.m5.2xlarge"],
            transform_instances=["ml.m5.xlarge"],
            model_package_group_name="SNOWBERTPackageGroup",
            image_uri=inference_image_uri,
            approval_status="PendingManualApproval",
            model_metrics=model_metrics,
        )
    )
    
    # condition step for evaluating model quality and branching execution
    cond_gte = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step=step_eval,
            property_file=evaluation_report,
            json_path="multiclass_classification_metrics.f1.value"
        ),
        right=0.70,
    )
    
    step_fail = FailStep(name="BERTSNOWModelPerformanceFailure")
    
    step_cond = ConditionStep(
        name="CheckF1BERTEvaluation",
        conditions=[cond_gte],
        if_steps=[step_model_create, step_model_register],
        else_steps=[step_fail]
    )
    
    return step_cond


def get_pipeline_definition():
    pipeline_session = PipelineSession()
    data_file_name = upload_data("data/incident2023.csv")
    step_process = define_preprocessing_step(data_file_name)
    step_train, hf_estimator = define_training_step(step_process)
    step_eval, evaluation_report = define_evaluation_step(step_train, step_process)
    step_cond = define_condition_step(step_train, step_eval, pipeline_session, evaluation_report)
    
    snow_pipeline = Pipeline(
        name="SNOWBERTTextClassification",
        steps=[step_process, step_train, step_eval, step_cond],
        sagemaker_session=pipeline_session,
    )
    
    return snow_pipeline

if __name__ == "__main__":
    write_env_variables()
    pipeline = get_pipeline_definition()
    
    print("###### Creating/updating a SageMaker Pipeline with the following definition:")
    parsed = json.loads(pipeline.definition())
    print(json.dumps(parsed, indent=2, sort_keys=True))
    
    upsert_response = pipeline.upsert(
            role_arn=os.getenv("ROLE")
        )
    print("\n###### Created/Updated SageMaker Pipeline: Response received:")
    print(upsert_response)
        
    try:
        
        execution = pipeline.start()
        print(f"\n###### Execution started with PipelineExecutionArn: {execution.arn}")

        print("Waiting for the execution to finish...")
        execution.wait()
        print("\n#####Execution completed. Execution step details:")

        print(execution.list_steps())
        # Todo print the status?
    except Exception as e:  # pylint: disable=W0703
        print(f"Exception: {e}")
        sys.exit(1)
