from sagemaker.base_serializers import JSONSerializer
import boto3
import json

def output_fn(prediction, accept_type="application/json"):
    if accept_type == "application/json":
        # Initialize s3 client and load id2label map
        s3_client = boto3.client("s3")
        bucket = "sagemakerstanfordprojects"
        prefix = "snowtextclassification/data/output/id2label/id2label.json"
        resource = s3_client.get_object(Bucket=bucket, Key=prefix)

        id2label = json.loads(resource["Body"].read().decode('utf-8'))
        
        # Prediction in the form:
        # [{"label" : "LABEL_1", "score": 0.2348726347}]
        # Prediction in form: "LABEL_3". We need "3".
        label_index = prediction[0]["label"][6:]
        
        prediction[0]["label"] = id2label[label_index]
        
        return JSONSerializer(accept_type).serialize(prediction)
    else:
        raise ValueError("Prediction is not JSON-like type but it is type: {0}, data: {1}".format(type(prediction), prediction))