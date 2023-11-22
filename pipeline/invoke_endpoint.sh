#!/bin/bash

# Define your AWS credentials and region
export AWS_ACCESS_KEY_ID='AKIAUOFCMA4HJ2JCLDX6'
export AWS_SECRET_ACCESS_KEY='s++hiiNbRFiE0fsC3uppsyhpJRFuhZWUwv3Lz9ry'
export AWS_DEFAULT_REGION='us-east-1'

# Define the SageMaker endpoint name
endpoint_name='huggingface-bert-base-uncased'

# Define the payload you want to send (as a JSON string)
payload="{\"inputs\" : \"$PREDICTION\"}"

# Define the date in ISO8601 format
amz_date=$(date -u +'%Y%m%dT%H%M%SZ')

# Define the region and service
region='us-east-1'
service='sagemaker'

prod_variant_name='default-variant-name'

# Define the canonical request
canonical_request=$(cat <<EOF
POST
/endpoints/$endpoint_name/invocations
content-type:application/json
host:$service.$region.amazonaws.com
x-Amzn-Invoked-Production-Variant: $prod_variant_name
x-amz-date:$amz_date

$(echo -n $payload | sha3-256sum | awk '{print $1}')
EOF
)

# Define the string to sign
credential_scope=$(date -u +'%Y%m%d')/$region/$service/aws4_request
string_to_sign=$(cat <<EOF
AWS4-HMAC-SHA256
$amz_date
$credential_scope
$(echo -n $canonical_request | sha3-256sum | awk '{print $1}')
EOF
)

echo "STRING TO SIGN"
echo $string_to_sign


# Generate the signing key
k_date=$(echo -n $amz_date | openssl dgst -sha256 -hmac "AWS4$AWS_SECRET_ACCESS_KEY" | awk '{print $2}')
k_region=$(echo -n $region | openssl dgst -sha256 -mac HMAC -macopt hexkey:$k_date | awk '{print $2}')
k_service=$(echo -n $service | openssl dgst -sha256 -mac HMAC -macopt hexkey:$k_region | awk '{print $2}')
k_signing=$(echo -n "aws4_request" | openssl dgst -sha256 -mac HMAC -macopt hexkey:$k_service | awk '{print $2}')

echo "DATE"
echo $k_date
echo "REGION"
echo $k_region
echo "SERVICE"
echo $k_service
echo "SIGNING"
echo $k_signing

# Generate the signature
signature=$(echo -n $string_to_sign | openssl dgst -sha256 -mac HMAC -macopt hexkey:$k_signing | awk '{print $2}')

# Define the Authorization header
export authorization_header="AWS4-HMAC-SHA256 Credential=$AWS_ACCESS_KEY_ID/$credential_scope, SignedHeaders=content-type;host;x-amz-date, Signature=$signature"

echo "Auth Header"
echo $authorization_header
echo "AMZ DATE"
echo $amz_date
echo "PAYLOAD"
echo $payload
echo "ENDPOINT"
echo "https://$service.$region.amazonaws.com/endpoints/$endpoint_name/invocations"
echo "Signature"
echo $signature

# Make the curl request
curl -X POST -H "Content-Type: application/json" -H "x-Amzn-Invoked-Production-Variant: $prod_variant_name" -H "X-Amz-Date: $amz_date" -H "Authorization: $authorization_header" -d "$payload" https://runtime.$service.$region.amazonaws.com/endpoints/$endpoint_name/invocations

