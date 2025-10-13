# Build & push worker image to ECR

## Upload key files to EC2 instance (from local terminal)
EC2_PUBLIC_IP=$(aws ec2 describe-instances --instance-ids i-0acd0ef9dde406493 --region us-west-1 --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
scp -v -i smartfarm-key.pem ga_lambda.py ga_core.py Dockerfile ec2-user@${EC2_PUBLIC_IP}:/home/ec2-user/smartfarm-lambda/

## Verify files transferred to EC2 (from EC2 terminal)
ssh -i smartfarm-key.pem ec2-user@${EC2_PUBLIC_IP}
ls -l ga_core.py ga_lambda.py Dockerfile

## Build the docker image in EC2 instance (from EC2 terminal)
export DOCKER_BUILDKIT=0
export DOCKER_DEFAULT_PLATFORM=linux/amd64
docker build -t smartfarm-ga-lambda-classic:lambda-x86 .

## Tag and push to us-west-1 (from EC2 terminal)
REGION=us-west-1
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REPO=smartfarm-ga-lambda-classic
TAG=lambda-x86
FUNCTION_NAME=smartfarm-ga-eval

aws ecr create-repository --repository-name $REPO --region $REGION || true
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

docker tag  ${REPO}:${TAG} ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:${TAG}
docker push ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:${TAG}

## Verify the manifest type is Lambda compatible (Docker v2 schema-2) (from EC2 terminal)
aws ecr describe-images --repository-name $REPO --region $REGION --query 'imageDetails[].{Tags:imageTags, Media:imageManifestMediaType}'

## Create or update Lambda in us-west-1 (from EC2 terminal)
IMAGE_URI=${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:${TAG}
ROLE_ARN=$(aws iam get-role --role-name smartfarm-lambda-job-role --query 'Role.Arn' --output text)

# x86_64 build: (from EC2 terminal)
aws lambda create-function \
  --function-name $FUNCTION_NAME \
  --package-type Image \
  --code ImageUri=$IMAGE_URI \
  --role $ROLE_ARN \
  --architectures x86_64 \
  --timeout 600 --memory-size 2048 \
  --region $REGION

# If the function already exists in us-west-1, use:
aws lambda update-function-code \
  --function-name $FUNCTION_NAME \
  --image-uri $IMAGE_URI \
  --region $REGION

aws lambda wait function-updated \
  --function-name $FUNCTION_NAME \
  --region $REGION

VER=$(aws lambda publish-version \
  --function-name $FUNCTION_NAME \
  --region $REGION --query Version --output text)
echo "Published version: $VERSION"

aws lambda update-alias \
  --function-name $FUNCTION_NAME \
  --name $ALIAS \
  --function-version $VERSION \
  --region $REGION

aws lambda put-provisioned-concurrency-config \
  --function-name $FUNCTION_NAME \
  --qualifier $ALIAS \
  --provisioned-concurrent-executions 200 \
  --region $REGION

## For parallelism, set reserve concurrency (cap of "warm" parallel invocations)
### This sets the reserve to 800, but I requested up to 1000
aws lambda put-function-concurrency \
  --function-name $FUNCTION_NAME \
  --reserved-concurrent-executions 800 \
  --region us-west-1

### Check reserve concurrency
aws lambda get-function-concurrency --function-name $FUNCTION_NAME

## For parallelism, set provisioned concurrency (a set number of pre-warmed invocations)
### Publish a version and alias it "live"
VERSION=$(aws lambda publish-version --function-name $FUNCTION_NAME --region us-west-1 --query Version --output text)
ALIAS=live
aws lambda create-alias --function-name $FUNCTION_NAME --name $ALIAS --function-version $VERSION --region us-west-1

### Pre-warm 200 executions (I chose the number of strings per generation)
aws lambda put-provisioned-concurrency-config \
  --function-name $FUNCTION_NAME --qualifier $ALIAS \
  --provisioned-concurrent-executions 200 \
  --region us-west-1

### Check provision concurrency
aws lambda get-provisioned-concurrency-config \
    --function-name $FUNCTION_NAME \
    --qualifier $ALIAS

## Handy commands for listing roles/ARNs
aws iam list-roles --query 'Roles[].RoleName' --output text
aws iam get-role --role-name smartfarm-batch-job-role --query 'Role.Arn' --output text
aws iam get-role --role-name ecsTaskExecutionRole --query 'Role.Arn' --output text
aws iam get-role --role-name AWSBatchServiceRole --query 'Role.Arn' --output text

## (ONE-TIME) Create a batch service role
aws iam create-role --role-name AWSBatchServiceRole \
  --assume-role-policy-document '{
    "Version":"2012-10-17",
    "Statement":[{"Effect":"Allow","Principal":{"Service":"batch.amazonaws.com"},"Action":"sts:AssumeRole"}]
  }' || true

aws iam attach-role-policy --role-name AWSBatchServiceRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSBatchServiceRole

## (ONE-TIME) Create an ECS task execution role - use to pull image/send logs
aws iam create-role --role-name ecsTaskExecutionRole \
  --assume-role-policy-document '{
    "Version":"2012-10-17",
    "Statement":[{"Effect":"Allow","Principal":{"Service":"ecs-tasks.amazonaws.com"},"Action":"sts:AssumeRole"}]
  }' || true

aws iam attach-role-policy --role-name ecsTaskExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy

## (ONE-TIME) Create a batch job role - use to read/write to s3
aws iam create-role --role-name smartfarm-batch-job-role \
  --assume-role-policy-document '{
    "Version":"2012-10-17",
    "Statement":[{"Effect":"Allow","Principal":{"Service":"ecs-tasks.amazonaws.com"},"Action":"sts:AssumeRole"}]
  }'

aws iam put-role-policy --role-name smartfarm-batch-job-role --policy-name SmartfarmS3Access \
  --policy-document '{
    "Version":"2012-10-17",
    "Statement":[
      {"Effect":"Allow","Action":["s3:ListBucket"],"Resource":["arn:aws:s3:::carlas-smartfarm"]},
      {"Effect":"Allow","Action":["s3:GetObject","s3:PutObject"],"Resource":["arn:aws:s3:::carlas-smartfarm/*"]}
    ]
  }'
