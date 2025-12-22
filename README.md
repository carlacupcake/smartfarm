# Running the genetic algorithm on AWS Lambda

## First-time function creation in Lambda
1. Go to AWS Console > Lambda
2. Click "Create function".
3. Choose "Author from scratch". Set...
   - Function name: `smartfarm-ga-eval`
   - Runtime: `Python 3.10`
   - Architecture: `x86_64`
4. Permissions: Under "Default execution role", select "Create new role with basic Lambda permissions".
5. Click "Create function".

## Upload zip to console
On the function page:
1. Go to the "Code" tab.
2. At the top right of the "Code source" box, click "Upload from".
3. Select ".zip file".
4. Locally, from the directory `src/smartfarm/core/aws/`, zip the latest versions of
   - `ga_lambda_handler.py`
   - `ga_lambda_helpers.py`
5. Click "Deploy".
6. In the "Runtime settings" section, make sure:
   - Handler = `ga_lambda_handler.lambda_handler`
7. In the "Layers" section, click "Add a layer".
   - Use the dropdown to select `AWSSDKPandas-Python310` and then version 27 (Note: this is what worked in `us-west-2`...it may be different in other regions).
   - Click "Add".
8. (Optional) Now, go to Lambda itself (not the specific Lambda function we are creating).
   - Go to "Layers" and click "Create layer".
   - Name the layer `mpmath-layer` and upload the zip `core/python/mpmath-layer.zip`.
   - For compatible runtimes, explicitly select Python 3.10.
   - Click "Create".
9. (Optional) Navigate back to the Lambda function `smartfarm-ga-eval`.
   - In "Code > Layers", click "Add a layer".
   - Select "Custom layers" and then the `mpmath-layer` you just created.
   - Click "Add".
10. Go to the "Configuration" tab.
   - Set "Memory" to 1024 MB.
   - Set "Timeout" to 1 minute.
   - Save.

## Test Lambda function after creation
1. Go to the "Test" tab.
2. Create a new event with a name of your choice e.g. "test".
3. Paste in the content from `src/smartfarm/core/aws/lambda_test.json`.
4. Click "Test".
5. You should see `{"statusCode": 200, "body": "{\"costs\": [1.0, 1.0]}"}` or similar.

# For running MPC

## Creating a Linux compatible layer
1. Go to AWS CloudShell.
2. Run `mkdir python`.
3. Run `pip3 install --target python "mpmath"`. Or, replace `"mpmath"` with whatever library you need.
4. Run `zip -r mpmath.zip python`.
5. Download the zip file to `core/python`.

## Running MPC weight sweep on EC2
1. Go to `EC2 > Instances` and launch an Amazon Linux c3.4 large instance. Use the smartfarm-key and allow `ssh` traffic.
2. From the directory `core/aws`, run `scp -i smartfarm-key.pem -r mpc <public-dns>:~/`.
3. Then run `ssh -i "smartfarm-key.pem" ec2-user@<public-dns>`.
4. In the EC2 instance, run the following series of commands:
```
sudo dnf update -y
sudo dnf install -y docker
sudo systemctl start docker
sudo usermod -aG docker $USER
exit
```
5. Then run `ssh -i "smartfarm-key.pem" ec2-user@<public-dns>`, optionally followed by `docker run hello-world`.
6. If docker gives a positive message, then run 
```
cd mpc
docker build -t smartfarm-mpc .
docker run --rm smartfarm-mpc
```
8. It may take >30 minutes to run.

## Building the Ipopt layer for Lambda
1. docker build -f Dockerfile.lambda -t smartfarm-mpc-lambda .
2. Sanity check: `docker run --rm -it --entrypoint bash smartfarm-mpc-lambda -lc "which ipopt && ipopt -v | head"` should yield `/opt/conda/bin/ipopt` and a version banner.
3. Sanity check that `awslambdaric` is installed: `docker run --rm -it --entrypoint bash smartfarm-mpc-lambda -lc "python -c 'import awslambdaric; print(\"ok\")'"`.
4. Create an ECR repo: 
```aws ecr create-repository \
  --repository-name smartfarm-mpc-lambda \
  --region us-west-1
```
5. Use the "repositoryUri" in the next step.
6. Login Docker to ECR: `aws ecr get-login-password --region us-west-1 | docker login --username AWS --password-stdin 870678672224.dkr.ecr.us-west-1.amazonaws.com/`
7. Tag and push to ECR:
```
docker tag smartfarm-mpc-lambda:latest \
  870678672224.dkr.ecr.us-west-1.amazonaws.com/smartfarm-mpc-lambda

docker push 870678672224.dkr.ecr.us-west-1.amazonaws.com/smartfarm-mpc-lambda:latest
```
10. docker buildx build \
  --platform linux/amd64 \
  --tag 870678672224.dkr.ecr.us-west-1.amazonaws.com/smartfarm-mpc-lambda:latest \
  --provenance=false \
  --sbom=false \
  --file Dockerfile.lambda \
  --push \
  .
11. Go Lambda and create a function from container, using the container we just created. Make sure to increase the timeout time.


