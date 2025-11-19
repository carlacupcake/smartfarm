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
   - `lambda_handler.py`
   - `lambda_member.py`
5. Click "Deploy".
6. In the "Runtime settings" section, make sure:
   - Handler = `lambda_handler.lambda_handler`
7. In the "Layers" section, click "Add a layer".
   - Use the dropdown to select `AWSSDKPandas-Python310` and then version 27 (Note: this is what worked in `us-west-2`...it may be different in other regions).
   - Click "Add".
8. Go to the "Configuration" tab.
   - Set "Memory" to 1024 MB.
   - Set "Timeout" to 1 minute.
   - Save.

## Test Lambda function after creation
1. Go to the "Test" tab.
2. Create a new event with a name of your choice e.g. "test".
3. Paste in the content from `src/smartfarm/core/aws/lambda_test.json`.
4. Click "Test".
5. You should see `{"statusCode": 200, "body": "{\"costs\": [1.0, 1.0]}"}` or similar.