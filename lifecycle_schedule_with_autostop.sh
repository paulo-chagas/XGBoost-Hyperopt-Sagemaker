set -e

ENVIRONMENT=conda_python3
NOTEBOOK_FILE="/home/ec2-user/SageMaker/optimize_and_train.ipynb"
AUTO_STOP_FILE="/home/ec2-user/SageMaker/auto-stop.py"
 
echo "Activating conda env"
source /home/ec2-user/anaconda3/bin/activate "$ENVIRONMENT"

echo "Starting notebook"
nohup jupyter nbconvert --to notebook --inplace --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 --execute "$NOTEBOOK_FILE" &

# echo "Decativating conda env"
# source /home/ec2-user/anaconda3/bin/deactivate

# PARAMETERS
IDLE_TIME=3600 # 1 hour

echo "Fetching the autostop script"
wget https://raw.githubusercontent.com/aws-samples/amazon-sagemaker-notebook-instance-lifecycle-config-samples/master/scripts/auto-stop-idle/autostop.py


echo "Detecting Python install with boto3 install"

# Find which install has boto3 and use that to run the cron command. So will use default when available
# Redirect stderr as it is unneeded
if /usr/bin/python -c "import boto3" 2>/dev/null; then
    PYTHON_DIR='/usr/bin/python'
elif /usr/bin/python3 -c "import boto3" 2>/dev/null; then
    PYTHON_DIR='/usr/bin/python3'
else
    # If no boto3 just quit because the script won't work
    echo "No boto3 found in Python or Python3. Exiting..."
    exit 1
fi

echo "Found boto3 at $PYTHON_DIR"


echo "Starting the SageMaker autostop script in cron"

(crontab -l 2>/dev/null; echo "*/5 * * * * $PYTHON_DIR $PWD/autostop.py --time $IDLE_TIME --ignore-connections >> /var/log/jupyter.log") | crontab -

