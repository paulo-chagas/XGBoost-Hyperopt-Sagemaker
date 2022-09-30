set -e

ENVIRONMENT=conda_python3
NOTEBOOK_FILE="/home/ec2-user/SageMaker/optimize_and_train.ipynb"
 
echo "Activating conda env"
source /home/ec2-user/anaconda3/bin/activate "$ENVIRONMENT"

echo "Starting notebook"
nohup jupyter nbconvert --to notebook --inplace --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 --execute "$NOTEBOOK_FILE" &

