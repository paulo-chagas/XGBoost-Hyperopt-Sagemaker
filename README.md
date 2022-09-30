# XGBoost-Hyperopt-Sagemaker

Baseline architecture for XGBoost training on Sagemaker, called from Lambda. The Lambda function can be triggered with desired events. Model artifacts are stored on S3.

The proposal here is that a Lambda function starts a Sagemaker instance with a predefined lifecycle configuration that, on instance start, automatically runs a predefined notebook. The last cell stops the entire instance to avoid unnecessary costs.
