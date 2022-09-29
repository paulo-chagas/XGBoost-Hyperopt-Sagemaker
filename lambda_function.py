import boto3
import json
import pandas as pd
from lambda_utils import get_domain_dict, get_df_from_query_file


s3_resource = boto3.resource('s3')
sagemaker_client = boto3.client('sagemaker', region_name='us-east-1')
lambda_client = boto3.client('lambda', region_name='us-east-1')


def lambda_handler(event, context):
    df = get_df_from_query_file('./query_all_data.sql')
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date', inplace=True,
                   ascending=True, ignore_index=True)
    df.fillna(0, inplace=True)
    df.reset_index(drop=True, inplace=True)

    input_tmp = '/tmp/data.csv'
    df.to_csv(input_tmp, index=False, header=True)

    bucket = 'bucket_name'
    domain_dict = get_domain_dict()

    for domain in domain_dict.keys():
        target_problem = domain_dict[domain]['target_problem']

        key = f'{target_problem}/train/data.csv'
        s3_resource.Bucket(bucket).Object(key).upload_file(input_tmp)

    sagemaker_client.start_notebook_instance(
        NotebookInstanceName='schedule-training-notebook')
    print('Scheduled')

    return {
        'statusCode': 200,
        'body': json.dumps('Finished!')
    }
