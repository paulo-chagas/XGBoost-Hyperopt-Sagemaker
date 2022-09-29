from typing import List
import psycopg2
import boto3
import pandas as pd
import os
import json

s3_client = boto3.client('s3', region_name='us-east-1')


def redshift_conn():
    redshift_database = os.environ["DB_NAME"]
    redshift_host = os.environ["DB_HOST"]
    redshift_password = os.environ["DB_PASS"]
    redshift_port = os.environ["DB_PORT"]
    redshift_user = os.environ["DB_USER"]

    return psycopg2.connect(
        dbname=redshift_database,
        user=redshift_user,
        password=redshift_password,
        port=redshift_port,
        host=redshift_host,
        connect_timeout=60
    )


def get_df_from_query_file(path: str) -> pd.DataFrame:
    with open(path, 'r') as file:
        query = file.read().replace('\n', ' ')
    return get_df_from_query(query)


def get_df_from_query(sql: str) -> pd.DataFrame:
    conn = redshift_conn()
    return pd.read_sql(sql, conn)


def get_max_predicted_date() -> str:
    conn = redshift_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
                    SELECT MAX(confirmed_date) FROM schema.table
                    """)
        result = cur.fetchall()
    except Exception as e:
        print(str(e))
    conn.commit()
    cur.close()
    conn.close()
    return result


def get_df_from_s3(bucket: str, key: str, header='infer') -> pd.DataFrame:
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(obj['Body'], header=header)
    return df


def get_feature_info_from_s3(bucket: str, key: str, to_skip: List) -> dict:
    info = {}
    df = get_df_from_s3(bucket, key)

    for column in df.columns:
        if column not in to_skip:
            info[column] = {'mean': df[column].mean(), 'std': df[column].std()}
    return info


def get_request(batch_job_name: str,
                model_name: str,
                output_path: str,
                input_path: str) -> dict:
    request = {
        "TransformJobName": batch_job_name,
        "ModelName": model_name,
        "MaxConcurrentTransforms": 1,
        "TransformOutput": {
            "Accept": "text/csv",
            "S3OutputPath": output_path,
            "AssembleWith": "Line"
        },
        "TransformInput": {
            "ContentType": "text/csv",
            "SplitType": "Line",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": input_path
                }
            }
        },
        "TransformResources": {
            "InstanceType": "ml.m4.xlarge",
            "InstanceCount": 1
        },
        "DataProcessing": {
            "InputFilter": "$[4:]",
            "OutputFilter": "$",
            "JoinSource": "Input"
        }
    }
    return request


def get_max_date() -> str:
    conn = redshift_conn()
    cur = conn.cursor()
    result = ''
    try:
        cur.execute("""
            SELECT TO_CHAR(MAX(confirmed_date), 'YYYY-MM-DD') FROM schema.table
                    """)
        result = cur.fetchall()
    except Exception as e:
        print(str(e))
    conn.commit()
    cur.close()
    conn.close()
    return result[0][0]


def insert_df_on_table(df: pd.DataFrame):
    conn = redshift_conn()
    cur = conn.cursor()

    for date, col1, col2, col3 in zip(df['date'],
                                      df['col1'],
                                      df['col2'],
                                      df['col3']):
        try:
            cur.execute("""
                        INSERT INTO schema.table
                        VALUES (%s, %s, %s, %s, %s, %s);
                        """,
                        (date, col1, col2, col3, 0, json.dumps({})))
        except Exception as e:
            print(str(e))
            break

    conn.commit()
    cur.close()
    conn.close()


def get_domain_dict():
    target_dict = {
        "target": {
            "target_problem": "target-prediction",
            "TARGET": "column_target"
        }
    }
    return target_dict
