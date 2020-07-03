import os
import sys
import yaml
import time
import logging
import argparse

import warnings
warnings.filterwarnings("ignore", "Your application has authenticated using end user credentials")

from datetime import datetime, timedelta

from google.oauth2 import service_account
from google.cloud import bigquery, storage
from google.cloud.exceptions import Forbidden, NotFound
from googleapiclient import discovery

from google_pandas_import_export import GooglePandasImportExport
from model.model import MyModel
import model.parameters as param

logger = logging.getLogger(__name__)


def is_success(ml_engine_service, project_id, job_id):
    wait = 20 # seconds
    timeout_preparing = timedelta(seconds=900)
    timeout_running = timedelta(hours=6)
    api_call_time = datetime.now()
    api_job_name = "projects/{project_id}/jobs/{job_name}".format(project_id=project_id, job_name=job_id)
    job_description = ml_engine_service.projects().jobs().get(name=api_job_name).execute()
    while not job_description["state"] in ["SUCCEEDED", "FAILED", "CANCELLED"]:
        time.sleep(2)
        if job_description["state"] == "PREPARING":
            delta = datetime.now() - api_call_time
            if delta > timeout_preparing:
                logger.error("[ML] PREPARING stage timeout after %ss --> CANCEL job '%s'" %(delta.seconds, job_id))
                ml_engine_service.projects().jobs().cancel(name=api_job_name, body={}).execute()
                raise Exception
        if job_description["state"] == "RUNNING":
            delta = datetime.now() - api_call_time
            if delta > timeout_running + timeout_preparing:
                logger.error("[ML] RUNNING stage timeout after %ss --> CANCEL job '%s'" %(delta.seconds, job_id))
                ml_engine_service.projects().jobs().cancel(name=api_job_name, body={}).execute()
                raise Exception
        logger.info("[ML] NEXT UPDATE for job '%s' IN %ss (%ss ELAPSED IN %s STAGE)" %(job_id,
                                                                                       wait,
                                                                                       delta.seconds,
                                                                                       job_description["state"]))
        job_description = ml_engine_service.projects().jobs().get(name=api_job_name).execute()
        time.sleep(wait)
    logger.info("Job '%s' done" % job_id)
    if job_description["state"] == "SUCCEEDED":
        logger.info("Job '%s' succeeded!" % job_id)
        return True
    else:
        logger.error(job_description["errorMessage"])
        return False


def ml_job(ml_engine_service, project_account, bucket_id, task):

    job_parent = "projects/{project}".format(project=project_account)
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    job_id = "job_{}_{}".format(account_id, now_str)

    job_body = {'trainingInput':
                {'pythonVersion': param.ml_pythonVersion,
                 'runtimeVersion': param.ml_runtimeVersion,
                 'scaleTier': param.ml_typology[task]['ml_scaleTier'],
                 'region': param.ml_region,
                 'pythonModule': 'model.model',
                 'args': ["--project_account", project_account,
                          "--bucket_id", bucket_id,
                          "--task", task],
                 'packageUris': [
                    "gs://%s/%s/mymodel-0.0.1-py3-none-any.whl"%(bucket_id, param.bucket_path),
                    "gs://%s/%s/google_pandas_import_export-1.1rc0-py3-none-any.whl"%(bucket_id, param.bucket_path),
                 ],
                 'masterType': param.ml_typology[task]['ml_masterType']
                 },
                'jobId': job_id}

    logging.info("job_body: %s" % job_body)
    logging.info("job_parent: %s" % job_parent)
    logging.info("creating a job ml: %s" % job_id)
    return ml_engine_service.projects().jobs().create(parent=job_parent, body=job_body), job_id


if __name__ == '__main__':

    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('google').setLevel(logging.WARNING)
    logging.getLogger('googleapiclient').setLevel(logging.WARNING)
    logging.getLogger('google_auth_httplib2').setLevel(logging.WARNING)
    logging.getLogger('googleapiclient.discovery_cache').setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description="LSTM Network",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--account_id", dest="account_id", help="account id to work on")
    parser.add_argument("--date", dest="date", help="airflow ds variable")
    parser.add_argument("--task", dest="task", choices=['preprocess', 'train', 'test', 'predict'], help="task to perform")
    parser.add_argument("--env", dest="env", default="local", choices=['local', 'cloud'], help="environment")
    parser.add_argument("--conf", dest="conf", default="../conf/dag.yaml",
                        help="absolute or relative path to configuration file")

    if len(sys.argv) < 7:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    root_logger = logging.getLogger()
    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.setLevel(logging.DEBUG) # INFO
    root_logger.addHandler(console_handler)

    account_id = args.account_id

    with open(args.conf, 'r') as f:
        config = yaml.load(f)

    # DEFINE BQ CLIENT
    project_account = config['google_cloud']['project_prefix'] + account_id
    if config['google_cloud']['credentials_json_file'] != "":
        credentials = service_account.Credentials.from_service_account_file(
                        config['google_cloud']['credentials_json_file'])
        gs_client_account = storage.Client(project=project_account, credentials=credentials)
        bq_client_account = bigquery.Client(project=project_account, credentials=credentials)
    else:
        credentials = None
        gs_client_account = storage.Client(project=project_account)
        bq_client_account = bigquery.Client(project=project_account)

    # DEFINE DATASET REF
    dataset_ref = bq_client_account.dataset(param.dataset_id)
    try:
        bq_client_account.get_dataset(dataset_ref)
    except NotFound as nf:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = "EU"
        bq_client_account.create_dataset(dataset)

    # DEFINE BUCKET
    bucket_id = config['google_gcs']['bucket_id']
    bucket = gs_client_account.bucket(bucket_id)

    startTime = datetime.now()
    logger.info("PROCESS BEGINS")

    # START TASK
    logger.info("TASK %s"%(args.task))

    # LOCAL
    m = MyModel(project_account, bucket_id, 'local')
    # CLOUD
    ml_engine_service = discovery.build('ml', 'v1', credentials=credentials)

    if args.task == 'preprocess':

        # INSTANTIATE GPIE
        gpie = GooglePandasImportExport(bq_client=bq_client_account, dataset_ref=dataset_ref,
                                        bucket=bucket, gs_dir_path_in_bucket=param.bucket_path)

        query = param.query.format(project_account=project_account)
        
        gpie.convey(source='query', destination='gs', query=query, data_name=param.df)

    if args.env == 'local':
        if args.task == 'preprocess':
            m.preprocess()
        elif args.task == 'train':
            m.train()
        elif args.task == 'test':
            m.test()

    elif args.env == 'cloud':
        job, job_id = ml_job(ml_engine_service, project_account, bucket_id, args.task)
        job.execute()
        result = is_success(ml_engine_service, project_account, job_id)
        logger.info("JOB ML: %s"%result)

    logger.info("TASK %s DONE"%(args.task))

    logger.info("PROCESS ENDED (TOOK %s)"%str(datetime.now() - startTime))