import logging
import os
from datetime import timedelta, datetime

import yaml

from airflow.models import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.exceptions import AirflowException

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

class DagBuilder:
    CONF = "/path/to/dag.yaml"
    PYTHON_VENV_PATH = "/venv/bin/python"
    PYTHON_JOBS_PATH = "/python"

    def __init__(self):

        with open(DagBuilder.CONF, "r") as yaml_config:
            conf = yaml.load(yaml_config)
            self.tasks = conf['tasks']
            self.path = conf['path']
            self.env = conf['env']

        self.das_python_path = self.path + DagBuilder.PYTHON_VENV_PATH
        self.das_jobs_path = self.path + DagBuilder.PYTHON_JOBS_PATH

    def __get_accounts(self):

        return [ '000087' ]

    def create_template_dag(self, dag_id):
        default_args = {
            'owner': 'airflow',
            'depends_on_past': True
        }

        _dag = DAG(
            dag_id=dag_id,
            schedule_interval=None,# "0 0 * * *",
            start_date=datetime.strptime("2019-02-01", '%Y-%m-%d'),
            default_args=default_args,
            max_active_runs=1
        )

        return _dag

    def create_template_task(self, dag_id, account_id, task, env):

        execute_command = self.das_python_path + " " +\
                          self.das_jobs_path + "/main.py" +\
                          " --account_id {}".format(account_id) +\
                          " --date {}".format('{{ ds }}') +\
                          " --task {}".format(task) +\
                          " --env {}".format(env) +\
                          " --conf {}".format(DagBuilder.CONF)

        execute = BashOperator(
            task_id=task,
            bash_command=execute_command,
            depends_on_past=True,
            retries=3,
            retry_delay=timedelta(minutes=3),
            dag=globals()[dag_id])

        return execute

    def create_account_dags(self):
        accounts = self.__get_accounts()
        template_dag_id = "IA_{}"

        for account_id in accounts:

            dag_id = template_dag_id.format(account_id)

            globals()[dag_id] = self.create_template_dag(dag_id)

            tasks = [self.create_template_task(dag_id, account_id, task, self.env)
                for task in self.tasks]
                
            for i, task in enumerate(tasks[:-1]):
                current = task
                current >> tasks[i+1]

dag_builder = DagBuilder()
dag_builder.create_account_dags()
