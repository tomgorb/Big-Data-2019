import logging
import random
import argparse
import os, sys

import pandas as pd
import numpy as np

from google.cloud import storage
from datetime import datetime, timedelta
from tensorflow.keras.models import model_from_json
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf

from sklearn.metrics import roc_curve, auc, brier_score_loss

from google_pandas_import_export import GooglePandasImportExport
import model.rnn as rnn
import model.display as display
import model.parameters as param

logger = logging.getLogger(__name__)


class MyModel:
    def __init__(self, project_account, bucket_id, env):
        super(MyModel, self).__init__()
        self.env = env
        self.project_account = project_account
        self.bucket_id = bucket_id
        # INSTANTIATE GPIE
        bucket = storage.Client(project=self.project_account).bucket(self.bucket_id)
        self.gpie = GooglePandasImportExport(local_dir_path=param.local_dir_path, 
                                             bucket=bucket, gs_dir_path_in_bucket=param.bucket_path)

    def preprocess(self):

        logger.info("PREPROCESSING...")

        df = self.gpie.convey(source='gs', destination='dataframe', data_name=param.df)

        df.loc[:, 'timestamp'] = df.hit_timestamp.apply(
            lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S UTC'))

        df = df[[param.key, param.event_timestamp, param.target[0]]+param.embedding+param.dummy+param.numerical]

        df_sorted = df.sort_values([param.key, param.event_timestamp], ascending=[True, True], inplace=False)
        df_sorted.reset_index(drop=True, inplace=True)
        
        # EMBEDDING
        df_label_encoded = pd.DataFrame()
        for feature in param.embedding:
            values = df_sorted[feature].unique()
            encoding_labels = np.arange(values.shape[0])+1
            dict_labels = dict(zip(values,encoding_labels))
            df_label_encoded[feature] = df_sorted[feature].apply(lambda x: dict_labels[x])
        df_sorted.drop(param.embedding, inplace=True, axis=1)

        # DUMMY
        df_dummy = pd.get_dummies(df_sorted[param.dummy])
        df_sorted.drop(param.dummy, inplace=True, axis=1)
        
        # DELTA TIME
        serieDeltaTime = df_sorted[param.event_timestamp].diff()
        mask = df_sorted.person_id != df_sorted.person_id.shift(1)
        serieDeltaTime[mask] = timedelta(0)
        serieDeltaTime  = serieDeltaTime.astype('timedelta64[s]')
        df_delta_time = serieDeltaTime.to_frame(name='delta_time')
        df_sorted.drop(param.event_timestamp, inplace=True, axis=1)

        # CONCAT
        dfs_concat = (df_label_encoded,
                      df_dummy,
                      df_delta_time, 
                      df_sorted)
        df = pd.concat(dfs_concat, axis=1)
        df = df.fillna(-1)

        self.gpie.convey(source='dataframe', destination='gs', dataframe=df, data_name=param.df_preprocessed)

        logger.info("PREPROCESSING OK")

    def train(self, verbose=2):
        pass

        logger.info("TRAINING...")

        df = self.gpie.convey(source='gs', destination='dataframe', data_name=param.df_preprocessed, delete_in_gs=False)

        df = df[df.person_id.str.contains(param.regex['train'])]
        df.reset_index(drop=True, inplace=True)

        embed_shape = {}
        for feature in param.embedding:
            embed_shape[feature] = int(np.max(df[feature].values))+1

        person_ids_df = df.person_id.unique()
        _ = random.shuffle(person_ids_df)

        y = rnn.get_target(df, param.target)
        df.drop(param.target[0], inplace=True, axis=1)

        feat_seq = list(set(df.columns) - set([param.key, param.event_timestamp] + param.embedding))

        X_seq, dict_Xs_emb = rnn.shape_data(df, set(person_ids_df), feat_seq, param.embedding, param.max_len)

        model = rnn.model(feat_seq, param.max_len, embed_shape, param.emb_dim, self.env)

        model_inputs = [X_seq] + [value for key, value in dict_Xs_emb.items()]
        classes_weights = {0 : 1., 1 : y.shape[0]/y.sum()-1}
        logger.info(classes_weights)

        tbCallBack = TrainValTensorBoard(log_dir='gs://%s/%s/graph'%(self.bucket_id, param.bucket_path))

        model.fit(model_inputs, y, epochs=param.epochs, 
                                   batch_size=param.batch_size,
                                   class_weight=classes_weights,
                                   validation_split=0.15,
                                   callbacks=[tbCallBack],
                                   verbose=verbose)

        # SAVE MODEL
        model_json = model.to_json()
        with open("/tmp/%s.json"%param.model_name, "w") as json_file:
            json_file.write(model_json)
        model.save_weights("/tmp/%s.h5"%param.model_name)
        self.gpie.convey(source='local', destination='gs', data_name=param.model_name)

        predictions = model.predict(model_inputs, verbose=0)

        logger.info("BRIER SCORE:  %f"%brier_score_loss(y, predictions))

        fpr, tpr, thresholds = roc_curve(y, predictions)
        logger.info("AUC: %f"%auc(fpr, tpr))

        fig = display.get_roc_curve(y, predictions)
        fig_name = "%s_%s.png"%(param.model_name, 'train')
        fig.savefig("%s/%s"%(param.local_dir_path, fig_name), bbox_inches='tight')
        self.gpie.convey(source='local', destination='gs', data_name=fig_name)

        logger.info("TRAINING OK")


    def test(self):

        logger.info("TEST...")

        # LOAD MODEL
        self.gpie.convey(source='gs', destination='local', data_name=param.model_name+".json", delete_in_gs=False)
        self.gpie.convey(source='gs', destination='local', data_name=param.model_name+".h5", delete_in_gs=False)
        json_file = open(os.path.join(param.local_dir_path, param.model_name+".json"), 'r')
        model_json = json_file.read()
        json_file.close()
        model = model_from_json(model_json)
        model.load_weights(os.path.join(param.local_dir_path, param.model_name+".h5"))

        df = self.gpie.convey(source='gs', destination='dataframe', data_name=param.df_preprocessed, delete_in_gs=False)

        df = df[df.person_id.str.contains(param.regex['test'])]
        df.reset_index(drop=True, inplace=True)

        person_ids_df = df.person_id.unique()
        _ = random.shuffle(person_ids_df)

        y = rnn.get_target(df, param.target)
        df.drop(param.target[0], inplace=True, axis=1)

        feat_seq = list(set(df.columns) - set([param.key, param.event_timestamp] + param.embedding))

        X_seq, dict_Xs_emb = rnn.shape_data(df, set(person_ids_df), feat_seq, param.embedding, param.max_len)

        model_inputs = [X_seq] + [values for emb, values in dict_Xs_emb.items()]

        predictions = model.predict(model_inputs, verbose=0)

        logger.info("BRIER SCORE:  %f"%brier_score_loss(y, predictions))

        fpr, tpr, thresholds = roc_curve(y, predictions)
        logger.info("AUC: %f"%auc(fpr, tpr))

        fig = display.get_roc_curve(y, predictions)
        fig_name = "%s_%s.png"%(param.model_name, 'test')
        fig.savefig("%s/%s"%(param.local_dir_path, fig_name), bbox_inches='tight')
        self.gpie.convey(source='local', destination='gs', data_name=fig_name)

        logger.info("TEST OK")


class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir, **kwargs):
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="LSTM Network",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--project_account", dest="project_account", help="GCP Project Account")
    parser.add_argument("--bucket_id", dest="bucket_id", help="GCP Bucket ID")
    parser.add_argument("--task", dest="task", choices=['preprocess', 'train', 'test'], help="Task to perform")

    if len(sys.argv) < 7:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    m = MyModel(args.project_account, args.bucket_id, 'cloud')

    if args.task == 'preprocess':
        m.preprocess()

    elif args.task == 'train':
        m.train()

    elif args.task == 'test':
        m.test()

