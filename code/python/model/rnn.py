import logging
import numpy as np
import pandas as pd

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, CuDNNLSTM, LSTM, Embedding, BatchNormalization, Activation

logger = logging.getLogger(__name__)


def model(feat_seq, max_len, embed_shape, emb_dim, env):

    inputs = []
    models = []

    input_ = Input(shape=(max_len, len(feat_seq)), dtype='float32', name='input_1')
    lstm_ = CuDNNLSTM(30, name='cudnnlstm_1')(input_) if env=='cloud' else LSTM(30, name='lstm_1')(input_)

    inputs.append(input_)
    models.append(lstm_)

    for key, value in embed_shape.items():
        input_ = Input(shape=(max_len,), dtype='float32', name='input_%s'%key)
        inputs.append(input_)
        embedding_ = Embedding(value, emb_dim, input_length=max_len, name='embedding_%s'%key)(input_)
        lstm_ = CuDNNLSTM(30, name='cudnnlstm_%s'%key)(embedding_) if env=='cloud' else LSTM(30, name='lstm_%s'%key)(embedding_)
        models.append(lstm_)

    models_merged = Concatenate(axis=1)(models)

    output = Dense(1, use_bias=False)(models_merged)
    output = BatchNormalization()(output)
    output = Activation("sigmoid")(output)

    model_params = {'loss':'binary_crossentropy', 
                    'optimizer':'adam', 
                    'metrics':['acc']}
    model = Model(inputs=inputs, outputs=output)
    model.compile(**model_params)

    logger.info(model.summary())

    return model


def pad_sequence_matrices(X, maxlen=None, padding='pre', truncating='pre'):

    newX = []

    for matrix in X:
        
        if matrix.shape[0] == maxlen:
            newX.append(matrix)
        
        if matrix.shape[0] > maxlen:
            nb_rows_removed = matrix.shape[0] - maxlen
            if(truncating == 'pre'):     
                newX.append(matrix[nb_rows_removed:, :])
            if(truncating == 'post'):
                newX.append(matrix[:-nb_rows_removed, :])
                
        if matrix.shape[0] < maxlen:
            nb_rows_added = maxlen - matrix.shape[0]
            padded_mat = np.zeros(shape=(nb_rows_added, matrix.shape[1]))
            if(padding == 'pre'):
                newX.append(np.append(padded_mat, matrix, axis=0))
            if(padding == 'post'):
                newX.append(np.append(matrix, padded_mat, axis=0))
    
    return np.asarray(newX)


def shape_data(df, person_ids, feat_seq, embedding, maxlen):

    valid_X_rows = df['person_id'].isin(person_ids)
    df_X_valid = df[valid_X_rows]
    
    unique_person_ids, index_cut_pids = np.unique(df_X_valid.person_id.values, return_index=True)
    
    dict_Xs_emb = {}
    for feature_emb in embedding:
        feat_emb_to_cut = df_X_valid[feature_emb].values
        new_shape = (feat_emb_to_cut.shape[0],1)
        feat_emb_to_cut = feat_emb_to_cut.reshape(new_shape)
        dict_Xs_emb[feature_emb] = np.split(feat_emb_to_cut, index_cut_pids[1:], axis=0)
        dict_Xs_emb[feature_emb] = np.asarray(dict_Xs_emb[feature_emb])

    sequences = np.split(df_X_valid[feat_seq].values, index_cut_pids[1:], axis=0)
    dict_X_seq_data = {'person_id': unique_person_ids,'sequence': sequences}
    df_X_seq = pd.DataFrame(data=dict_X_seq_data)
    X_seq = df_X_seq['sequence'].values
       
    X_seq = pad_sequence_matrices(X_seq, maxlen=maxlen)
    for key in dict_Xs_emb:
        assert 0 not in dict_Xs_emb[key]
        dict_Xs_emb[key] = pad_sequence_matrices(dict_Xs_emb[key], maxlen=maxlen)

    for key, values in dict_Xs_emb.items():
        new_shape = (values.shape[0],values.shape[1])
        dict_Xs_emb[key] = values.reshape(new_shape)

    return X_seq, dict_Xs_emb


def get_target(df, target):

    y = np.array([1 if m>target[1] else 0 for m in df.groupby('person_id')[target[0]].sum()])
    return y
