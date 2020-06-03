from keras.layers import Input, Dense, Dropout, Activation, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.models import Model, Sequential
from keras import backend as K
from sklearn.utils import class_weight
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from get_data import GetData

import os
import json
import pandas as pd
import numpy as np
import mlflow.keras
import mlflow
import model_artifact as ma

with open('/home/tanlab-server/CellLines/L1000CDS_subset.json', 'r') as f:
    L = json.load(f)


def build_model(input_dim):
    model = Sequential()
    model.add(Dense(units=64, input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(units=32))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(units=32))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy')

    return model


for cell_line in ['PC3', 'MCF7', 'VCAP', 'A549', 'A375', 'HT29']:
    for descriptor in ['ecfp']:
        for target in ['up', 'dn']:
            mlflow.set_tracking_uri('http://193.140.108.166:5000')
            mlflow.set_experiment(cell_line + '_experiments_min_100')
            mlflow.start_run()
            mlflow.set_tag('Author', 'RIZA')

            if descriptor == 'ecfp':
                mlflow.log_param('useChirality', 'False')
                obj = GetData(L=L, cell_line=cell_line, descriptor=descriptor, n_fold=5, random_state=42,
                              random_genes=False, useChirality=False)
            elif descriptor == 'jtvae':
                mlflow.log_param('useChirality', 'False')
                obj = GetData(L=L, cell_line=cell_line, descriptor=descriptor, n_fold=5, random_state=42,
                              random_genes=False, csv_file='JTVAE_Representations.csv')
            else:
                mlflow.log_param('useChirality', 'False')
                obj = GetData(L=L, cell_line=cell_line, descriptor=descriptor, n_fold=5, random_state=42,
                              random_genes=False)

            mlflow.log_param('Model Type', 'Single-Task-NN')
            if target == 'up':
                mlflow.log_param('target', 'up_genes')
                x, y, folds = obj.get_up_genes()
            elif target == 'dn':
                mlflow.log_param('target', 'down_genes')
                x, y, folds = obj.get_down_genes()

            lst_y = []
            for i in range(978):
                lst_y.append(np.count_nonzero(y.iloc[:,i]))
            df = pd.DataFrame({'genes': lst_y})
            indexes = df[df['genes'] >= 100].index.values
            del lst_y, df
            y = y[y.columns[indexes]]
            x.drop(['SMILES'], axis=1, inplace=True)
            print(y)
            
            mlflow.log_param('cell_line', cell_line)
            mlflow.log_param('descriptor', descriptor)
            mlflow.log_param('n_fold', '5')
            mlflow.log_param('random_state', '42')
            mlflow.log_param('batch_size', '8')
            mlflow.log_param('epochs', '100')
            mlflow.log_param('optimizer', 'adam')
            mlflow.log_param('loss', 'binary_crossentropy')
            mlflow.log_param('early_stopping', 'True')
            
            for idx in range(len(indexes)):
                folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(x, y.iloc[:,idx]))
                scores = []
                for i, (trn, val) in enumerate(folds):
                    fold_scores = []
                    print(i+1, 'fold.')

                    trn_x = x.iloc[trn,:].values
                    trn_y = y.iloc[trn,idx].values
                    val_x = x.iloc[val,:].values
                    val_y = y.iloc[val,idx].values
                    
                    if descriptor == 'jtvae':
                        scaler = StandardScaler().fit(trn_x)
                        trn_x = scaler.transform(trn_x)
                        val_x = scaler.transform(val_x)

                    print('Train shapes:', trn_x.shape, trn_y.shape)
                    print('Test shapes:', val_x.shape, val_y.shape)

                    es = EarlyStopping(monitor='val_loss',
                                       min_delta=0,
                                       patience=3,
                                       verbose=0,
                                       mode='auto')

                    w = class_weight.compute_class_weight('balanced', np.unique(val_y), val_y)

                    with open('model_fold_' + str(i + 1) + '.txt', 'a') as file:
                        file.write('GENE_' + str(indexes[idx]) + '_WEIGHTS:\n\t')
                        file.write('{0: ' + str(w[0]) + '\n\t' +
                                   ' 1: ' + str(w[1]) + '}\n')

                        weights = {0: w[0],
                                   1: w[1]}

                    model = build_model(trn_x.shape[1])
                    model.fit(trn_x, trn_y, validation_data=(val_x, val_y), class_weight=weights,
                              batch_size=8, epochs=100, verbose=1, callbacks=[es])

                    mlflow.keras.log_model(model, 'model_fold_' + str(i + 1) + '_' + str(indexes[idx]))
                    print('Model saved in run %s' % mlflow.active_run().info.run_uuid)

                    pred = model.predict(val_x)
                    fold_scores.append(roc_auc_score(val_y, pred))
                    K.clear_session()
                    scores.append(fold_scores)

                scores = np.asanyarray(scores)
                mean_auc = np.mean(scores, axis=0)
                mlflow.log_metric('Gene_' + str(indexes[idx]) + '_AUC', mean_auc[0])
            
            for i in range(5):
                ma.log_file(file_name='model_fold_' + str(i + 1) + '.txt',
                            artifact_path=mlflow.get_artifact_uri(), delete_local=False)
                os.remove('model_fold_' + str(i + 1) + '.txt')

            ma.log_artifact(mlflow.get_artifact_uri())
            mlflow.end_run()
