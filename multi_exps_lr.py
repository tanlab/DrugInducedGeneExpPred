from keras.layers import Input, Dense, Dropout, Activation, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.models import Model
from keras import backend as K
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.multioutput import ClassifierChain
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from get_data import GetData

import os
import json
import pandas as pd
import numpy as np
import mlflow.sklearn
import mlflow
import model_artifact as ma


with open('/home/tanlab-server/CellLines/L1000CDS_subset.json', 'r') as f:
    L = json.load(f)


def build_model(input_dim, output_dim):
    base_lr = LogisticRegression()
    model = ClassifierChain(base_lr, order='random', random_state=42)

    return model


for descriptor in ['ecfp']:
    for cell_line in ['VCAP', 'A549', 'A375', 'PC3', 'MCF7', 'HT29']:
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

            mlflow.log_param('Model Type', 'Multi-Task-LR')
            if target == 'up':
                mlflow.log_param('target', 'up_genes')
                x, y, folds = obj.get_up_genes()
            elif target == 'dn':
                mlflow.log_param('target', 'down_genes')
                x, y, folds = obj.get_down_genes()

            lst_y = []
            for i in range(978):
                lst_y.append(np.count_nonzero(y.iloc[:, i]))
            df = pd.DataFrame({'genes': lst_y})
            indexes = df[df['genes'] >= 100].index.values
            del lst_y, df
            y = y[y.columns[indexes]]
            x.drop(['SMILES'], axis=1, inplace=True)
            folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(x, y.iloc[:, 0]))
            print(y)

            mlflow.log_param('cell_line', cell_line)
            mlflow.log_param('descriptor', descriptor)
            mlflow.log_param('n_fold', '5')
            mlflow.log_param('random_state', '42')
            mlflow.log_param('base_model', 'LogisticRegression')
            mlflow.log_param('mt-wrapper', 'ClassifierChain')
            mlflow.log_param('loss', 'binary_crossentropy')

            scores = []
            for i, (trn, val) in enumerate(folds):
                fold_scores = []
                print(i + 1, 'fold.')

                trn_x = x.iloc[trn,:].values
                trn_y = y.iloc[trn,:].values
                val_x = x.iloc[val,:].values
                val_y = y.iloc[val,:].values

                if descriptor == 'jtvae':
                    scaler = StandardScaler().fit(trn_x)
                    trn_x = scaler.transform(trn_x)
                    val_x = scaler.transform(val_x)

                trn_y_list = []
                val_y_list = []

                for j in range(trn_y.shape[1]):
                    trn_y_list.append(trn_y[:,j])
                    val_y_list.append(val_y[:,j])

                print('Train shapes:', trn_x.shape, trn_y.shape)
                print('Test shapes:', val_x.shape, val_y.shape)

                
                model = build_model(trn_x.shape[1], trn_y.shape[1])
                model.fit(trn_x, trn_y)

                mlflow.sklearn.log_model(model, 'model_fold_' + str(i + 1))
                print('Model saved in run %s' % mlflow.active_run().info.run_uuid)

                for i in range(trn_y.shape[1]):
                    if trn_y.shape[1] == 1:
                        pred = model.predict(val_x)
                    else:
                        pred = model.predict(val_x)[:,i]
                    fold_scores.append(roc_auc_score(val_y_list[i], pred))

                scores.append(fold_scores)

            scores = np.asanyarray(scores)
            mean_auc = np.mean(scores, axis=0)

            for idx in range(len(indexes)):
                mlflow.log_metric('Gene_' + str(indexes[idx]) + '_AUC', mean_auc[idx])

            ma.log_artifact(mlflow.get_artifact_uri())
            mlflow.end_run()
