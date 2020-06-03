from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import class_weight
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
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

ecfp_control = 0


for cell_line in ['MCF7', 'A549', 'HT29']:
    for target in ['up', 'dn']:
        for descriptor in ['ecfp', 'ecfp']:
            mlflow.set_tracking_uri('http://193.140.108.166:5000')
            mlflow.set_experiment(cell_line + '_experiments_min_100')
            mlflow.start_run()
            mlflow.set_tag('Author', 'RIZA')

            if descriptor == 'ecfp':
                if ecfp_control == 0:
                    mlflow.log_param('useChirality', 'True')
                    obj = GetData(L=L, cell_line=cell_line, descriptor=descriptor, n_fold=5, random_state=42,
                                  random_genes=False, useChirality=True)
                    ecfp_control = 1
                else:
                    mlflow.log_param('useChirality', 'False')
                    obj = GetData(L=L, cell_line=cell_line, descriptor=descriptor, n_fold=5, random_state=42,
                                  random_genes=False, useChirality=False)
                    ecfp_control = 0
            elif descriptor == 'jtvae':
                mlflow.log_param('useChirality', 'False')
                obj = GetData(L=L, cell_line=cell_line, descriptor=descriptor, n_fold=5, random_state=42,
                              random_genes=False, csv_file='JTVAE_Representations.csv')
            else:
                mlflow.log_param('useChirality', 'False')
                obj = GetData(L=L, cell_line=cell_line, descriptor=descriptor, n_fold=5, random_state=42,
                              random_genes=False)

            mlflow.log_param('Model Type', 'Single-Task-RF')
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
            mlflow.log_param('base_model', 'RandomForest')
            mlflow.log_param('loss', 'binary_crossentropy')
            
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

                    w = class_weight.compute_class_weight('balanced', np.unique(val_y), val_y)

                    with open('model_fold_' + str(i + 1) + '.txt', 'a') as file:
                        file.write('GENE_' + str(indexes[idx]) + '_WEIGHTS:\n\t')
                        file.write('{0: ' + str(w[0]) + '\n\t' +
                                   ' 1: ' + str(w[1]) + '}\n')

                        weights = {0: w[0],
                                   1: w[1]}
                        
                    model = RandomForestClassifier(n_estimators=150, class_weight=weights)
                    model.fit(trn_x, trn_y)

                    mlflow.sklearn.log_model(model, 'model_fold_' + str(i + 1) + '_' + str(indexes[idx]))
                    print('Model saved in run %s' % mlflow.active_run().info.run_uuid)

                    pred = model.predict_proba(val_x)[:,1]
                    fold_scores.append(roc_auc_score(val_y, pred))
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
