import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score, f1_score
from ml import helper
from sklearn.metrics import plot_confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import warnings
import time

warnings.filterwarnings("ignore")

rd = 42
time_train = dict()
time_test = dict()

def getData(data_type):
    if data_type == 'wine':
        data_1, data_1_feature, data_1_label = helper.get_wine_data()
    if data_type == 'cancer':
        data_1, data_1_feature, data_1_label = helper.get_cancer_data()
    if data_type == 'energy':
        data_1, data_1_feature, data_1_label = helper.get_energy_data()
    if data_type == 'heart':
        data_1, data_1_feature, data_1_label = helper.get_heart_data()
    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(data_1_feature, data_1_label, test_size=0.3,
                                                        random_state=rd)
    return X_train, X_test, y_train, y_test


def runExp(fold, res_df, run_type, smt=False, cross_validate=False, v_curve=False, l_curve=False, data_type=None):
    X_train, X_test, y_train, y_test = getData(data_type)

    ccp_alphas = list(DecisionTreeClassifier(random_state=rd).cost_complexity_pruning_path(X_train, y_train,
                                                                                           sample_weight=None)
                      .ccp_alphas)
    param_grid_dict = {'dt': {'max_depth': np.arange(1, 20), 'ccp_alpha': np.linspace(ccp_alphas[1], max(ccp_alphas), 10)},
                       'knn': {'n_neighbors': np.arange(1, 20, 2), 'metric': ['euclidean', 'manhattan']},
                       'boost': {'n_estimators': np.arange(1, 500, 100), 'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.5, 1.0]},
                       'nn': {'learning_rate_init': [0.001, 0.01, 0.1, 0.2, 0.5, 1.0], 'hidden_layer_sizes': [(50,),(100,),(50,50),(100,100), (50,50,50), (100,100,100)]},
                       'svm': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf', 'poly', 'sigmoid']}
                       }

    algo_objs = {'dt': DecisionTreeClassifier(random_state=rd),
                 'knn': KNeighborsClassifier(),
                 'boost': AdaBoostClassifier(random_state=rd),
                 'nn': MLPClassifier(max_iter=1000, random_state=rd),
                 'svm': SVC(random_state=rd)}

    plot_params = {'dt_ccp_alpha': {'xticks': np.linspace(min(ccp_alphas), max(ccp_alphas), 10)},
                   'knn_metric': {'xticks': np.arange(2), 'label_list': ['euclidean', 'manhattan']},
                   'boost_learning_rate': {'xticks': [0.001, 0.01, 0.1, 0.2, 0.5, 1.0]},
                   'svm_C': {'xticks': [0.1, 1, 10, 100]},
                   'svm_kernel': {'xticks': np.arange(4), 'label_list': ['linear', 'rbf', 'poly', 'sigmoid']},
                   #'svm_gamma': {'xticks': np.arange(2), 'label_list': ['scale', 'auto']},
                   'nn_learning_rate_init': {'xticks': [0.001, 0.01, 0.1, 0.2, 0.5, 1.0]},
                   'nn_hidden_layer_sizes': {'data_x': ['(50,)', '(100,)', '(50,50)', '(100,100)', '(50,50,50)', '(100,100,100)']},
                   }

    if not cross_validate:
        for algo in param_grid_dict:
            cls = algo_objs.get(algo)
            if smt:
                sm = SMOTE(random_state=rd)
                X_train, y_train = sm.fit_resample(X_train, y_train)
            print('################# ', algo, ' #################')
            cls.fit(X_train, y_train)
            y_pred = cls.predict(X_test)
            # print(classification_report(y_test, y_pred))
            f1 = f1_score(y_test, y_pred, average=None)
            print('f1_score :', f1)
            accuracy = accuracy_score(y_test, y_pred)
            print('accuracy :', round(accuracy, 3))
            res_df = res_df.append({'Algo': algo, 'Run_Type': run_type, 'Accuracy': round(accuracy, 3),
                                    'f1_0': round(f1[0], 3), 'f1_1': round(f1[1], 3)}, ignore_index=True)
            print('\n')

    else:
        for algo in param_grid_dict:
            if smt:
                sm = SMOTE(random_state=rd)
                X_train, y_train = sm.fit_resample(X_train, y_train)
            cls = algo_objs.get(algo)

            if v_curve:
                for param in param_grid_dict.get(algo):
                    # Validation curve
                    train_scores_param_1, test_scores_param_1 = validation_curve(
                        cls, X_train, y_train, param_range=param_grid_dict.get(algo).get(param),
                        param_name=param, cv=fold, scoring='accuracy')

                    xticks = param_grid_dict.get(algo).get(param)
                    label_list_val = None
                    log_val = False

                    plot_param_values = plot_params.get(algo + '_' + param)
                    data_x = param_grid_dict.get(algo).get(param)
                    if plot_param_values:
                        if plot_param_values.get('xticks') is not None:
                            xticks = plot_param_values.get('xticks')
                        if plot_param_values.get('label_list') is not None:
                            label_list_val = plot_param_values.get('label_list')
                        if plot_param_values.get('log') is not None:
                            log_val = plot_param_values.get('log')
                        if plot_param_values.get('data_x') is not None:
                            data_x = plot_param_values.get('data_x')

                    helper.plot_curve(data_x, train_scores_param_1, test_scores_param_1,
                                      'Validation Curve for ' + param + '(' + algo + ')', param, 'Score',
                                      xticks, data_type + '/' + algo + '_validation_curve_' + param + '.png', label_list=label_list_val,
                                      log=log_val)

            cls_best = GridSearchCV(cls, param_grid=param_grid_dict.get(algo), cv=fold)
            print('################# ', algo, ' - with CV #################')
            print('Best Param')
            start_time = time.time()
            grid_res = cls_best.fit(X_train, y_train)
            end_time = time.time()
            time_train[algo] = round(end_time - start_time, 5)
            print(cls_best.best_params_)
            start_time = time.time()
            y_pred = grid_res.best_estimator_.predict(X_test)
            end_time = time.time()
            time_test[algo] = round(end_time - start_time, 5)
            plot_confusion_matrix(cls_best, X_test, y_test)
            plt.savefig('images/' + data_type +'/' + algo + '_confusion_matrix.png')
            plt.clf()
            f1 = f1_score(y_test, y_pred, average=None)
            print('f1_score :', f1)
            accuracy = accuracy_score(y_test, y_pred)
            print('accuracy :', round(accuracy, 3))
            res_df = res_df.append({'Algo': algo, 'Run_Type': run_type, 'Accuracy': round(accuracy, 3),
                                    'f1_0': round(f1[0], 3), 'f1_1': round(f1[1], 3)}, ignore_index=True)

            if algo=='nn':
                loss_c = grid_res.best_estimator_.loss_curve_
                helper.plot_loss_curve(loss_c, data_type + '/' + algo + '_loss_curve.png')
            if l_curve:
                # Learning curve with best params
                train_sizes, train_scores, test_scores = learning_curve(grid_res.best_estimator_,
                                                                        X_train, y_train,
                                                                        train_sizes=np.linspace(0.1, 1.0, 10),
                                                                        cv=fold, scoring='accuracy')
                helper.plot_curve(np.linspace(0.1, 1.0, 10) * 100, train_scores, test_scores,
                                  'Learning Curve (' + algo + ')', 'Percentage of Training Examples', 'Score',
                                  np.linspace(0.1, 1.0, 10) * 100, data_type + '/' + algo + '_learning_curve.png')
        helper.plot_timing_curve(time_train, time_test, data_type)
    return res_df


def analysis_run(d_type):
    res_df = pd.DataFrame(columns=['Algo', 'Run_Type', 'Accuracy', 'f1_0', 'f1_1'])
    run_type = ''
    print('Normal Run- no smote')
    run_type = 'Normal-no_smote'
    res_df = runExp(None, res_df, run_type, smt=False, cross_validate=False, v_curve=False, l_curve=False,
                    data_type=d_type)
    run_type = 'Normal-smote'
    res_df = runExp(None, res_df, run_type, smt=True, cross_validate=False, v_curve=False, l_curve=False,
                    data_type=d_type)

    print('KFold')
    run_type = 'Kfold-no_smote'
    fold = KFold(n_splits=10)
    res_df = runExp(fold, res_df, run_type, smt=False, cross_validate=True, v_curve=False, l_curve=False,
                    data_type=d_type)
    print('KFold-smote')
    run_type = 'Kfold-smote'
    fold = KFold(n_splits=10)
    res_df = runExp(fold, res_df, run_type, smt=True, cross_validate=True, v_curve=False, l_curve=False,
                    data_type=d_type)

    print('StratifyKFold')
    run_type = 'Stratify-no_smote'
    fold = StratifiedKFold(n_splits=10)
    res_df = runExp(fold, res_df, run_type, smt=False, cross_validate=True, v_curve=False, l_curve=False,
                    data_type=d_type)
    print('StratifyKFold - smote')
    run_type = 'Stratify-smote'
    fold = StratifiedKFold(n_splits=10)
    res_df = runExp(fold, res_df, run_type, smt=True, cross_validate=True, v_curve=False, l_curve=False,
                    data_type=d_type)
    res_df.to_csv('analysis_output'+d_type+'.csv')
    print(res_df)


def final_run():
    res_df = pd.DataFrame(columns=['Algo', 'Run_Type', 'Accuracy', 'f1_0', 'f1_1'])

    run_type = 'Bench_Heart'
    res_df = runExp(None, res_df, run_type, smt=False, cross_validate=False, v_curve=False, l_curve=False,
                    data_type='heart')

    run_type = 'Tune_Heart'
    fold = KFold(n_splits=10)
    res_df = runExp(fold, res_df, run_type, smt=False, cross_validate=True, v_curve=True, l_curve=True,
                    data_type='heart')

    ##########################################################################################

    run_type = 'Bench_Wine'
    res_df = runExp(None, res_df, run_type, smt=False, cross_validate=False, v_curve=False, l_curve=False,
                    data_type='wine')
    run_type = 'Tune_Wine'
    fold = StratifiedKFold(n_splits=10)
    res_df = runExp(fold, res_df, run_type, smt=True, cross_validate=True, v_curve=True, l_curve=True,
                    data_type='wine')

    res_df['f1_mean'] = (res_df['f1_0'] + res_df['f1_1']) / 2
    print(res_df)
    res_df.to_csv('exp_output.csv')


if __name__ == "__main__":
    final_run()
    analysis_run('wine')
    analysis_run('heart')
    helper.plot_performance_graphs()



