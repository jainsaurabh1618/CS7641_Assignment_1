import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from time import process_time
import itertools
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder


def get_wine_data():
    rd = 42
    #data_1 = pd.read_csv('cancer').drop(columns=['id'])
    #data_1 = shuffle(data_1, random_state=rd)
    #data_1_feature = data_1.drop('class', axis=1)
    #data_1_label = data_1['class']

    wine = pd.read_csv('wine_red', sep=';')
    wine.loc[wine['quality'] <= 6, 'quality'] = 0
    #wine.loc[(wine['quality'] > 4) & (wine['quality'] <= 5), 'quality'] = 0
    wine.loc[wine['quality'] > 6, 'quality'] = 1
    data_1 = wine  # pd.concat([wine_low, wine_med, wine_high])
    data_1 = shuffle(data_1, random_state=rd)
    data_1_feature = data_1.drop('quality', axis=1)
    data_1_feature = data_1[['alcohol', 'sulphates', 'volatile acidity', 'citric acid', 'density']]
    #data_1_feature = StandardScaler().fit_transform(data_1_feature)
    data_1_label = data_1['quality']
    return data_1, data_1_feature, data_1_label

def get_cancer_data():
    rd = 42
    #data_1 = pd.read_csv('cancer').drop(columns=['id'])
    #data_1 = shuffle(data_1, random_state=rd)
    #data_1_feature = data_1.drop('class', axis=1)
    #data_1_label = data_1['class']

    cancer = pd.read_csv('../ml_bkp/cancer').drop('id', axis=1)
    cancer = shuffle(cancer, random_state=rd)
    data_1_feature = cancer.drop('class', axis=1)
    data_1_label = cancer['class']
    return cancer, data_1_feature, data_1_label

def get_energy_data():
    rd = 42

    energy = pd.read_csv('../ml_bkp/energy').drop('Y1', axis=1)
    energy.loc[energy['Y2'] <= 25, 'Y2'] = 0
    energy.loc[energy['Y2'] > 25, 'Y2'] = 1
    data_1 = shuffle(energy, random_state=rd)
    data_1_feature = data_1.drop('Y2', axis=1)
    data_1_label = data_1['Y2']
    return data_1, data_1_feature, data_1_label

def get_heart_data():
    rd = 42

    heart = pd.read_csv('heart')
    heart['ChestPainType'] = LabelEncoder().fit_transform(heart['ChestPainType'])
    heart['RestingECG'] = LabelEncoder().fit_transform(heart['RestingECG'])
    heart['ST_Slope'] = LabelEncoder().fit_transform(heart['ST_Slope'])
    heart['ExerciseAngina'] = LabelEncoder().fit_transform(heart['ExerciseAngina'])
    heart['Sex'] = LabelEncoder().fit_transform(heart['Sex'])

    data_1 = shuffle(heart, random_state=rd)
    data_1_feature = data_1.drop('HeartDisease', axis=1)
    data_1_label = data_1['HeartDisease']
    return data_1, data_1_feature, data_1_label


def plot_curve(data_x, train_score, cv_score, title, xlabel, ylabel, xticks, filename, label_list=None, log=False):
    if log:
        plt.semilogx(np.logspace(-3, 3, 10), np.mean(train_score, axis=1), 'o-', color='b', label='Train Score')
        plt.semilogx(np.logspace(-3, 3, 10), np.mean(cv_score, axis=1), 'o-', color='r', label='Cross Validation Score')
    else:
        plt.plot(data_x, np.mean(train_score, axis=1), 'o-', color='b', label='Train Score')
        plt.plot(data_x, np.mean(cv_score, axis=1), 'o-', color='r', label='Cross Validation Score')
    plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #if not log:
        #plt.xticks(xticks, label_list)
        #plt.xscale('log')
    plt.grid()
    plt.savefig('images/' + filename)
    plt.clf()


def plot_timing_curve(time_train, time_test, data_type):

    algo_names = list(time_train.keys())
    train_times = list(time_train.values())
    test_times = list(time_test.values())

    plt.barh(algo_names, train_times, height=.4)
    for index, value in enumerate(train_times):
        plt.text(value, index, str(value))

    plt.title('Train Time Comparison')
    plt.xlabel('Time Train(sec)')
    plt.ylabel('Algo Names')
    plt.savefig('images/' + data_type + '/train_time_comp.png')
    plt.clf()

    plt.barh(algo_names, test_times, height=.4)
    for index, value in enumerate(test_times):
        plt.text(value, index, str(value))

    plt.title('Test Time Comparison')
    plt.xlabel('Time Test(sec)')
    plt.ylabel('Algo Names')
    plt.savefig('images/' + data_type + '/test_time_comp.png')
    plt.clf()


def plot_loss_curve(data_loss, filename):

    plt.title('Loss Curve')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Loss')
    plt.plot(data_loss, color='b')
    plt.grid()
    plt.savefig('images/' + filename)
    plt.clf()


def plot_confusion_matrix(cm, classes, filename, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(2), range(2)):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True')
    plt.xlabel('Prediction')
    plt.savefig('images/' + filename)
    plt.clf()


def plot_performance_graphs():
    exp_data = pd.read_csv('exp_output.csv')
    for data_t in ['Wine', 'Heart']:
        for met in ['Accuracy', 'f1_mean']:
            tune_dt = round((exp_data.loc[(exp_data['Algo'] == 'dt') & (exp_data['Run_Type'] == 'Tune_'+data_t)]).iloc[0][met], 3)
            tune_knn = round((exp_data.loc[(exp_data['Algo'] == 'knn') & (exp_data['Run_Type'] == 'Tune_'+data_t)]).iloc[0][met], 3)
            tune_boost = round((exp_data.loc[(exp_data['Algo'] == 'boost') & (exp_data['Run_Type'] == 'Tune_'+data_t)]).iloc[0][met], 3)
            tune_nn = round((exp_data.loc[(exp_data['Algo'] == 'nn') & (exp_data['Run_Type'] == 'Tune_'+data_t)]).iloc[0][met], 3)
            tune_svm = round((exp_data.loc[(exp_data['Algo'] == 'svm') & (exp_data['Run_Type'] == 'Tune_'+data_t)]).iloc[0][met], 3)

            plt.barh(['dt', 'knn', 'boost', 'nn', 'svm'], [tune_dt, tune_knn, tune_boost, tune_nn, tune_svm], height=.4)
            for index, value in enumerate([tune_dt, tune_knn, tune_boost, tune_nn, tune_svm]):
                plt.text(value, index, str(value))

            plt.title(met)
            plt.xlabel('Score')
            plt.savefig('images/'+data_t.lower()+'/' + met.lower() + '_perf.png')
            plt.clf()

    for algo in ['dt', 'knn', 'boost', 'nn', 'svm']:
        algo_data = exp_data[exp_data['Algo'] == algo]
        data_x = ['Bench_Acc', 'Tune_Acc', 'Bench_f1', 'Tune_f1']

        # Wine data
        bench_acc = round(algo_data.loc[algo_data.Run_Type == 'Bench_Wine', 'Accuracy'].values[0], 3)
        tune_acc = round(algo_data.loc[algo_data.Run_Type == 'Tune_Wine', 'Accuracy'].values[0], 3)
        bench_f1 = round(algo_data.loc[algo_data.Run_Type == 'Bench_Wine', 'f1_mean'].values[0], 3)
        tune_f1 = round(algo_data.loc[algo_data.Run_Type == 'Tune_Wine', 'f1_mean'].values[0], 3)

        plt.barh(data_x, [bench_acc, tune_acc, bench_f1, tune_f1], height=.4)
        for index, value in enumerate([bench_acc, tune_acc, bench_f1, tune_f1]):
            plt.text(value, index, str(value))

        plt.title('Accuracy & f1_score(mean) - Bench vs Tune')
        plt.xlabel('Score')
        plt.savefig('images/wine/'+algo+'_perf.png')
        plt.clf()

        # Heart data
        bench_acc = round(algo_data.loc[algo_data.Run_Type == 'Bench_Heart', 'Accuracy'].values[0], 3)
        tune_acc = round(algo_data.loc[algo_data.Run_Type == 'Tune_Heart', 'Accuracy'].values[0], 3)
        bench_f1 = round(algo_data.loc[algo_data.Run_Type == 'Bench_Heart', 'f1_mean'].values[0], 3)
        tune_f1 = round(algo_data.loc[algo_data.Run_Type == 'Tune_Heart', 'f1_mean'].values[0], 3)

        plt.barh(data_x, [bench_acc, tune_acc, bench_f1, tune_f1], height=.4)
        for index, value in enumerate([bench_acc, tune_acc, bench_f1, tune_f1]):
            plt.text(value, index, str(value))

        plt.title('Accuracy & f1_score(mean) - Bench vs Tune')
        plt.xlabel('Score')
        plt.savefig('images/heart/' + algo + '_perf.png')
        plt.clf()


