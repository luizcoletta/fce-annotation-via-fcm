# ------------------------------------------------------------------------------
# REQUIRED PACKAGES
# ------------------------------------------------------------------------------
import numpy as np # Pacote para computação científica permitindo manipulação matricial, operações algébricas e estatísticas vetorizadas (https://www.numpy.org/)
import scipy as sp # Pacote dependente do NumPy voltado para estatística, otimização, álgebra linear, transformações de Fourier, processamento de sinais (https://scipy.org/scipylib/)
import pandas as pd # Pacote para análise e manipulação flexível de dados com estruturas similares aos Data Frames da linguagem R (https://pandas.pydata.org/)
import matplotlib.pyplot as plt # Pacote para visualização de dados gerando uma infinidade de diagramas e gráficos (https://matplotlib.org/)
import seaborn as sns # Biblioteca de visualização baseada no Matplotlib para gerar gráficos estatísticos mais atraentes (https://seaborn.pydata.org).
import plotly.express as px # Biblioteca para gerar visualizações mais sofisticadas que o Matplotlib (https://plotly.com/).
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import Isomap
from sklearn import svm
from sklearn import metrics
from scipy.spatial import distance
from itertools import combinations
import os
import os.path


# ------------------------------------------------------------------------------
# MAIN FUNCTIONS
# ------------------------------------------------------------------------------

def get_batch_data(train_data_path, test_data_path, class_index, join_data, size_batch, iter):
    col1 = np.array([list(range(1, 9))])
    # col2 = np.array([list(range(10,22))])
    col3 = np.array([np.hstack([[0] * 5, [1] * 3])])

    # train_data = np.concatenate((col1, col2), axis=0)
    train_data = np.concatenate((col1, col3), axis=0).T
    train_data = pd.DataFrame(train_data)

    col1 = np.array([list(range(9, 13))])
    # col2 = np.array([list(range(10,16))])
    col3 = np.array([np.hstack([[0] * 3, [1] * 1])])

    # test_data = np.concatenate((col1, col2), axis=0)
    test_data = np.concatenate((col1, col3), axis=0).T
    test_data = pd.DataFrame(test_data)

    df_training = []
    train = []
    train_labels = []
    if train_data_path:
        df_training = pd.read_csv(train_data_path)  # , header=None)
        # print(df_training)
        # df_training = train_data
        feat_index = list(range(df_training.shape[1]))
        feat_index.remove(class_index)
        train = df_training.iloc[:, feat_index].values
        train_labels = df_training.iloc[:, class_index].values

    df_test = []
    test = []
    test_labels = []
    if test_data_path:
        df_test = pd.read_csv(test_data_path)  # , header=None)
        # print(df_test)
        # df_test = test_data
        feat_index = list(range(df_test.shape[1]))
        feat_index.remove(class_index)
        test = df_test.iloc[:, feat_index].values
        test_labels = df_test.iloc[:, class_index].values

    if join_data:
        data = np.concatenate([train, test])
        data_labels = np.concatenate([train_labels, test_labels])
    else:
        data = train
        data_labels = train_labels
    # print(data.shape)
    # print(data_labels)

    num_objects = data.shape[0]

    folds = int(num_objects / size_batch)

    # print(folds)

    '''skf = StratifiedKFold(n_splits=folds)
    for train, test in skf.split(data, data_labels):
    print(train)
    print(test)
    #print('train -  {}   |   test -  {}'.format(np.bincount(y[train]), np.bincount(y[test])))'''

    i = 1
    test_data_fold = []
    test_labels_fold = []
    train_data_fold = []
    train_labels_fold = []
    # X, y = data, data_labels
    skf = StratifiedKFold(n_splits=folds, random_state=None, shuffle=False)

    for train_index, test_index in skf.split(data, data_labels):
        if (iter == i):
            # print ("\nIteração = ", i)
            # print("TEST-DATA:", data[test_index], "\nTEST-LABELS:", data_labels[test_index])
            # print("TRAIN-DATA:", data[train_index], "\nTRAIN-LABELS:", data_labels[train_index])
            test_data_fold = data[test_index]
            test_labels_fold = data_labels[test_index]
            train_data_fold = data[train_index]
            train_labels_fold = data_labels[train_index]
        i = i + 1

    '''while (train_index, test_index in skf.split(X, y)) and (iter<i):
        i= i+1
        print ("\nIteração= ", i)
        print("TRAIN:", data[train_index], "\nTEST:", data_labels[test_index])
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        #print("X_Test:", X_test)
        #print("y_Test:", y_test)
        #print (X_test.shape[0])
        #print (y_test.shape[0])
        print (data[train_index].shape[0])
        break'''

    '''for train_index, test_index in skf.split(X, y):
        i= i+1
        print ("\nIteração= ", i)
        print("TRAIN:", data[train_index], "\nTEST:", data_labels[test_index])
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        #print("X_Test:", X_test)
        #print("y_Test:", y_test)
        #print (X_test.shape[0])
        #print (y_test.shape[0])
        print (data[train_index].shape[0])'''

    return test_data_fold, test_labels_fold, train_data_fold, train_labels_fold


### ====================================================================================================================
### Functions for the IC-EDS approach based on Coletta et al. 2019
### --------------------------------------------------------------------------------------------------------------------
def svmClassification(train, train_labels, test):
    SVM = svm.SVC(tol=1.94, probability=True)
    SVM.fit(train, train_labels)
    probs = SVM.predict_proba(test)
    pred = SVM.predict(test)
    # print(np.around(probs,2))
    return [probs, pred]


# def hyperparametersTuning(...):
# TO DO
# https://scikit-learn.org/stable/modules/grid_search.html
# https://www.kaggle.com/udaysa/svm-with-scikit-learn-svm-with-parameter-tuning


### Generating subset of features randomly
def features_subset(tot_features, num_subsetfeat):
    features_list = list(range(0, tot_features))
    comb = combinations(features_list, num_subsetfeat)
    # perm = permutations(features_list, num_subsetfeat) # order matters, e.g.: (0,1) <> (1,0)
    subsetfeat_list = []
    for i in list(comb):
        subsetfeat_list.append(i)
    # ----------------------------------------------------------------------------------------------
    ### Versão baseada em sortear permutação e realizar um cortes na proporcao de 'size_subsetfeat'
    # features_list = list(range(0, num_features))
    #### INEFICIENTE, LISTA COM TODAS AS PERMUTAÇÕES POSSÍVEIS DE N FEATURES!!!
    # feat_perm = [p for p in permutations(features_list)]
    # size_feat_perm = len(list(feat_perm))
    # num_feat_perm = randrange(size_feat_perm)
    # sel_feat_perm = list(feat_perm)[num_feat_perm]
    # print(sel_feat_perm)
    # subsetfeat_list = []
    # int_size_subsetfeat = floor(len(sel_feat_perm) * size_subsetfeat)
    # for n in range(int_size_subsetfeat, num_features + 1, int_size_subsetfeat):
    #    subsetfeat_list.append(sel_feat_perm[n - int_size_subsetfeat:n])
    #    # print(sel_feat_perm[n-int_size_subsetfeat:n])
    # ----------------------------------------------------------------------------------------------
    return subsetfeat_list


def clusterEnsemble(data):
    ssfeat_list = features_subset(data.shape[1], 2)
    max_k = int(len(data) ** (1 / 3))  # equal to cubic root # int(math.sqrt(len(apat_iceds_norm)))
    num_init = 5  # 20
    range_n_clusters = list(range(2, max_k))

    silhouette_list = []
    clusterers_list = []
    cluslabels_list = []
    nuclusters_list = []

    matDist = np.array(euclidean_distances(data, data))

    for n_size_ssfeat in range(int(len(ssfeat_list))):

        # Subconjunto de features
        subset_feat = ssfeat_list[n_size_ssfeat]
        X = data[:, subset_feat]

        best_silhouette_avg = -1.0
        best_clusterer = []
        best_cluster_labels = []
        best_num_clusters = -1

        for n_clusters in range_n_clusters:
            for n_init in range(num_init):

                # Initialize the clusterer with n_clusters value and a random generator
                # seed of 10 for reproducibility.
                clusterer = KMeans(n_clusters=n_clusters, init='random')
                cluster_labels = clusterer.fit_predict(X)

                # The silhouette_score gives the average value for all the samples.
                # This gives a perspective into the density and separation of the formed clusters
                silhouette_avg = silhouette_score(X, cluster_labels)

                if (silhouette_avg > best_silhouette_avg):
                    best_silhouette_avg = silhouette_avg
                    best_clusterer = clusterer
                    best_cluster_labels = cluster_labels
                    best_num_clusters = n_clusters

                # print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
                # clusterer_plots(X, cluster_labels, n_clusters, clusterer)

        silhouette_list.append(best_silhouette_avg)
        clusterers_list.append(best_clusterer)
        cluslabels_list.append(best_cluster_labels)  ### vai usar para gera a matriz de similaridades abaixo
        nuclusters_list.append(best_num_clusters)

    ############# CONSENSO ###################
    cluslabels_list = np.array(cluslabels_list)
    caMatrix = np.array([[0] * cluslabels_list.shape[1]] * cluslabels_list.shape[1])

    for i in range(cluslabels_list.shape[
                       0]):  # for (int i = 0; i < cluEnsemble.length; i++) {  ### TAMANHO DA LISTA cluslabels_list
        for j in range(cluslabels_list.shape[
                           1]):  # for (int j = 0; j < data.numInstances(); j++) { ### len(cluslabels_list[0])
            for k in range(cluslabels_list.shape[
                               1]):  # for (int k = 0; k < data.numInstances(); k++) { ### len(cluslabels_list[0])
                if cluslabels_list[i][j] == cluslabels_list[i][k]:  ######## cluslabels_list[i][j] == cluslabels_list[i][k]
                    caMatrix[j][k] += 1
                if i == cluslabels_list.shape[0] - 1:
                    caMatrix[j][k] = caMatrix[j][k] / cluslabels_list.shape[0]  ### TAMANHO DA LISTA cluslabels_list
    # print("Best Silhoutte =", silhouette_list, " Number of Clusters =", nuclusters_list)
    return [silhouette_list, clusterers_list, cluslabels_list, nuclusters_list, caMatrix, matDist]


def remove_class(hidden_class, train, train_labels):
    train_labels.columns = ['Class']
    labeled_data = pd.concat([train, train_labels], axis=1, sort=False)
    labeled_data = labeled_data[labeled_data.Class != hidden_class]

    t = labeled_data.iloc[:, :-1]
    tl = labeled_data.iloc[:, -1:]

    tl.columns = [0]

    return [t, tl]


def increment_training_set(sel_objects, train, train_labels, test, test_labels):
    test = pd.DataFrame(test)
    test_labels = pd.DataFrame(test_labels)
    objects = test.iloc[sel_objects, :]
    objects_labels = test_labels.iloc[sel_objects, :]
    # print("Selected Objects Classes: " + str(objects_labels.values.ravel()))
    train = pd.DataFrame(train)
    train_labels = pd.DataFrame(train_labels)
    train.columns = objects.columns
    train_labels.columns = objects_labels.columns
    tr = pd.concat([train, objects], axis=0)
    trl = pd.concat([train_labels, objects_labels], axis=0)
    te = test.drop(test.index[sel_objects])
    tel = test_labels.drop(test_labels.index[sel_objects])
    return [tr.to_numpy(), trl.to_numpy(), te.to_numpy(), tel.to_numpy()]


def reduce_matrix(sel_objects, SSet):
    sim_mat = np.delete(SSet, np.s_[sel_objects], axis=0)
    sim_mat = np.delete(sim_mat, np.s_[sel_objects], axis=1)
    return sim_mat


def calc_class_entropy(p):
    e = [0] * p.shape[0]
    c = len(p[0, :])
    for i in range(p.shape[0]):
        e[i] = - np.sum(p[i, :] * np.log2(p[i, :])) / np.log2(c)
    return e


def calc_density(s):
    h = 5
    d = [0] * s.shape[0]
    for i in range(s.shape[0]):
        d[i] = np.sum(s[i, :][s[i, :].argsort()[h * (-1):]]) / h
    return d


def calc_low_density(d):
    h = 5
    l = [0] * d.shape[0]
    for i in range(d.shape[0]):
        l[i] = np.sum(d[i, :][d[i, :].argsort()[h * (-1):]]) / h
    return l


def c3e_sl(piSet, SSet, I, alpha):
    N = len(piSet)
    c = len(piSet[0, :])
    # piSet = np.array(piSet)
    y = [[1] * c] * N
    y = np.divide(y, c)
    labels = [-1] * N
    # y = pd.DataFrame(y)
    for k in range(0, I):
        for j in range(0, N):
            diffi = np.arange(0, N)
            cond = diffi != j
            t1 = np.array(SSet[j][cond])
            # http://mathesaurus.sourceforge.net/matlab-numpy.html
            # https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
            p1 = (np.transpose(t1 * np.ones([c, 1])) * y[cond, :]).sum(axis=0)
            p2 = sum(t1)
            y[j, :] = (piSet[j, :] + (2 * alpha * p1)) / (1 + 2 * alpha * p2)
            labels[j] = int(np.where(y[j, :] == np.max(y[j, :]))[0])
    return y, labels


def eds(train, test, y, SSet, DistMat):
    ### entropy measuse
    e = calc_class_entropy(y)
    candidates = e > np.percentile(e, 75)
    values = np.array(e)[candidates]

    #### density measure - não funciona bem!
    d = calc_density(SSet)
    candidates = d > np.percentile(d, 75)
    values = np.array(d)[candidates]

    ### low density measure
    l = calc_low_density(DistMat)
    candidates = l > np.percentile(l, 75)
    values = np.array(l)[candidates]

    #### silhouette measure
    from sklearn.metrics import silhouette_samples
    sil_test = np.concatenate([train, test])
    clabels = classAnnotation(sil_test)
    sil_values = silhouette_samples(sil_test, clabels[0])
    s = sil_values[len(test) * (-1):]
    candidates = s > np.percentile(s, 25)
    values = np.array(s)[candidates]

    ### ensembles
    el = np.multiply(e, l)
    candidates = el > np.percentile(el, 75)
    values = np.array(el)[candidates]

    sc = 1 - s
    esc = np.multiply(e, sc)
    candidates = esc > np.percentile(esc, 75)
    values = np.array(esc)[candidates]

    return [candidates, values]


def ic(probs, SSet, train, train_labels, test, test_labels):
    y = c3e_sl(probs, SSet, 5, 0.001)
    for k in range(10):
        e = calc_class_entropy(y)
        d = calc_density(SSet)
        w = eds(e, d, 5, SSet)
        [train, train_labels, test, test_labels] = increment_training_set(w, train, train_labels, test, test_labels)
        probs = svmClassification(train, train_labels, test, test_labels)
        SSet = reduce_matrix(w, SSet)
        y = c3e_sl(probs, SSet, 5, 0.001)
        print("Iteration " + str(k + 1) + " - Sizes: Training Set " + str(len(train)) + " - Test Set " + str(len(test)))

# TO TEST AFTER...
# https://scikit-learn.org/stable/modules/outlier_detection.html#outlier-detection
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html#sklearn.svm.OneClassSVM.score_samples
# https://scikit-learn.org/stable/auto_examples/svm/plot_oneclass.html#sphx-glr-auto-examples-svm-plot-oneclass-py
# https://scikit-learn.org/stable/auto_examples/linear_model/plot_sgdocsvm_vs_ocsvm.html#sphx-glr-auto-examples-linear-model-plot-sgdocsvm-vs-ocsvm-py
# https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_anomaly_comparison.html#sphx-glr-auto-examples-miscellaneous-plot-anomaly-comparison-py

def self_training(train, train_labels, test, test_labels):
    # https://scikit-learn.org/stable/modules/svm.html
    [probs, preds] = svmClassification(train, train_labels, test)
    # treinar um classificador com o train, train_labels
    # e testar a classificação dele no 'test'
    # vamos usar o probs para saber quais objetos nós vamos levar para training set
    # saida é: probs e preds

    for k in range(1, 11):

        e = calc_class_entropy(probs)
        # com o e vc precisa descobrir quais sao os índices de menor valor de entropia
        # talvez tenha que ordenar (sort()) de forma crescente e pegar os 2 primeiros,
        # mas deve-se saber qual é o índice!
        # o que queremos fazer é descobrir dos 8 classificados quais são os 2
        # com rótulos de classe mais confiáveis (de estarem certos) para incorporarmos
        # eles no training set (usaremos o probs pra isso).
        # saida: índices dos 2 mais confiáveis para ser um vetor w

        df_e = pd.DataFrame(e)
        df_e.sort_values(by=[0], inplace=True, ascending=False)

        # funcao q a partir de e retorna um w que sao os indices dos c mais confiaveis
        posicoes = df_e.index.values
        posicoes = posicoes.tolist()
        p = 96

        w = posicoes[
            0:p]  # posicoes[0:p] # índices (posição) dos objetos que serão retirados do conjunto de teste e colocados no conjunto de treino

        [train, train_labels, test, test_labels] = increment_training_set(w, train, train_labels, test, test_labels)

        # https://scikit-learn.org/stable/modules/svm.html
        if (len(test) > 0):
            [probs, preds] = svmClassification(train, train_labels, test)

        print("Iteration " + str(k) + " - Sizes: Training Set " + str(len(train)) + " - Test Set " + str(len(test)))

        print(
            pd.crosstab(pd.Series(test_labels.ravel(), name='Real'), pd.Series(preds, name='Predicted'), margins=True))
        classes = ['wilt', 'rest']
        print(metrics.classification_report(test_labels, preds, target_names=classes))


##### INICIO DO ALGORITMOS #####

### https://archive.ics.uci.edu/ml/datasets/wilt
train_data_path = 'https://raw.githubusercontent.com/luizfsc/datasets/master/Oak-Pine-Wilt/training.csv'
test_data_path = 'https://raw.githubusercontent.com/luizfsc/datasets/master/Oak-Pine-Wilt/testing.csv'
# train_data_path = 'https://raw.githubusercontent.com/luizfsc/datasets/master/mastite.csv'
# test_data_path = ''

train_data, train_labels, _, _ = get_batch_data(train_data_path, test_data_path, 5, True, 967, 1)
# train_data, train_labels, _, _ = get_batch_data(train_data_path, test_data_path, 14, False, 8, 1)

for i in range(2, 4):  # range(2,5):
    print("\nTRAINING SET - part 1 - TEST SET - part " + str(i))
    test_data, test_labels, _, _ = get_batch_data(train_data_path, test_data_path, 5, True, 967, i)
    # test_data, test_labels, _, _ = get_batch_data(train_data_path, test_data_path, 14, False, 8, i)

    # print(np.array(test_data))
    # print(np.array(test_labels))

    self_training(train_data, train_labels, test_data, test_labels)