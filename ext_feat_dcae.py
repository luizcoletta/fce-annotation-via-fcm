
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    BatchNormalization,
    Flatten,
    Reshape,
    GaussianNoise,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import numpy as np
from math import sqrt

#- X_train: dados entrada de treinamento
#- y_train = rótulos de treinamento
#- filters = número de filtros nas camadas convolucionais
#- filter_sizes = tamanhos dos filtros convolucionais
#- sampling = Fator de redução na subamostragem
#- dp = nível de dropout (%/100)
#- epochs = número de épocas
#- batch_size = tamanho de lote
#- lr = taxa de aprendizado
#- momentum = momento do SGD
#- stddev = desvio padrão do ruído da entrada
#- pre_training = habilita o pré-treinamento
#- summary = habilita a visualização da arquitetura
#- save = permite salvar modelos e históricos
#*** Deve-se usar entradas (X_train) de tamanho "N",
#tal que "N" seja o quadrado de um número inteiro e
#potência do fator de subamostragem (sampling), para
#que seja possível redimensioná-las e reconstruí-las
#adequadamente

# Função do DCAE (DENOISING CONVOLUTIONAL AUTOENCODER)
def dcae(X_train,
        y_train,
        filters = (32, 64),
        filter_sizes = (12, 6),
        sampling = 2,
        dp = 0.2,
        epochs = (30, 60),
        batch_size = 32,
        lr = 0.001,
        momentum = 0.9,
        stddev = 0.2,
        pre_training = True,
        summary = True,
        save = False ):

    # Configurações gerais
    act_hidden = 'tanh'
    act_classifier = 'softmax'
    padding = 'same'
    loss_pt = 'mse'
    loss_ft = 'categorical_crossentropy'
    optimizer = SGD (lr = lr, momentum = momentum)
    val_split = 0.1

    # Opção para salvar o melhor modelo
    cp_pt = ModelCheckpoint('trained_models / dcae_pt . h5',
    save_best_only = True,
    monitor = 'val_loss',
    mode = 'max')

    # Opção para salvar o melhor modelo
    cp_ft = ModelCheckpoint('trained_models / dcae_ft . h5',
    save_best_only = True,
    monitor = 'val_accuracy',
    mode = 'max')

    # Parada antecipada
    es = EarlyStopping(monitor = 'val_loss', patience = 2)

    # Arquitetura do DCAE
    my_input = Input(shape = (X_train.shape[1],))
    reshaped_in = Reshape((int(sqrt(X_train.shape[1])),
                           int(sqrt(X_train.shape[1])), 1))(my_input)
    noise = GaussianNoise(stddev)(reshaped_in)
    conv_1 = Conv2D(filters[0], filter_sizes[0], activation = act_hidden,
                    padding = padding)(noise)
    bn_1 = BatchNormalization()(conv_1)
    down_1 = MaxPooling2D((sampling, sampling), padding = padding)(bn_1)
    dp_1 = Dropout(dp)(down_1)
    conv_2 = Conv2D(filters[1], filter_sizes[1], activation = act_hidden,
                    padding = padding)(dp_1)
    bn_2 = BatchNormalization()(conv_2)
    down_2 = MaxPooling2D((sampling, sampling), padding = padding)(bn_2)
    dp_2 = Dropout(dp)(down_2)
    up_1 = UpSampling2D((sampling, sampling))(dp_2)
    conv_3 = Conv2D(filters[0], filter_sizes[1], activation = act_hidden,
                    padding = padding)(up_1)
    up_2 = UpSampling2D((sampling, sampling))(conv_3)
    conv_4 = Conv2D (1, filter_sizes[0], padding = padding)(up_2)
    reshaped_out = Flatten()(conv_4)

    # Configuração e treinamento do DCAE
    ae = Model(my_input, reshaped_out)
    ae.compile(loss = loss_pt, optimizer = optimizer)
    if pre_training:
        if summary:
            ae.summary()
        dcae_hist_pt = ae.fit(X_train, X_train, epochs = epochs[0],
                              batch_size = batch_size, verbose = summary,
                              validation_split = val_split, shuffle = True,
                              callbacks = [es, cp_pt])

        # Salvando os históricos de pré-treinamento
        if save:
            np.savetxt ('hists/dcae_pt_loss_train_.txt',
                dcae_hist_pt. history ['loss'])
            np.savetxt ('hists/dcae_pt_loss_val.txt',
                dcae_hist_pt . history ['val_loss'])

        # Achatamento e adição do classificador
        fl = Flatten()(dp_2)
        my_output = Dense(y_train.shape[1], activation = act_classifier)(fl)

        # Configuração e treinamento do classificador
        dcae = Model(my_input, my_output)
        dcae.compile(loss = loss_ft, optimizer = optimizer, metrics =['accuracy'])
        if summary:
            dcae.summary()
        dcae_hist_ft = dcae.fit(X_train, y_train, epochs = epochs[1],
                                batch_size = batch_size, verbose = summary,
                                validation_split = val_split, shuffle = True,
                                callbacks = [cp_ft])

    # Salvando os históricos de refinamento
    if save:
        np.savetxt('hists / dcae_ft_acc_train . txt', dcae_hist_ft.history['accuracy'])
        np.savetxt('hists / dcae_ft_acc_val . txt', dcae_hist_ft.history['val_accuracy'])
        np.savetxt('hists / dcae_ft_loss_train . txt', dcae_hist_ft.history['loss'])
        np.savetxt('hists / dcae_ft_loss_val . txt', dcae_hist_ft.history['val_loss'])

    return dcae
    

# Carregando os conjuntos de treinamento
X_train_raw = np.loadtxt('data/X_train_raw.txt')
X_train_zscore = np.loadtxt('data/X_train_zscore.txt')
y_train = np . loadtxt('data/y_train.txt')

# Opções
summary = True
save = True
n_samples = 190400

# Treinando o DCAE
print('\n************ DCAE ************\n')
model = dcae(X_train_zscore[:n_samples],
        y_train[:n_samples],
        epochs = (20, 60),
        summary = summary,
        save = save,
        pre_training = True)




