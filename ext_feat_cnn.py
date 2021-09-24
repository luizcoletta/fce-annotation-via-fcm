### https://gogul.dev/software/flower-recognition-deep-learning
### https://github.com/efidalgo/AutoBlur-CNN-Features

# filter warnings
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# keras imports
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

# other imports
from sklearn.manifold import Isomap
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import datetime
import time
import pandas as pd
#import json
#import h5py

import csv

from utils import existing_directory

def dimensionality_reduction(df_selData, n_comp):
    print("[INFO] feature reduction of " + str(len(df_selData[0])) + " for " + str(n_comp))
    isomap = Isomap(n_components=n_comp)
    selData = isomap.fit_transform(pd.DataFrame(df_selData))
    return selData


datasets_available = ["res10"]  # ["Soccer"]

# datasets to be analyzed
dataset = datasets_available
# test_set = [0.25, 0.25, 0.25]

for ds in range(0, len(dataset)):
    # Select the configuration file available
    #file = 'conf_' + dataset[ds] + '.json'
    
    # load the user configs. It has been created a json file per dataset
    #with open(file) as f:    
    #  config = json.load(f)
      
    # config variables
    model_name = "vgg16"
    weights = "imagenet"
    include_top = 0
    images_path = "results/" + dataset[ds] + "/images"
    results = "output/" + dataset[ds] + "/" + model_name + "/results" + dataset[ds] + ".txt"
    # features_path = "output/" + dataset[ds] + "/" + model_name + "/features.h5"
    # labels_path   = "output/" + dataset[ds] + "/" + model_name + "/labels.h5"
    # model_path    = "output/" + dataset[ds] + "/" + model_name + "/model"
    # test_size     = test_set[ds]
    
    # check if the output directory exists, if not, create it.
    existing_directory("results")
    existing_directory("results/deep_features")
    existing_directory("results/deep_features/" + dataset[ds])
    
    # start time
    print("[STATUS] start time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
    start = time.time()
    
    # create the pretrained models
    # check for pretrained weight usage or not
    # check for top layers to be included or not
    if model_name == "vgg16":
        base_model = VGG16(weights=weights)
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
        image_size = (224, 224)
    elif model_name == "vgg19":
        base_model = VGG19(weights=weights)
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
        image_size = (224, 224)
    elif model_name == "resnet50":
        base_model = ResNet50(weights=weights)
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('flatten').output)
        image_size = (224, 224)
    elif model_name == "inceptionv3":
        base_model = InceptionV3(include_top=include_top, weights=weights, input_tensor=Input(shape=(299, 299, 3)))
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('flatten').output)
        image_size = (299, 299)
    elif model_name == "inceptionresnetv2":
        base_model = InceptionResNetV2(include_top=include_top, weights=weights, input_tensor=Input(shape=(299, 299, 3)))
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('conv_7b').output)
        image_size = (299, 299)
    elif model_name == "mobilenet":
        base_model = MobileNet(include_top=include_top, weights=weights, input_tensor=Input(shape=(224, 224, 3)), input_shape=(224, 224, 3))
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('conv1_relu').output)
        image_size = (224, 224)
    elif model_name == "xception":
        base_model = Xception(weights=weights)
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
        image_size = (299, 299)
    else:
        base_model = None
    
    # check if the output directory exists, if not, create it.
    existing_directory("results/deep_features/" + dataset[ds] + "/" + model_name)
    
    print("[INFO] successfully loaded base model and model...")
    
    # path to training dataset
    files = os.listdir(images_path)
    files = sorted(files)
    
    # encode the labels
    # print("[INFO] encoding labels...")
    ### AQUI TEM QUE ALTERAR QDO. USAR NOSSA BASE DE DADOS, PARECE QUE PEGA OS LABELS PELOS NOMES DOS ARQUIVOS
    # le = LabelEncoder()
    # le.fit([tl for tl in train_labels])
    
    # variables to hold features and labels
    features = []
    labels = []

    # loop over all the labels in the folder
    for i, img_name in enumerate(files):
        label = img_name[-5:-4]
        # for image_path in glob.glob(cur_path + "/*.jpg"):
        #for image_path in range(0, len(list_files)):
        # print ("[INFO] Processing - " + str(count) + " named " + list_files[image_path])
        img = image.load_img(images_path + "/" + img_name, target_size=image_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feature = model.predict(x)
        flat = feature.flatten()
        features.append(flat)
        labels.append(int(label)+1)
        print("[INFO] processed for image " + str(i+1) + ": " + img_name)
    # print ("[INFO] completed label - " + label)
    
    # encode the labels using LabelEncoder
    # le = LabelEncoder()
    # le_labels = le.fit_transform(labels)
    
    # get the shape of training labels
    # print("[STATUS] training labels: {}".format(le_labels))
    # print("[STATUS] training labels shape: {}".format(le_labels.shape))
    
    # save features and labels
    '''try:
        h5f_data = h5py.File(features_path, 'w')
    except:
        a=1;'''

    features = dimensionality_reduction(features, 8)

    ### Aqui tinha que salver em CSV mesmo...
    # Concact for features w/ labels
    # data = np.c_[features, le_labels]
    data = np.c_[features, labels]
    # Save all data in .csv file
    np.savetxt("results/deep_features/" + dataset[ds] + "/" + model_name + "/data.csv", [p for p in data], delimiter=',', fmt='%f')

    '''h5f_data.create_dataset('dataset_1', data=np.array(features))
    h5f_label = h5py.File(labels_path, 'w')
    h5f_label.create_dataset('dataset_1', data=np.array(le_labels))
    h5f_data.close()
    h5f_label.close()'''
    
    # save model and weights
    ''' model_json = model.to_json()
    with open(model_path + str(test_size) + ".json", "w") as json_file:
      json_file.write(model_json)
    
    # save weights
    model.save_weights(model_path + str(test_size) + ".h5")
    print("[STATUS] saved model and weights to disk..")
    
    print ("[STATUS] features and labels saved..")'''
    
    # end time
    end = time.time()
    print("[STATUS] end time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
