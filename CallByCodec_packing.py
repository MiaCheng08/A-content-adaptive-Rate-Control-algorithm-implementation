# -*- coding:utf-8 -*-
"""

"""

from tensorflow import keras
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing
from tensorflow.keras import backend as K
import csv
from math import isnan
import math
import warnings

def notZero(x):
    return x != 0


def parseLookahead(path):
    round = 1
    avgScore = {}
    len = 0
    csvFile = open(path, 'r', encoding='UTF-8')
    reader = csv.reader(csvFile, delimiter=',')
    fieldnames = next(reader)
    reader = csv.DictReader(csvFile, fieldnames=fieldnames, delimiter=',')
    for row in reader:
        ****** #confidential
    csvFile.close()
    return avgScore


def getCRF(pred_param):
    featureNum = len(pred_param['curr_feature'])
    # Input data
    csvFile = pd.read_csv(pred_param['existedfeaturePath'], header=None)
    existed = np.array(csvFile, dtype=float)
    curr_feature = np.array(pred_param['curr_feature']).reshape(1,featureNum)
    features = np.concatenate((existed, curr_feature), axis=0)
    # process data
    for i in range(featureNum):
        features[:, i] = preprocessing.scale(features[:, i])

    curr_feature = features[-1]
    curr_feature = np.expand_dims(curr_feature, axis=0)#这个必须加上
    # model3
    model = keras.models.load_model(pred_param['modelPath'], custom_objects={'curve_simility_cby_lossV1': curve_simility_cby_lossV1})
    params = model.predict(curr_feature).flatten()#, batch_size=1
    crf = np.polyval(params.tolist(), math.log(float(pred_param['targetR']) / 1000))
    return crf


def curve_simility_cby_lossV1(y_true, y_pred):  # y_true, y_predz
    R = np.arange(0.2, 12, 0.5, dtype=np.float32)
    len = R.size
    lnR = [math.log(x) for x in R]
    lnR = tf.reshape(tf.constant(lnR), [1, len])
    lnR0 = tf.pow(lnR, 0)
    lnR2 = tf.pow(lnR, 2)
    crf = tf.concat([lnR2, lnR, lnR0], 0)
    b_true = tf.matmul(y_true, crf)
    b_pred = tf.matmul(y_pred, crf)

    # lossV1
    loss = tf.abs(tf.subtract(b_true, b_pred))
    return K.mean(loss)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    pred_param = {}
    record_features = {}
    featureList = ******#confidential

    pred_param['modelPath'] = "model.h5"
    pred_param['existedfeaturePath'] = "data.txt"
    optionPath = "CliOption.txt"
    optionWriter = open(optionPath)
    string = optionWriter.read().split("--")
    coding_option = {}

    for option in string:
        if option:
            key, value = option.split(" ",1)
            coding_option[key] = value
    featurePath = coding_option['feature'] 
    feature_Lookahead = parseLookahead(featurePath)

    pred_param['videopath'] = coding_option['input']

    pred_param['targetR'] = coding_option['targetR']

    record_features['fps'] = coding_option['fps']

    record_features['weight'],record_features['height'] = coding_option['input-res'].split('x')

    for item in feature_Lookahead[1]:
        record_features.setdefault(item, float(feature_Lookahead[1][item]))

    pred_param['curr_feature'] = []
    for item in featureList:
        pred_param['curr_feature'].append(float(record_features[item]))

    crf = getCRF(pred_param)
    print(crf)
