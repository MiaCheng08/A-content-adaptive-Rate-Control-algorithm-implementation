# -*- coding:utf-8 -*-
"""
针对使用阶段的库函数
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
    cost = []
    ssdY = []
    ssdU = []
    ssdV = []
    sumY = []
    sumU = []
    sumV = []
    AC = []
    MVX = []
    MVY = []
    MV = []
    words = []
    IMB = []

    avgScore = {}
    len = 0
    csvFile = open(path, 'r', encoding='UTF-8')
    reader = csv.reader(csvFile, delimiter=',')
    fieldnames = next(reader)
    reader = csv.DictReader(csvFile, fieldnames=fieldnames, delimiter=',')
    for row in reader:
        if row['Encode Order'] != "one round LOOKAHEAD is over":
            #解析look ahead信息
            avgScore.setdefault(round, {})  # round用来标记当前字典记录的是第几轮look ahead的pre结果
            len += 1
            avgScore[round]['end F'] = row['POC']
            avgScore[round].setdefault('block8Num', row['block8Num'])
            # avgScore.setdefault('fps', row['fps'])
            # avgScore.setdefault('block8Num', row['block8Num'])
            # avgScore.setdefault('block8Num', row['block8Num'])
            cost.append(float(row['EstCost']))
            ssdY.append(float(row['wp_ssdY']))
            ssdU.append(float(row['wp_ssdU']))
            ssdV.append(float(row['wp_ssdV']))
            sumY.append(float(row['wp_sumY']))
            sumU.append(float(row['wp_sumU']))
            sumV.append(float(row['wp_sumV']))
            AC.append(float(row['AC']))
            if row['MVX'] != '-':
                MVX.append(float(row['MVX']))
                MVY.append(float(row['MVY']))
                MV.append(float(row['MV']))
                words.append(float(row['words']))
            if row['IMB'] != '-':
                IMB.append(float(row['IMB']))
        else:
            #一轮look ahead解析结束
            avgScore[round]['Look Len'] = len
            avgScore[round]['start F'] = int(avgScore[round]['end F']) + 1 - len
            cost = list(filter(notZero, cost))
            avgScore[round]['EstCost'] = np.mean(cost)
            ssdY = list(filter(notZero, ssdY))
            avgScore[round]['ssdY'] = np.mean(ssdY)
            ssdU = list(filter(notZero, ssdU))
            avgScore[round]['ssdU'] = np.mean(ssdU)
            ssdV = list(filter(notZero, ssdV))
            avgScore[round]['ssdV'] = np.mean(ssdV)
            sumY = list(filter(notZero, sumY))
            avgScore[round]['sumY'] = np.mean(sumY)
            sumU = list(filter(notZero, sumU))
            avgScore[round]['sumU'] = np.mean(sumU)
            sumV = list(filter(notZero, sumV))
            avgScore[round]['sumV'] = np.mean(sumV)
            AC = list(filter(notZero, AC))
            avgScore[round]['AC'] = np.mean(AC)
            avgScore[round]['MVX'] = 0 if isnan(np.mean(MVX)) else np.mean(MVX)
            avgScore[round]['MVY'] = 0 if isnan(np.mean(MVY)) else np.mean(MVY)
            avgScore[round]['MV'] = 0 if isnan(np.mean(MV)) else np.mean(MV)
            avgScore[round]['words'] = 0 if isnan(np.mean(words)) else np.mean(words)
            avgScore[round]['IMB'] = 0 if isnan(np.mean(IMB)) else np.mean(IMB)
            round += 1
            len = 0
            cost = []
            ssdY = []
            ssdU = []
            ssdV = []
            sumY = []
            sumU = []
            sumV = []
            AC = []
            MVX = []
            MVY = []
            IMB = []
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
    featureList = ['fps',  'EstCost', 'ssdY', 'ssdU', 'ssdV', 'sumY', 'sumU',
                   'sumV', 'AC', 'IMB', 'MV', 'words']
    # featureList = ['weight', 'height','origBitrate','fps',  'EstCost', 'ssdY', 'ssdU', 'ssdV', 'sumY', 'sumU',
    #                'sumV', 'AC', 'IMB', 'MVX', 'MVY']

    pred_param['modelPath'] = "model.h5"
    pred_param['existedfeaturePath'] = "data.txt"
    optionPath = "CliOption.txt"
    optionWriter = open(optionPath)
    string = optionWriter.read().split("--")
    coding_option = {}
    # 解析编码命令行参数
    for option in string:
        if option:
            key, value = option.split(" ",1)
            coding_option[key] = value
    featurePath = coding_option['feature']  # 编码器输出look ahead特征文件所在路径
    feature_Lookahead = parseLookahead(featurePath)#获取look ahead特征, 包括当前特征cover的起始和终点视频帧

    pred_param['videopath'] = coding_option['input']#原始视频所在路径，获取他用来做深入的特征提取

    pred_param['targetR'] = coding_option['targetR']#2.3 Mbps

    record_features['fps'] = coding_option['fps']

    record_features['weight'],record_features['height'] = coding_option['input-res'].split('x')

    for item in feature_Lookahead[1]:
        record_features.setdefault(item, float(feature_Lookahead[1][item]))

    record_features['EstCost'] = record_features['EstCost'] / record_features['block8Num']
    pred_param['curr_feature'] = []
    for item in featureList:
        pred_param['curr_feature'].append(float(record_features[item]))

    crf = getCRF(pred_param)
    print(crf)
