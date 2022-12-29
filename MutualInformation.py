import numpy as np


def entropy(probX):

    ent = 0
    for i in range(len(probX)):
        if probX[i] > 0.0:
            ent += -probX[i]*np.log2(probX[i])

    return ent


def joint_entropy(probXY):

    ent = 0
    row, column = probXY.shape
    for i in range(row):
        for j in range(column):
            if probXY[i,j] > 0.0:
                ent += -probXY[i, j]*np.log2(probXY[i, j])

    return ent


def cond_entropy(probX, probXY):

    HX = entropy(probX)
    HXY = joint_entropy(probXY)
    HYcondX = HXY - HX

    return HYcondX


def mutual_information(probX, probY, probXY):

    HX = entropy(probX)
    HY = entropy(probY)
    HXY = joint_entropy(probXY)
    MI = HX + HY - HXY

    return MI


def extractProbabilities(probXY):

    row, column = probXY.shape
    probX = np.zeros([row])
    probY = np.zeros([row])
    probXcondY = 0
    probYcondX = 0

    for i in range(row):
        if np.sum(probXY[i, :]) > 0:
            probXcondY = probXY[i, i]/np.sum(probXY[i, :])

        if np.sum(probXY[:, i]) > 0:
            probYcondX = probXY[i, i]/np.sum(probXY[:, i])

        if probXcondY > 0.0:
            probY[i] = probXY[i, i]/probXcondY

        if probYcondX > 0.0:
            probX[i] = probXY[i, i]/probYcondX

    return probX, probY

