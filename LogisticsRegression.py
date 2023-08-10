import math
import random

irisFile = open("bezdekIris.data", "r")
dataset = []
dataLines = irisFile.readlines()
for line in dataLines:
    splits = line.strip().split(",")
    sepalLength = (float)(splits[0])
    sepalWidth = (float)(splits[1])
    petalLength = (float)(splits[2])
    petalWidth = (float)(splits[3])
    classAttribute = (str)(splits[4])
    dataset.append((sepalLength, sepalWidth, petalLength, petalWidth, classAttribute))

irisFile.close()

def compute_f(w1, w2, w3, w4, b, x1, x2, x3, x4):
    return w1 * x1 + w2 * x2 + w3 * x3 + w4 * x4 + b

def compute_dfk_dwij(i, j, k, x):
    if (i == k):
        return x[j - 1]
    else:
        return 0
        
def compute_dfk_dbi(i, k):
    if (i == k):
        return 1
    else:
        return 0

def predictLabel(P0, P1, P2):
    if (P0 > P1 and P0 > P2):
        return 0
    elif (P1 > P2):
        return 1
    else:
        return 2

def computeProbabilities(w11, w12, w13, w14, b1, w21, w22, w23, w24, b2, x1, x2, x3, x4):
    P0 = 1 / (1 + math.exp(-1 * compute_f(w11, w12, w13, w14, b1, x1, x2, x3, x4)))
    P1 = (1 - P0) / (1 + math.exp(-1 * compute_f(w21, w22, w23, w24, b2, x1, x2, x3, x4)))
    P2 = 1 - P0 - P1
    return P0, P1, P2
    
def labelToNumber(label):
    if (label == "Iris-setosa"):
        return 0
    elif(label == "Iris-versicolor"):
        return 1
    elif(label == "Iris-virginica"):
        return 2
    else:
        print("unknown label")
        
def numberToLabel(number):
    if (number == 0):
        return "Iris-setosa"
    elif (number == 1):
        return "Iris-versicolor"
    elif (number == 2):
        return "Iris-virginica"
    else:
        print("unknown label")

def computeLogLikelihood(dataset, w11, w12, w13, w14, b1, w21, w22, w23, w24, b2):
    L = 0
    for example in dataset:
        P0, P1, P2 = computeProbabilities(w11, w12, w13, w14, b1, w21, w22, w23, w24, b2, example[0], example[1], example[2], example[3])
        label = labelToNumber(example[4])
        L += 1/2 * (1 - label) * (2 - label) * math.log(P0) + label * (2 - label) * math.log(P1) - 1/2 * label * (1 - label) * math.log(P2)
    return L

def compute_dP0_dwij(P0, i, j, w11, w12, w13, w14, b1, x1, x2, x3, x4):
    return P0 * P0 * math.exp(-1 * compute_f(w11, w12, w13, w14, b1, x1, x2, x3, x4)) * compute_dfk_dwij(i, j, 1, [x1, x2, x3, x4])
        
def compute_dP1_dwij(P0, P1, dP0_dwij, i, j, w21, w22, w23, w24, b2, x1, x2, x3, x4):
    return (-dP0_dwij / (1 - P0)) + compute_dfk_dwij(i, j, 2, [x1, x2, x3, x4]) / (1 + math.exp(compute_f(w21, w22, w23, w24, b2, x1, x2, x3, x4)))

# derivative of logLikelihood (L) with respect to the matrix w of the ith vector and jth dimension
def compute_dL_dwij(dataset, i, j, w11, w12, w13, w14, b1, w21, w22, w23, w24, b2):
    dL_dwij = 0
    for example in dataset:
        P0, P1, P2 = computeProbabilities(w11, w12, w13, w14, b1, w21, w22, w23, w24, b2, example[0], example[1], example[2], example[3])
        label = labelToNumber(example[4])
        dP0_dwij = compute_dP0_dwij(P0, i, j, w11, w12, w13, w14, b1, example[0], example[1], example[2], example[3])
        dP1_dwij = compute_dP1_dwij(P0, P1, dP0_dwij, i, j, w21, w22, w23, w24, b2, example[0], example[1], example[2], example[3])
        dP2_dwij = -dP0_dwij - dP1_dwij
        dL_dwij += 1/2 * (1 - label) * (2 - label) * (dP0_dwij / P0) + label * (2 - label) * (dP1_dwij / P1) - 1/2 * label * (1 - label) * (dP2_dwij / P2)
    return dL_dwij

def compute_dP0_dbi(P0, i, w11, w12, w13, w14, b1, x1, x2, x3, x4):
    return P0 * P0 * math.exp(-1 * compute_f(w11, w12, w13, w14, b1, x1, x2, x3, x4)) * compute_dfk_dbi(i, 1)
        
def compute_dP1_dbi(P0, P1, dP0_dbi, i, w21, w22, w23, w24, b2, x1, x2, x3, x4):
    return (-dP0_dbi / (1 - P0)) + compute_dfk_dbi(i, 2) / (1 + math.exp(compute_f(w21, w22, w23, w24, b2, x1, x2, x3, x4)))
    
def compute_dL_dbi(dataset, i, w11, w12, w13, w14, b1, w21, w22, w23, w24, b2):
    dL_dbi = 0
    for example in dataset:
        P0, P1, P2 = computeProbabilities(w11, w12, w13, w14, b1, w21, w22, w23, w24, b2, example[0], example[1], example[2], example[3])
        label = labelToNumber(example[4])
        dP0_dbi = compute_dP0_dbi(P0, i, w11, w12, w13, w14, b1, example[0], example[1], example[2], example[3])
        dP1_dbi = compute_dP1_dbi(P0, P1, dP0_dbi, i, w21, w22, w23, w24, b2, example[0], example[1], example[2], example[3])
        dP2_dbi = -dP0_dbi - dP1_dbi
        dL_dbi += 1/2 * (1 - label) * (2 - label) * (dP0_dbi / P0) + label * (2 - label) * (dP1_dbi / P1) - 1/2 * label * (1 - label) * (dP2_dbi / P2)
    return dL_dbi
    
def learnModel(dataset, learningRate, w11, w12, w13, w14, b1, w21, w22, w23, w24, b2):
    dL_dw11 = compute_dL_dwij(dataset, 1, 1, w11, w12, w13, w14, b1, w21, w22, w23, w24, b2) * learningRate
    dL_dw12 = compute_dL_dwij(dataset, 1, 2, w11, w12, w13, w14, b1, w21, w22, w23, w24, b2) * learningRate
    dL_dw13 = compute_dL_dwij(dataset, 1, 3, w11, w12, w13, w14, b1, w21, w22, w23, w24, b2) * learningRate
    dL_dw14 = compute_dL_dwij(dataset, 1, 4, w11, w12, w13, w14, b1, w21, w22, w23, w24, b2) * learningRate
    dL_db1 = compute_dL_dbi(dataset, 1, w11, w12, w13, w14, b1, w21, w22, w23, w24, b2) * learningRate
    
    dL_dw21 = compute_dL_dwij(dataset, 2, 1, w11, w12, w13, w14, b1, w21, w22, w23, w24, b2) * learningRate
    dL_dw22 = compute_dL_dwij(dataset, 2, 2, w11, w12, w13, w14, b1, w21, w22, w23, w24, b2) * learningRate
    dL_dw23 = compute_dL_dwij(dataset, 2, 3, w11, w12, w13, w14, b1, w21, w22, w23, w24, b2) * learningRate
    dL_dw24 = compute_dL_dwij(dataset, 2, 4, w11, w12, w13, w14, b1, w21, w22, w23, w24, b2) * learningRate
    dL_db2 = compute_dL_dbi(dataset, 2, w11, w12, w13, w14, b1, w21, w22, w23, w24, b2) * learningRate
    
    return w11 + dL_dw11, w12 + dL_dw12, w13 + dL_dw13, w14 + dL_dw14, b1 + dL_db1, w21 + dL_dw21, w22 + dL_dw22, w23 + dL_dw23, w24 + dL_dw24, b2 + dL_db2

def reportEpoch(currentEpoch, dataset, w11, w12, w13, w14, b1, w21, w22, w23, w24, b2):
    print("Epoch", currentEpoch, " LogLikelihood:", computeLogLikelihood(dataset, w11, w12, w13, w14, b1, w21, w22, w23, w24, b2))

def trainModel(dataset, numEpochs, learningRate, w11, w12, w13, w14, b1, w21, w22, w23, w24, b2):
    for i in range(numEpochs):
        if (i % 1000 == 0):
            reportEpoch(i, dataset, w11, w12, w13, w14, b1, w21, w22, w23, w24, b2)
        w11, w12, w13, w14, b1, w21, w22, w23, w24, b2 = learnModel(dataset, learningRate, w11, w12, w13, w14, b1, w21, w22, w23, w24, b2)
       
    reportEpoch(numEpochs, dataset, w11, w12, w13, w14, b1, w21, w22, w23, w24, b2)
    return w11, w12, w13, w14, b1, w21, w22, w23, w24, b2

def computeAccuracy(dataset, w11, w12, w13, w14, b1, w21, w22, w23, w24, b2):
    correctPredictions = 0
    totalLabels = len(dataset)
    for example in dataset:
        P0, P1, P2 = computeProbabilities(w11, w12, w13, w14, b1, w21, w22, w23, w24, b2, example[0], example[1], example[2], example[3])
        predictedLabel = predictLabel(P0, P1, P2)
        actualLabel = labelToNumber(example[4])
        if (predictedLabel == actualLabel):
            correctPredictions += 1
    return correctPredictions / totalLabels

trainingSet = []
testSet = []
random.seed(100)
for i in range(len(dataset) // 5):
    j = random.randrange(len(dataset))
    testSet.append(dataset[j])
    dataset.pop(j)
    
for example in dataset:
    trainingSet.append(example)

w11, w12, w13, w14, b1, w21, w22, w23, w24, b2 = trainModel(trainingSet, 8000, 0.0001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
#P0, P1, P2 = computeProbabilities(w11, w12, w13, w14, b1, w21, w22, w23, w24, b2, 5.1,3.5,1.4,0.2) # Iris-setosa
#print(P0, P1, P2)
#print(numberToLabel(predictLabel(P0, P1, P2)))
print("training set accuracy:", computeAccuracy(trainingSet, w11, w12, w13, w14, b1, w21, w22, w23, w24, b2))
print("test set accuracy:", computeAccuracy(testSet, w11, w12, w13, w14, b1, w21, w22, w23, w24, b2))