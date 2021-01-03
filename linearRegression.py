import numpy as np

data_size = 10000
number_of_features = 1
degree_of_hypothesis = 3
min_x = 0
max_x = 10
gaussian_noise_mean = 0
gaussian_noise_sd = 1
data = []


#generate data based on a function that follows a 3x^3 - 2x^2 + 3 with a gaussian noise
#x is uniformly distributed across min_x, max_x
def initData(data_size, min_x,max_x,gaussian_noise_mean,gaussian_noise_sd):
    data = []
    for i in range(data_size):
        x = np.random.uniform(min_x,max_x)
        gaussian_noise = np.random.normal(gaussian_noise_mean,gaussian_noise_sd)
        y = 5*x + 7 + gaussian_noise

        data.append((x,y))
    return data

#assuming data is 1-dimensional for now
def getPredictions(data, featureVector):
    predictions = []

    for x in data:
        val = 0
        for i in range(len(featureVector)):
            val += featureVector[i] * x[0] ** i 
        predictions.append([x,val])
    return predictions

def updateParam(predictions,featureVector, learningRate):
    x0 = 0
    for i in range(len(predictions)):
        x0 += (predictions[i][0][1] - predictions[i][1]) * 1
    featureVector[0] = featureVector[0] + (learningRate*x0)/len(predictions)
    x1 = 0
    for i in range(len(predictions)):
        x1 += (predictions[i][0][1] - predictions[i][1]) * predictions[i][0][0]
    featureVector[1] = featureVector[1] + (learningRate*x1)/len(predictions)

    return [featureVector[0],featureVector[1]]
    #[[[x,y],prediction]]
def errorFunction(predictions):
    error = 0
    for i in range(len(predictions)):
        error += 0.5 * (predictions[i][0][1] - predictions[i][1])**2
    error /= len(predictions)
    return error
def epoch(iteration, featureVector, learningRate, data):
    for i in range(iteration):
        predictions = getPredictions(data, featureVector)
        error = errorFunction(predictions)
        featureVector = updateParam(predictions, featureVector, learningRate)
        print(featureVector, error)

featureVector = [2,1]
data = initData(data_size, min_x, max_x, gaussian_noise_mean, gaussian_noise_sd)
epoch(10,featureVector,0.01,data)



