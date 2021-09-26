#%%
#numpy, pandas and amth external libraries for manipulation of matrix arrays:
import numpy as np
import pandas as pd
import math
#matplotlib library for outputting graphs for neural network analysis:
import matplotlib._pylab_helpers
import matplotlib.pyplot as plt


#CONFIGURATION OF THE ANN IN TERMS OF NUMBER OF INPUT/OUTPUT NODES FROM THE DATASET AND NUMBER OF HIDDEN NODES
#Ability to change the no. of inputs, outputs, and hidden layers for the neural network as mentioned on specification
inputNodesNo = int(input("Enter the number of inputs in your network"))
outputNodesNo = int(input("Enter the number of outputs in your network"))
hiddenRange = ([math.ceil(inputNodesNo/2),(2*inputNodesNo)])
hiddenNodesNo = int(input("Enter the number of hidden nodes in your network between ranges " + str(hiddenRange[0]) + " to " + str(hiddenRange[1])))
dataFileName1 = input("Enter the name of your txt file :")
totalInpOut1 = inputNodesNo + outputNodesNo 
totalInpOut2 = inputNodesNo + outputNodesNo 

#DATA PREPROCESSING:
#Formatting the dataset to their corresponding rows and the items to floats for matrix manipulation:
data = open(dataFileName1, "r")
dataString = data.read()
dataList = np.array(dataString.split())
for i in range(0, len(dataList),1):
    dataList[i] = float(dataList[i])

def chunk_list(listName,chunk_size):
    for i in range (0, len(listName),chunk_size):
        yield listName[i:i + chunk_size]
formattedFullDataArray = np.array(list(chunk_list(dataList,totalInpOut1)))

#Function for splitting the dataset into 3 parts for cross validation:
def split_three(rawDataset, ratio=[0.6, 0.2, 0.2]):
    training, validation, testing = ratio
    assert(np.sum(ratio) == 1.0)
    indicies_for_splitting = [int(len(rawDataset) * training), int(len(rawDataset) * (training+validation))]
    train, val, test = np.split(rawDataset, indicies_for_splitting)
    return train, val, test

#RANDOMISING, SPLITTING, AND STANDARDISING THE FULL DATASET:
#Creating seperate matrices for the Input and Output data in the dataset
randomisedDataSet = np.random.permutation(formattedFullDataArray)
InputArray = np.delete(randomisedDataSet, np.s_[inputNodesNo:totalInpOut2], axis = 1)
OutputArray = np.delete(randomisedDataSet, np.s_[0:inputNodesNo], axis = 1).astype(np.float64)
#splitting the input and output arrays further into the 3 datasets for K-Fold Cross Validation:
TrainingInputs, ValidationInputs, TestInputs = split_three(InputArray)
TrainingOutputs, ValidationOutputs, TestOutputs = split_three(OutputArray)
#Concatenating the Training and validation sets to get the max and min values of this new set to apply when standardising:
TrainValidationInputs = np.concatenate((TrainingInputs, ValidationInputs), axis=0)
TrainValidationOutputs = np.concatenate((TrainingOutputs, ValidationOutputs), axis=0)
#print(TrainValidationOutputs)
TrainValidationOutputs= np.array(TrainValidationOutputs).astype(np.float64)
TrainValidationInputs= np.array(TrainValidationInputs).astype(np.float64)
TrainingOutputs= np.array(TrainingOutputs).astype(np.float64)
TrainingInputs= np.array(TrainingInputs).astype(np.float64)
ValidationInputs= np.array(ValidationInputs).astype(np.float64)
ValidationOutputs= np.array(ValidationOutputs).astype(np.float64)
TestInputs= np.array(TestInputs).astype(np.float64)
TestOutputs= np.array(TestOutputs).astype(np.float64)
#Standardising the 3 dataset's inputs and outputs:
standardisedTrainInput=((0.8*((TrainingInputs-np.min(TrainValidationInputs,0))) / ((np.max(TrainValidationInputs,0))-np.min(TrainValidationInputs,0))).astype(np.float64)+0.1)
standardisedTrainOutput=((0.8*((TrainingOutputs-np.min(TrainValidationOutputs,0))) / ((np.max(TrainValidationOutputs,0))-np.min(TrainValidationOutputs,0))).astype(np.float64)+0.1)
standardisedValidationInput=((0.8*((ValidationInputs-np.min(TrainValidationInputs,0))) / ((np.max(TrainValidationInputs,0))-np.min(TrainValidationInputs,0))).astype(np.float64)+0.1)
standardisedValidationOutput=((0.8*((ValidationOutputs-np.min(TrainValidationOutputs,0))) / ((np.max(TrainValidationOutputs,0))-np.min(TrainValidationOutputs,0))).astype(np.float64)+0.1)
standardisedTestInput=((0.8*((TestInputs-np.min(TrainValidationInputs,0))) / ((np.max(TrainValidationInputs,0))-np.min(TrainValidationInputs,0))).astype(np.float64)+0.1)
standardisedTestOutput=((0.8*((TestOutputs-np.min(TrainValidationOutputs,0))) / ((np.max(TrainValidationOutputs,0))-np.min(TrainValidationOutputs,0))).astype(np.float64)+0.1)




#FUNCTIONS FOR ANALYSING ANN PERFORMANCE:
def MeanSquaredError(predictions, actual):
    predictionsDestandardised =((((predictions-0.1)/0.8)*((np.max(TrainValidationOutputs,0))-np.min(TrainValidationOutputs,0))).astype(np.float64)+np.min(TrainValidationOutputs,0))
    N = actual.size 
    mse = ((predictionsDestandardised - ValidationOutputs)**2).sum() / (2*ValidationOutputs.size)
    return np.round(mse,2)

def Precision(predictions, actual):
    predictionsDestandardised =((((predictions-0.1)/0.8)*((np.max(TrainValidationOutputs,0))-np.min(TrainValidationOutputs,0))).astype(np.float64)+np.min(TrainValidationOutputs,0))
    predicions_correct = predictionsDestandardised - ValidationOutputs
    Precision = predicions_correct.mean()
    return np.round(Precision,2)

def RMSE(predictions, actual):
    predictionsDestandardised =((((predictions-0.1)/0.8)*((np.max(TrainValidationOutputs,0))-np.min(TrainValidationOutputs,0))).astype(np.float64)+np.min(TrainValidationOutputs,0))
    rmse = math.sqrt(((predictions - ValidationOutputs)**2).sum() / (2*ValidationOutputs.size))
    return np.round(rmse,2)

def CE(predictions,actual):
    predictionsDestandardised =((((predictions-0.1)/0.8)*((np.max(TrainValidationOutputs,0))-np.min(TrainValidationOutputs,0))).astype(np.float64)+np.min(TrainValidationOutputs,0))
    setMean = np.mean(ValidationOutputs)
    ce=1-((((predictionsDestandardised - ValidationOutputs)**2).sum() / (2*ValidationOutputs.size))/(((ValidationOutputs - setMean)**2).sum()))
    return np.round(ce,2)
    
def RSqr(predictions,actual):
    predictionsDestandardised =((((predictions-0.1)/0.8)*((np.max(TrainValidationOutputs,0))-np.min(TrainValidationOutputs,0))).astype(np.float64)+np.min(TrainValidationOutputs,0))
    setMean = np.mean(ValidationOutputs)
    rsqr=((((predictionsDestandardised - setMean))*(ValidationOutputs-setMean)).sum()/(((ValidationOutputs-setMean)**2).sum())*(((predictionsDestandardised-(np.mean(predictionsDestandardised))**2).sum())))**2
    return np.round(rsqr,2)

def MSRE(predictions,actual):
    N=actual.size
    setMean=np.mean(standardisedValidationOutput)
    msre = (1/np.size(standardisedValidationOutput))*((((predictions-standardisedValidationOutput)/standardisedValidationOutput)**2).sum())
    return np.round(msre,2)


#PRODUCING MATRICES FOR THE BIAS SO THAT THE BIAS WEIGHT IS MULTIPLIED BY INPUT VAL 1 DURING SUMMATION:
biasForHiddenNodesTrain = (np.ones((1,len(standardisedTrainInput)), dtype=float)).T 
standardisedTrainInputwBias = np.append(standardisedTrainInput,biasForHiddenNodesTrain,1)
biasForOutputNodesTrain = (np.ones((1, len(standardisedTrainInput)), dtype=float)).T
biasForHiddenNodesValidation = (np.ones((1,len(standardisedValidationInput)), dtype=float)).T 
standardisedValidationInputwBias = np.append(standardisedValidationInput,biasForHiddenNodesValidation,1)
biasForOutputNodesValidation = (np.ones((1, len(standardisedValidationInput)), dtype=float)).T
biasForHiddenNodesTest = (np.ones((1,len(standardisedTestInput)), dtype=float)).T 
standardisedTestInputwBias = np.append(standardisedTestInput,biasForHiddenNodesTest,1)
biasForOutputNodesTest = (np.ones((1, len(standardisedTestInput)), dtype=float)).T

#PRODUCING THE WEIGHT MATRIX (INC BIAS) FOR THE INPUT-HIDDEN LAYER AND HIDDEN-OUTPUT LAYER:
lowerWBoundw1= (-2/(inputNodesNo))
upperWBoundw1= (2/(inputNodesNo))
lowerWBoundw2= (-2/(hiddenNodesNo))
upperWBoundw2= (2/(hiddenNodesNo))
Weights1 = np.random.uniform(low=lowerWBoundw1, high=upperWBoundw1, size=(inputNodesNo+1,hiddenNodesNo))
Weights2 = np.random.uniform(low=lowerWBoundw2, high=upperWBoundw2, size=(hiddenNodesNo+1, outputNodesNo))
#print(Weights1)
#print(Weights2)
dataSetRows = standardisedTrainOutput.size
monitoring = {"MeanSquaredError": [], "Precision": [], "RMSE" : [], "CE": [], "RSqr":[], "MSRE":[]}


#THE BACKPROPOGATION FORWARD PASS AND BACKWARD PASS ALGORITHM:

#Sigmoid Activation Function for the forward pass:
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#Setting the learning rate and the no. of epochs for the ANN model:
learning_rate = 0.1
epochs = 10000

for epoch in range(epochs):
    boldCounter = int(epoch)
    #Feedforward on the training dataset
    hiddenLayer = sigmoid(np.dot(standardisedTrainInputwBias,Weights1))
    hiddenLayerwBias = np.append(hiddenLayer,biasForOutputNodesTrain,1)
    outputLayer = sigmoid(np.dot(hiddenLayerwBias,Weights2))

    #BACKPROPOGATION on the training dataset
    output_layer_delta = (outputLayer - standardisedTrainOutput) * outputLayer * (1 - outputLayer)
    hidden_layer_delta = np.dot(output_layer_delta, Weights2.T) * hiddenLayerwBias * (1 - hiddenLayerwBias)
    hidden_layer_delta=np.delete(hidden_layer_delta,hiddenNodesNo,1)
    #Weight updates basic:
    Weights2BoldD = Weights2
    Weights1BoldD = Weights1
    Weights2 = Weights2 - learning_rate * np.dot(hiddenLayerwBias.T, output_layer_delta) / dataSetRows
    Weights1 = Weights1 - learning_rate * (np.dot(standardisedTrainInputwBias.T, hidden_layer_delta)) / dataSetRows

    #Bold Drive Improvement Implementation:
    if(boldCounter == 1000):
        boldCounter = 0
        errorFuncO= standardisedTrainOutput - outputLayer
        hiddenLayer = sigmoid(np.dot(standardisedTrainInputwBias,Weights1))
        hiddenLayerwBias = np.append(hiddenLayer,biasForOutputNodesTrain,1)
        outputLayer = sigmoid(np.dot(hiddenLayerwBias,Weights2))
        errorFuncN=standardisedTrainOutput - outputLayer

        errorFuncOMean = (np.sum(errorFuncO))/dataSetRows
        errorFuncNMean = (np.sum(errorFuncN))/dataSetRows
        errorPercentageChange = (((errorFuncNMean-errorFuncOMean)/errorFuncOMean) * 100.0)
        if(errorPercentageChange > 4.0 and learning_rate>0.01 and learning_rate<0.5):
            #increase the learning rate by 5% and keep the weights the same as the old values
            learning_rate=learning_rate*1.05
            Weights1= Weights2BoldD 
            Weights2 = Weights2BoldD

        elif(learning_rate>0.01 and learning_rate<0.5):
            learning_rate = learning_rate *0.7


    #Feedforward on the validation set:
    hiddenLayerV = sigmoid(np.dot(standardisedValidationInputwBias,Weights1))
    hiddenLayerwBiasV = np.append(hiddenLayerV,biasForOutputNodesValidation,1)
    outputLayerV = sigmoid(np.dot(hiddenLayerwBiasV,Weights2))

    #Monitor validation dataset performance for evaluation
    accV = Precision(outputLayerV, standardisedValidationOutput)
    mseV = MeanSquaredError(outputLayerV, standardisedValidationOutput)
    rmseV = RMSE(outputLayerV,standardisedValidationOutput)
    ceV = CE(outputLayerV,standardisedValidationOutput)
    rSqrV = CE(outputLayerV,standardisedValidationOutput)
    msreV = MSRE(outputLayerV,standardisedValidationOutput)
    monitoring["Precision"].append(accV)
    monitoring["MeanSquaredError"].append(mseV)
    monitoring["RMSE"].append(rmseV)
    monitoring["CE"].append(ceV)
    monitoring["RSqr"].append(rSqrV)
    monitoring["MSRE"].append(msreV)


#TESTING THE ANN ON THE TEST DATASET:
#Feedforward on test dataset:
hiddenLayerT = sigmoid(np.dot(standardisedTestInputwBias,Weights1))
hiddenLayerwBiasT = np.append(hiddenLayerT,biasForOutputNodesTest,1)
outputLayerT = sigmoid(np.dot(hiddenLayerwBiasT,Weights2))

#Destandardising Test Dataset Predictions for Evaluation:
destandardisedTestOutputs=((((standardisedTestOutput-0.1)/0.8)*((np.max(TrainValidationOutputs,0))-np.min(TrainValidationOutputs,0))).astype(np.float64)+np.min(TrainValidationOutputs,0))
destandardisedPredictedTestOut=((((outputLayerT-0.1)/0.8)*((np.max(TrainValidationOutputs,0))-np.min(TrainValidationOutputs,0))).astype(np.float64)+np.min(TrainValidationOutputs,0))
destandardisedOutputDiff = TestOutputs - destandardisedPredictedTestOut 
destandardisedMean = destandardisedOutputDiff.mean()


#Test Dataset Tables and Graphs:
testResultsTable = np.round(np.concatenate((TestOutputs,destandardisedPredictedTestOut,destandardisedOutputDiff),1),2)
testResultsTable = pd.DataFrame(data=testResultsTable, columns=["Actual Outputs", "Predicted Outputs", "Difference"])
print("")
print("Table Showing Results of Test Dataset: ")
print(testResultsTable)
print("")
mseTest = ((destandardisedPredictedTestOut-TestOutputs)**2).sum()/(2*(np.size(TestOutputs)))
RMSETest = math.sqrt(mseTest)
print("Root Mean Square Error of Test Dataset:")
print(round(RMSETest,2))
TestOutputsMean =np.mean(TestOutputs)
CETest=1-((mseTest)/(((TestOutputs-TestOutputsMean)**2).sum()))
print("Coeffecient Effeciency of Test Dataset:")
print(round(CETest,2))
destandardisedPredictedTestOutMean=np.mean(destandardisedPredictedTestOut)
RSqrTest=((((TestOutputs-TestOutputsMean))*(destandardisedPredictedTestOut-destandardisedPredictedTestOutMean)).sum()/(((TestOutputs-TestOutputsMean)**2).sum())*(((destandardisedPredictedTestOut-destandardisedPredictedTestOutMean)**2).sum()))**2
print("Coeffecient of Determination")
print(round(RSqrTest,2))
MSRETest = (1/np.size(TestOutputs))*((((destandardisedPredictedTestOut-TestOutputs)/TestOutputs)**2).sum())
print("Mean Squared Relative Error")
print(MSRETest,2)
scatterGraphTest = plt.scatter(TestOutputs,destandardisedPredictedTestOut)
print(scatterGraphTest,2)

#Categorising Results into groups to further measure accuracy of performance: 
x=0
TestOutputsConc = zip(outputLayerT,standardisedTestOutput)
for item1,item2 in TestOutputsConc:
    if (item1<-2 and item2<-0.2):
        x=x+1
    elif (item1<0.2 and item2<0.2):
        x=x+1
    elif (item1>0.2 and item2>0.2):
        x=x+1
print("ACCURACY MEASURED BY CORRECT CATEGORISATIONS OF LOW, MEDIUM, OR HIGH FLOOD RISK:" , x )


#Validation Set Tables and Graphs for Evaluation:
monitoring_df = pd.DataFrame(monitoring)
print("")
print("Table and Graph showing MSE and Precision Results on Validation Dataset: ")
print("")
print(monitoring_df)
fig, axes = plt.subplots(1, 6, figsize=(35,4))
monitoring_df.MeanSquaredError.plot(ax=axes[0], title="Mean Squared Error")
monitoring_df.Precision.plot(ax=axes[1], title="Precision")
monitoring_df.RMSE.plot(ax=axes[2], title="RMSE")
monitoring_df.CE.plot(ax=axes[3], title="CE")
monitoring_df.RSqr.plot(ax=axes[4], title="RSqr")
monitoring_df.MSRE.plot(ax=axes[5], title="MSRE")



# %%
