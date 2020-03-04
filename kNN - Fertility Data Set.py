"""
This program takes in a data set from the UCI Machine Learning Repository and tests several versions of various K-Nearest Neighbour algorithms. Parameters that can be modified between trials in various combinations: k-value, distance type, inverse scoring, normalization


David Nguyen, Mohawk College, September 23 2019
"""
# imports
from matplotlib import pyplot as plt
import numpy as np
import csv
import random

# https://stackoverflow.com/questions/34399172/why-does-my-python-code-print-the-extra-characters-%C3%AF-when-reading-from-a-tex
# include encoding parameter to remove extra characters (" ï»¿ ") from reading in data
# open csv file
dataFile = open("data.csv", "r", encoding='utf-8-sig')

# create CSV readers
reader = csv.reader(dataFile, delimiter=",")

# read in data and store into python lists
dataList = []
for row in reader:
    dataList.append(row)

# store array for feature names
featureNames = np.array(dataList[0])

# exclude first row of data which has the feature names
# for this data set, the labels are in the last column (column #10)
allData = np.array(dataList[1:])[:,0:9]
allLabels = np.array(dataList[1:])[:,9]


##functions
def randomizeHoldout(data, labels):
    """Creates randomized holdout sets from the data set in a 75% training and 25% testing split
    
    parameters:
    data (string array): numpy array of data to be randomly split
    labels( string array): numpy array of labels to be randomly split
    
    returns:
    trainingData: 75% of the data in random order
    trainingLabels: 75% of the labels in random order
    testingData: 25% of the data in random order 
    testingLabels 25% of the labels in random order
    """
    # variables to help with sorting/splitting data
    totalRows = len(data)
    cutoff = int(0.75*totalRows)
    
    # create related numpy array for use to split up the 25/75 training and testing data
    randomizer = np.arange(totalRows)
    np.random.shuffle(randomizer)
    
    # split the training and test data
    # use related arrays to randomize the rows and seperate the data into training (75%) and testing data (25%)
    trainingData = (data[randomizer.argsort()[:cutoff]])
    testingData = (data[randomizer.argsort()[cutoff:]])
    
    trainingLabels = (labels[randomizer.argsort()[:cutoff]])
    testingLabels = (labels[randomizer.argsort()[cutoff:]])
    
    # convert all values in the array to float
    trainingData = trainingData.astype(float)
    testingData = testingData.astype(float)
    
    return (trainingData, trainingLabels, testingData, testingLabels)

## WORKING function
def classify(allData, allLabels, k, distanceValue = 2, inverseScoring = False, normalizeData = False):
    """Predicts the labels of holdouts from data using NNK method and calculates the accuracy rate of the algorithm
    
    parameters:
    allData (string array): numpy array of all the data read in from the data set (excludes labels)
    allLabels (string array): numpy array of all the labels read in from the data set
    k (int): Parameter for setting number of 'k' nearest neighbours
    distanceValue (int, default = 2): Parameter for setting what distance formula to use in KNN calculation.
         1 = manhattan distance, 2 = euclidean distance, 3 = minkowski distance
    inverseScoring (boolean, default = False): Determines if algorithm will use the inverse distance for scoring labels
    normalizeData (boolean, default = False): Determine if data will be normalized before running KNN algorithm
    
    returns:
    no return values
    
    """
    # checks if user entered a valid distance parameter (1-3)
    # sets the string for the summary print statement
    if distanceValue == 1:
        distanceType = "Manhattan Distance"
    elif distanceValue == 2:
        distanceType = "Euclidean Distance"
    elif distanceValue == 3:
        distanceType = "Minkowsky Distance"
    else:
        print("Please enter a valid distanceValue between 1-3")
        return
    
    # temp variables
    accuracyResults = []
    numberOfRuns = 0
    allDataFloat = []
    
    allDataFloat = allData.astype(np.float)
    allColumnsMin = allDataFloat.min(axis=0)
    allColumnsMax = allDataFloat.max(axis=0)
    allColumnsRange = allColumnsMax - allColumnsMin
    
    allDataNormalized = (allDataFloat - allColumnsMin)/allColumnsRange
    
    
    # run the KNN simulation 5 times
    while numberOfRuns < 5:
        # temp variable to hold predicted labels
        calculatedLabels = []
        
        # generate a new holdout eachtime (randomized set of training/testing data) a new trial is run
        if normalizeData == False:
            (trainingData, trainingLabels, holdoutData, holdoutLabels) = randomizeHoldout(allData, allLabels)
        elif normalizeData == True:
            (trainingData, trainingLabels, holdoutData, holdoutLabels) = randomizeHoldout(allDataNormalized, allLabels)
        
        # run the nnk function on all rows of the testing data to predict labels
        for i in holdoutData:
            # calculates distance of new point from all other points in the training data on all features
            # take the absoluate value of x - y to avoid NAN from negative exponents
            distance = ((( abs(trainingData - i) ) ** distanceValue).sum(axis=1))**(1/distanceValue)

            # create dictionary of all labels in the training 
            labelDictionary = {}
            for label in trainingLabels:
                if label not in labelDictionary:
                    labelDictionary.update({label : 0})
            
            # REGULAR DISTANCE SCORING
            if inverseScoring == False:
                # sort the array of distances from closest to furthest
                # use related arrays to return the labels of the 'k' closest points
                votes = trainingLabels[distance.argsort()][:k]
                
                # print(votes)
                # tally up how many of the 'k' nearest neighbours belong to each label
                for vote in votes:
                    labelDictionary[vote] += 1
                
                # print(labelDictionary)
                # the label with the most occurences is the estimated label
                # resource: https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary/280156#280156
                voted = max(labelDictionary, key=labelDictionary.get)
                # print(voted)
                
            #INVERSE SCORING
            else:
                
                # calculate inverse distance
                inverseDistance = []
                for d in distance:
                    # if d == 0, inverse would be infinity... assign points with distance of 0 a score of 50
                    # 50 was assigned as the highest score since the highest scores seen that were distance of 0.03 were 33
                    if d == 0:
                        inverseDistance.append(100)
                    else:
                        inverseDistance.append(1/d)
    
                # set votes as values of inverse distance
                # use 'inverse distances' as score for each vote
                # add them up for each label
                # the label with the higher score from votes is the predicted label
                
    
                # inverse formula uses array slices of -k since the largest inverse distances (highest scoring) are at the end of the array
                # get labels and scores of highest inverse distances
                inverseVotes = trainingLabels[np.array(inverseDistance).argsort()][-k:]
                inverseScores = np.sort(np.array(inverseDistance))[-k:]
                # print(inverseVotes)
                # print(inverseScores)
                
                # amount of votes/scores will always be k, therefore the number of indexes will always be (k - 1)...
                counter = 0
                while counter < k:
                    # cycle through the votes / scores and update the dictionary accordingly
                    labelDictionary[inverseVotes[counter]] += inverseScores[counter]
                    counter += 1
                
                # print(labelDictionary)
    
                # return the label that had the most votes
                # resource: https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary/280156#280156
                voted = max(labelDictionary, key=labelDictionary.get)
                # print(voted)
            
            # add the label determined with NNK to array of predicted labels
            calculatedLabels.append(voted)
        
        # calculatedLables are all the labels predicted from NNK the algorithm
        # compare the predicted labels with the actual labels and calculate the % accuracy
        accuracy = ((int(sum(holdoutLabels == calculatedLabels)))/25)*100

        # add it to the array of accuracy results
        accuracyResults.append(accuracy)
        numberOfRuns += 1
    
    # average the accuracy of all 5 KNN simulations
    averageAccuracy = np.mean(np.array(accuracyResults))
    
    # print output to include what 'K' value used, the type of distance used, average accuracy and the accuracy of each individual trial
    print("K=" + str(k) + " | " + distanceType + " | InverseScoring: " + str(inverseScoring) + " | Normalization: " + str(normalizeData))
    print("Average Accuracy: " + str(averageAccuracy))
    print(str(accuracyResults) + "\n")
    
## main
# run the NNK function with different paramenters

# no inverse scoring, no normalized data
classify(allData, allLabels, 1, 2, False, False)
classify(allData, allLabels, 3, 1, False, False)
classify(allData, allLabels, 3, 2, False, False)
classify(allData, allLabels, 3, 3, False, False)
classify(allData, allLabels, 5, 2, False, False)
classify(allData, allLabels, 5, 2, False, True)
classify(allData, allLabels, 7, 2, False, False)
classify(allData, allLabels, 1, 1, True, False)
classify(allData, allLabels, 3, 2, True, False)
classify(allData, allLabels, 5, 2, True, False)
classify(allData, allLabels, 5, 2, True, True)
classify(allData, allLabels, 7, 1, True, False)
classify(allData, allLabels, 75, 2, True, True)