"""
NOTE: Before running this code, you must execute the following command in the shell:
conda install python-graphviz

This program takes in a data set from the UCI Machine Learning Repository and tests several versions of a Decision Tree learner. Parameters that can be modified between trials in various combinations. The bulk of the program uses code from Sam's Decision Tree sample on eLearn.

David Nguyen, Mohawk College, October 10 2019"""

import numpy as np
from sklearn import tree
import csv
import random
import graphviz

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

# exclude first row of data which has the feature names
# for this data set, the labels are in the last column (column #10)
allData = np.array(dataList[1:])[:,0:9]
allLabels = np.array(dataList[1:])[:,9]

# store array for feature names
featureNames = np.array(dataList[0])
featureNames = np.delete(featureNames, -1)

# create dictionary of all labels in the training
labelList = []
for label in allLabels:
    if label not in labelList:
        labelList.append(label)


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


def classify(outputFileName = "tree.dot", maxDepthValue = None, criterionSetting = "gini", minSamplesLeafValue = 1, maxFeaturesValue = None, minSamplesValue = 2):
    """Function to run sklearn's decision tree algorithm on data set

    Parameters:
    outputFileName (default: "tree.dot"): the name of the .pdf/.dot file to be outputted with the tree
    maxDepthValue (default: "None"): parameter for limiting tree depth. if none, sklearn sets no limit on tree depth.
    criterion (default: "gini"): how to measure the quality of a split. can be either "gini" or "entropy"
    minSamplesLeafValue (default: 1): minimum number of samples that a leaf must contain
    maxFeaturesValue (default: None): maximum number of features to compare when looking for a split. if default, sklearn sets max features = n_features
    minSamplesSplit (default: 2): minimum number of samples required to split an internal node.

    Returns:
    no return values

    """

    accuracyResults = []
    numberOfRuns = 0;

    while numberOfRuns < 5:
        # generate a new holdout eachtime (randomized set of training/testing data) a new trial is run
        (trainingData, trainingLabels, testingData, testingLabels ) = randomizeHoldout(allData, allLabels)

        # create the Decision Tree object
        clf = tree.DecisionTreeClassifier(max_depth = maxDepthValue,
            criterion = criterionSetting,
            min_samples_leaf = minSamplesLeafValue,
            max_features = maxFeaturesValue,
            min_samples_split = minSamplesValue)

        ## Train the Classifier
        # use the training data for this

        clf = clf.fit(trainingData, trainingLabels)

        # classify and measure accuracy
        trainingPrediction = clf.predict(trainingData)
        trainingCorrect = (trainingPrediction == trainingLabels).sum()     # Number of Trues

        testingPrediction = clf.predict(testingData)
        testingCorrect = (testingPrediction == testingLabels).sum()     # Number of Trues

        # store the testing results accuracy to use in average calculation later
        accuracyResults.append(testingCorrect/len(testingPrediction)*100)


        # graph the Decision Tree!
        dot_data = tree.export_graphviz(clf,
            out_file=None,
            feature_names=featureNames,
            class_names=labelList,
            filled=True,
            rounded=True,
            special_characters=True
        )

        graph = graphviz.Source(dot_data)
        graph.render(outputFileName)

        numberOfRuns +=1

    # average the accuracy of all 5 decision tree simulations
    averageAccuracy = np.mean(np.array(accuracyResults))

    # print output to include parameters used for this trial
    print("Name: " + outputFileName + "\nMax Depth: " + str(maxDepthValue) + " | Criterion: " + criterionSetting + "\nMinimum Samples Per Leaf: " + str(minSamplesLeafValue) + " | Maximum Features When Splitting: " + str(maxFeaturesValue) + "\nMinimum Samples To Split Node: " + str(minSamplesValue))
    print("Average Accuracy: " + str(averageAccuracy))
    print(accuracyResults)

## MAIN
classify("treeOne.dot")
classify("treeTwo.dot", None, "entropy")
classify("treeThree.dot", 2)
classify("treeFour.dot", 2, "entropy")
classify("treeFive.dot", 5, "entropy")
classify("treeSix.dot", None, "gini", 5)
classify("treeSeven.dot", None, "gini", 25)
classify("treeEight.dot", None, "gini", 5, 1)
classify("treeNine.dot", None, "entropy", 5, 5)
classify("treeTen.dot", None, "gini", 1, None, 40)
classify("treeEleven.dot", None, "entropy", 1, 3, 2)
classify("treeTwelve.dot", 3, "gini", 5, 9, 25)





