# KNN/Decision Tree Analysis of Fertility Data

# Data Description

- Data set name: Fertility
- Source: UCI Machine Learning Repository
- Description/range of features:
  - Season: season in which the analysis was performed.
    - 1) winter, 2) spring, 3) Summer, 4) fall
    - (-1, -0.33, 0.33, 1)
  - Age: age at the time of analysis.
    - 18-36
    - (0, 1)
  - Child diseases: diseases experienced as a child (ie , chicken pox, measles, mumps, polio)
    - 1) yes, 2) no
    - (0, 1)
  - Serious Trauma: if the participant experienced accident or serious trauma
    - 1) yes, 2) no
    - (0, 1)
  - Surgical intervention: if the participant has had surgical intervention
    - 1) yes, 2) no
    - (0, 1)
  - High fevers: if the participant experienced a high fever within the last year
    - 1) less than three months ago, 2) more than three months ago, 3) no
    - (-1, 0, 1)
  - Alcohol consumption: Frequency of alcohol consumption
    - 1) several times a day, 2) every day, 3) several times a week, 4) once a week, 5) hardly ever or never
    - (0, 1)
  - Smoking Habit: The participants level of smoking habit
    - 1) never, 2) occasional 3) daily
    - (-1, 0, 1)
  - Number of hours spent sitting per day
    - 1-16
    - (0, 1)
  - Fertility: Diagnosis of participant&#39;s fertility
    - Diagnosis normal (N), altered (O)
- Classification task: Classification, Regression
- Statistics
  - number of features: 10
  - number of items in each class:
    - Altered Fertility (O): 12                Normal Fertility (N): 88

# The Tests

## Part 1: k-Nearest Neighbour

The testing and training split were computed using a function called randomizeHoldout. This function would create an array of randomized numbers and a cut-off index of 75% of the length of total rows. Using those two values and Numpy&#39;s related arrays, 4 new arrays containing the testing/training data and labels can be created and returned.

The KNN algorithm was coded into the classify function which had parameters to modify the k parameter, distance, weighted voting using inverse scoring and normalization. I also increased the number of trials done with varying parameters as I felt 5 were too few to make comparisons.

The &#39;K&#39; parameter was set to determine how many training data points would be taken into consideration when determining the label of a new testing data point. Values of &#39;K&#39; were set at only odd values to avoid ties.

The distance parameter was used to set which integer m to use in the Minkowski distance formula. The three variations used values of 1, 2 and 3 for Manhattan, Euclidean and Minkowski Distance of 3 respectively.

The inverse scoring parameter determined whether the KNN algorithm would use weighted scoring when determining which label to assign new test data. The &#39;K&#39; nearest neighbours were no longer ranked purely on which ones were the closest to the test item but rather by the inverse of their distance. This results in nearby neighbours with low distances scoring significantly higher when determining the label of the test item. If the distance of the nearest neighbour was 0, they were assigned a high score of 100 since taking the inverse of 0. I set it to score only 100 instead of a very large number like 10000 since I felt that if there were enough neighbours of another label to outscore 100, it should be labelled as such since the point with distance of 0 is likely an outlier.

The normalization parameter uses the following formula to place all values between 0 and 1:

  **v**** norm**=(**v **-** min**)/**r**. This makes it so all features of the data are equally weighted.

## Part 2: Decision Trees

The testing/training split was computed the same way as it was in the kNN trials.

The Decision Tree algorithms were coded into the classify function which had parameters to modify SKLearn&#39;s decision tree algorithm with: output file name, max\_depth, criterion, min\_samples\_leaf, max\_features and min\_samples\_split. I also increased the number of trials with varying parameters as I felt 5 were too few to make accurate comparisons.

The output file name variable allowed me to specific what the .pdf file would be named so that each time a trial was run, it would have its own decision tree graphic.

The max\_depth parameter set the depth of the tree (how many levels/tiers appeared in the tree).

The criterion parameter controlled what function was used to measure the quality of the split. The default value of &quot;gini&quot;, uses Gini Impurity, is the measure of the probability of a randomly chosen element being misclassified if each split is adopted. Whereas the alternative value of &quot;entropy&quot;, uses Information Gain, is the measure of how much each split reduces the amount of information needed to correctly classify the items. Both essentially measure how pure the split created is.

The min\_samples\_leaf parameter set how many samples a leaf must contain at the bottom of the tree.

The max\_features parameter set how maximum number of features the algorithm took into consideration when looking for the best split.

The min\_samples\_split parameter determined how many samples an internal node must have before it can be split.
