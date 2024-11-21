Objective

This project aims to develop a model to predict the frequency of claims from policies in a motor insurance portfolio. The approach is to estimate the claims count (modeled as a Poisson distribution) for various policies by using different model types (linear regression, Poisson regression, clustering, Poisson regression tree, and boosting trees) and evaluation them to identify the best model.

Data Overview

The dataset “CaseStudyData.csv” contains 180,000 values and 7 columns for each insurance policy, representing the following features: Car weight, Annually driven distance, Age of the driver, Age of the car, Gender of the driver (male/female), Exposure associated with the policy, Claims count

Each row in the dataset represents a different policy. The datas set has no missing values.

Task Breakdown

a) Simple Data Exploration

Perform initial data exploration to understand the dataset and provide summary statistics for each feature.

Visualize the distributions of key variables (e.g., age, car weight, distance driven) by histogram and pie chart.

b) Poisson Regression Model 

Fit a GLM and do backward selection method to find the regression coefficients and estimate claim frequency

c) Clustering

Perform k-means clustering, identify the optimal number of clusters and assign a cluster label to each data point and plot the clusters on a scatter plot.

d) Poisson Regression with Clusters 

Using the optimal number of clusters from part (c), add the cluster label as a new feature (x6) to the dataset.

Fit a GLM and do backward selection method to find the regression coefficients and estimate claim frequency.

e) Fit and Prune Poisson Regression Tree

Fit a full tree, determine the optimal size of the tree then prune it.

Estimating claim frequency for the same case as in part (b).

f) Poisson Boosting Tree with no base model

Fit a tree with no base model, tuning hyper parameters i.e: number of boosting steps, tree size, and shrinkage parameter via grid search

Estimating claim frequency for the same case as in part (b).

g) Poisson Boosting Tree with Base Model 

Fit a tree with base model GLM in b), tuning hyper parameters i.e: number of boosting steps, tree size, and shrinkage parameter via grid search

Estimating claim frequency for the same case as in part (b).

h) Model Comparison 

Perform 10-fold cross-validation to compare the models from parts (b), (d), (e), (f), and (g).

Select the best model based on the cross-validation error and report the results.
