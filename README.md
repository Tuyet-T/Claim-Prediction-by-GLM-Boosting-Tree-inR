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

````````````````````````````````````````````````````````````````````````````````````````````````````````````````````

ACST8095 Assignment
Name: Ngoc Tuyet Trinh
Date: 2024-10-20
To ensure the reproducibility, first clear the current environment and add set.seed (10)

rm(list = ls())
set.seed(10)
Question a)

First, I load the required library and import the dataset

library(readr)
library(ggplot2)
## Warning: package 'ggplot2' was built under R version 4.3.1
library(rpart)
library(rpart.plot)
## Warning: package 'rpart.plot' was built under R version 4.3.3
library(caret)
## Warning: package 'caret' was built under R version 4.3.3
## Loading required package: lattice
Assg <- read_csv("CaseStudyData.csv")
## Rows: 180000 Columns: 7
## -- Column specification --------------------------------------------------------
## Delimiter: ","
## chr (1): gender
## dbl (6): Counts, distance, weight, age, carage, exposure
## 
## i Use `spec()` to retrieve the full column specification for this data.
## i Specify the column types or set `show_col_types = FALSE` to quiet this message.
There are 180000 values and 7 columns.Gender is a categorical variable. Some basic statistics are provided:

str(Assg)
## spc_tbl_ [180,000 x 7] (S3: spec_tbl_df/tbl_df/tbl/data.frame)
##  $ Counts  : num [1:180000] 0 0 0 0 0 0 0 0 0 0 ...
##  $ gender  : chr [1:180000] "female" "male" "male" "male" ...
##  $ distance: num [1:180000] 32 8 10 15 36 36 15 14 6 11 ...
##  $ weight  : num [1:180000] 1385 1607 1299 3288 1948 ...
##  $ age     : num [1:180000] 27 63 62 32 27 28 32 59 61 66 ...
##  $ carage  : num [1:180000] 21 9 9 9 10 6 16 25 25 10 ...
##  $ exposure: num [1:180000] 0.909 0.938 0.89 0.82 0.927 ...
##  - attr(*, "spec")=
##   .. cols(
##   ..   Counts = col_double(),
##   ..   gender = col_character(),
##   ..   distance = col_double(),
##   ..   weight = col_double(),
##   ..   age = col_double(),
##   ..   carage = col_double(),
##   ..   exposure = col_double()
##   .. )
##  - attr(*, "problems")=<externalptr>
summary(Assg)
##      Counts          gender             distance      weight    
##  Min.   :0.0000   Length:180000      Min.   : 2   Min.   : 653  
##  1st Qu.:0.0000   Class :character   1st Qu.:13   1st Qu.:1444  
##  Median :0.0000   Mode  :character   Median :19   Median :1914  
##  Mean   :0.1223                      Mean   :21   Mean   :2017  
##  3rd Qu.:0.0000                      3rd Qu.:28   3rd Qu.:2523  
##  Max.   :4.0000                      Max.   :50   Max.   :3996  
##       age            carage         exposure     
##  Min.   :19.00   Min.   : 2.00   Min.   :0.8000  
##  1st Qu.:34.00   1st Qu.: 9.00   1st Qu.:0.8500  
##  Median :55.00   Median :18.00   Median :0.8998  
##  Mean   :49.27   Mean   :16.45   Mean   :0.8998  
##  3rd Qu.:64.00   3rd Qu.:24.00   3rd Qu.:0.9497  
##  Max.   :77.00   Max.   :29.00   Max.   :1.0000
There is no missing value

sapply(Assg, function(x) sum(is.na(x)))
##   Counts   gender distance   weight      age   carage exposure 
##        0        0        0        0        0        0        0
Plot histogram for the numerical attributes

par(mfrow = c(2,2))
hist(Assg$exposure, main = "Distribution of Exposure", xlab = "Exposure", col = "skyblue")
hist(Assg$distance, main = "Distribution of Distance", xlab = "Distance", col = "green")
hist(Assg$weight, main = "Distribution of Weight", xlab = "Weight", col = "yellow")
hist(Assg$age, main = "Distribution of Age", xlab = "Age", col = "orange")


Plot pie chart for categorical attribute “gender”

par(mfrow = c(1, 1))
Assg$gender <- as.factor(Assg$gender)
gender_plot <- as.data.frame(table(Assg$gender))
colnames(gender_plot) <- c("Gender", "Frequency")
gender_plot$Percentage <- round((gender_plot$Frequency / sum(gender_plot$Frequency)) * 100, 1)
ggplot(gender_plot, aes(x = "", y = Frequency, fill = Gender)) +
  geom_bar(width = 1, stat = "identity") +  
  coord_polar("y") +  
  labs(title = "Gender Distribution") +  
  theme_void() + 
  geom_text(aes(label = paste0(Percentage, "%")), 
            position = position_stack(vjust = 0.5)) +
  scale_fill_manual(values = c("female" = "lightpink", "male" = "lightblue")) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))


Question b)

First, I fit the glm() for ln(λ) considering all the linear terms, quadratic terms (excluding x5) and mixed terms (excluding x5) as required in the question.

model <- glm(Counts ~ weight + distance + age + carage + gender+
              I(weight^2) + I(distance^2) + I(age^2) + I(carage^2)+
              weight*distance + weight*age + weight*carage +
              distance*age + distance*carage + 
              age*carage,
              data = Assg, family = poisson(), offset = log(exposure))
summary(model)
## 
## Call:
## glm(formula = Counts ~ weight + distance + age + carage + gender + 
##     I(weight^2) + I(distance^2) + I(age^2) + I(carage^2) + weight * 
##     distance + weight * age + weight * carage + distance * age + 
##     distance * carage + age * carage, family = poisson(), data = Assg, 
##     offset = log(exposure))
## 
## Coefficients:
##                   Estimate Std. Error z value Pr(>|z|)    
## (Intercept)     -6.031e+00  2.097e-01 -28.764  < 2e-16 ***
## weight           2.193e-04  7.260e-05   3.021  0.00252 ** 
## distance        -1.240e-03  4.562e-03  -0.272  0.78569    
## age              5.090e-02  5.623e-03   9.052  < 2e-16 ***
## carage           1.378e-01  9.457e-03  14.574  < 2e-16 ***
## gendermale       2.776e-03  1.377e-02   0.202  0.84025    
## I(weight^2)     -8.194e-09  1.180e-08  -0.694  0.48759    
## I(distance^2)    1.106e-04  5.657e-05   1.956  0.05048 .  
## I(age^2)         7.418e-05  5.106e-05   1.453  0.14622    
## I(carage^2)      1.205e-04  2.289e-04   0.526  0.59869    
## weight:distance -6.008e-07  8.696e-07  -0.691  0.48966    
## weight:age      -3.633e-07  6.298e-07  -0.577  0.56404    
## weight:carage    1.404e-06  1.316e-06   1.067  0.28588    
## distance:age     1.934e-05  4.368e-05   0.443  0.65802    
## distance:carage -1.538e-05  9.081e-05  -0.169  0.86554    
## age:carage      -1.934e-03  6.922e-05 -27.937  < 2e-16 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## (Dispersion parameter for poisson family taken to be 1)
## 
##     Null deviance: 96974  on 179999  degrees of freedom
## Residual deviance: 91801  on 179984  degrees of freedom
## AIC: 133617
## 
## Number of Fisher Scoring iterations: 6
I conduct backward selection method via step() to select the best model

set.seed(10)
model_new<- step(model, direction ="backward", trace = 0)
model_new
## 
## Call:  glm(formula = Counts ~ weight + age + carage + I(distance^2) + 
##     I(age^2) + age:carage, family = poisson(), data = Assg, offset = log(exposure))
## 
## Coefficients:
##   (Intercept)         weight            age         carage  I(distance^2)  
##    -6.042e+00      1.761e-04      5.060e-02      1.446e-01      7.610e-05  
##      I(age^2)     age:carage  
##     7.438e-05     -1.938e-03  
## 
## Degrees of Freedom: 179999 Total (i.e. Null);  179993 Residual
## Null Deviance:       96970 
## Residual Deviance: 91800     AIC: 133600
Given the benchmark, the λ = 0.01716377

lambda<-predict(model_new, list(weight = 1500, distance = 15, age = 25, carage = 4, gender ="male", exposure = 1), type="response")
lambda
##          1 
## 0.01716377
Plotting λ vs age (x3)

xage<-seq(min(Assg$age),max(Assg$age),0.5)
y<-predict(model_new, list(age=xage,
                                 weight=rep(1500,length(xage)),
                                 distance=rep(15,length(xage)),
                                 carage=rep(4,length(xage)),
                                 gender=rep("male",length(xage)),
                                 exposure=rep(1,length(xage))), type="response")

plot(xage,y,type = "l",main = "Intensity versus Age", xlab="age",ylab="intensity", col = "darkblue",lwd = 2)


Question c)

Plot the Total within sum of squares versus number of clusters. As observed, the optimal number of cluster is 4

x <- data.frame(x3 = Assg$age, x4 = Assg$carage)
wss <- c()
for (i in 1:10) wss[i] <- sum(kmeans(x, i, nstart=15)$withinss)
plot(1:10, wss, type="b", xlab="Number of Clusters", ylab="Total within groups sum of squares")


Assign the cluster label to the dataset

x3x4 <- kmeans(x,4,nstart = 15)
Assg$cluster <- as.factor(x3x4$cluster)
Plot

plot(x,col=Assg$cluster,cex = 1.5,pch = 20, xlab = "x3", ylab = "x4",
     main = "K-means Clustering with K = 4", font.main = 2)


Question d)

Fit the glm() for ln(λ) considering all the linear and mixed terms

model2 <- glm(Counts ~ weight + distance + age + carage + gender + cluster + 
                    weight:distance + weight:age + weight*carage + weight*gender + weight*cluster +
                    distance*age + distance*carage + distance*gender + distance*cluster +
                    age*carage + age*gender + age*cluster +
                    carage*gender + carage*cluster +
                    gender*cluster,
                  data = Assg, family = poisson(),offset = log(exposure))
summary(model2)
## 
## Call:
## glm(formula = Counts ~ weight + distance + age + carage + gender + 
##     cluster + weight:distance + weight:age + weight * carage + 
##     weight * gender + weight * cluster + distance * age + distance * 
##     carage + distance * gender + distance * cluster + age * carage + 
##     age * gender + age * cluster + carage * gender + carage * 
##     cluster + gender * cluster, family = poisson(), data = Assg, 
##     offset = log(exposure))
## 
## Coefficients:
##                       Estimate Std. Error z value Pr(>|z|)    
## (Intercept)         -4.017e+00  6.530e-01  -6.152 7.67e-10 ***
## weight               3.730e-04  1.459e-04   2.556   0.0106 *  
## distance             1.428e-02  1.028e-02   1.390   0.1646    
## age                  2.463e-02  9.880e-03   2.493   0.0127 *  
## carage               5.350e-02  5.143e-02   1.040   0.2983    
## gendermale           5.225e-02  2.222e-01   0.235   0.8141    
## cluster2             3.350e-01  4.192e-01   0.799   0.4242    
## cluster3            -2.304e-01  8.029e-01  -0.287   0.7741    
## cluster4             3.876e-01  4.296e-01   0.902   0.3670    
## weight:distance     -5.791e-07  8.733e-07  -0.663   0.5073    
## weight:age          -3.382e-06  2.127e-06  -1.590   0.1118    
## weight:carage       -5.102e-07  4.431e-06  -0.115   0.9083    
## weight:gendermale    2.388e-05  1.877e-05   1.273   0.2032    
## weight:cluster2     -6.663e-05  7.551e-05  -0.882   0.3775    
## weight:cluster3      4.378e-05  6.683e-05   0.655   0.5124    
## weight:cluster4     -6.913e-05  9.303e-05  -0.743   0.4574    
## distance:age        -9.560e-05  1.486e-04  -0.643   0.5201    
## distance:carage     -2.280e-04  3.100e-04  -0.736   0.4619    
## distance:gendermale -1.884e-03  1.307e-03  -1.442   0.1494    
## distance:cluster2   -3.710e-03  5.280e-03  -0.703   0.4823    
## distance:cluster3    3.539e-03  4.677e-03   0.757   0.4492    
## distance:cluster4   -3.344e-04  6.515e-03  -0.051   0.9591    
## age:carage          -5.452e-04  7.718e-04  -0.706   0.4800    
## age:gendermale      -1.801e-03  3.190e-03  -0.565   0.5724    
## age:cluster2        -2.764e-02  5.829e-03  -4.741 2.12e-06 ***
## age:cluster3        -9.401e-05  1.183e-02  -0.008   0.9937    
## age:cluster4        -3.886e-03  1.175e-02  -0.331   0.7408    
## carage:gendermale    5.881e-03  6.644e-03   0.885   0.3761    
## carage:cluster2     -2.681e-02  2.577e-02  -1.041   0.2981    
## carage:cluster3      6.029e-03  8.309e-03   0.726   0.4681    
## carage:cluster4     -1.276e-02  2.516e-02  -0.507   0.6120    
## gendermale:cluster2 -4.574e-02  1.133e-01  -0.404   0.6865    
## gendermale:cluster3 -6.901e-02  1.003e-01  -0.688   0.4915    
## gendermale:cluster4 -1.576e-01  1.396e-01  -1.129   0.2591    
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## (Dispersion parameter for poisson family taken to be 1)
## 
##     Null deviance: 96974  on 179999  degrees of freedom
## Residual deviance: 91338  on 179966  degrees of freedom
## AIC: 133190
## 
## Number of Fisher Scoring iterations: 6
Perform backward selection method to select the best model

model_new2<- step(model2, direction ="backward", trace = 0)
model_new2
## 
## Call:  glm(formula = Counts ~ weight + distance + age + carage + gender + 
##     cluster + distance:gender + age:cluster, family = poisson(), 
##     data = Assg, offset = log(exposure))
## 
## Coefficients:
##         (Intercept)               weight             distance  
##          -3.0674236            0.0001759            0.0046585  
##                 age               carage           gendermale  
##           0.0093299            0.0166915            0.0433579  
##            cluster2             cluster3             cluster4  
##          -0.1582888            0.5378466            0.3309167  
## distance:gendermale         age:cluster2         age:cluster3  
##          -0.0018978           -0.0277271           -0.0080096  
##        age:cluster4  
##          -0.0116754  
## 
## Degrees of Freedom: 179999 Total (i.e. Null);  179987 Residual
## Null Deviance:       96970 
## Residual Deviance: 91350     AIC: 133200
Given benchmark in b), For cluster 1, λ = 0.08902562

lambda2_1<-predict(model_new2, list(weight = 1500, distance = 15, age = 25, carage = 4, gender ="male",cluster = "1", exposure = 1), type="response")
lambda2_1
##          1 
## 0.08902562
Given benchmark in b), For cluster 2, λ = 0.03799512

lambda2_2<-predict(model_new2, list(weight = 1500, distance = 15, age = 25, carage = 4, gender ="male",cluster = "2", exposure = 1), type="response")
lambda2_2
##          1 
## 0.03799512
Given benchmark in b), For cluster 3, λ = 0.1247772

lambda2_3<-predict(model_new2, list(weight = 1500, distance = 15, age = 25, carage = 4, gender ="male",cluster = "3", exposure = 1), type="response")
lambda2_3
##         1 
## 0.1247772
Given benchmark in b), For cluster 4, λ = 0.09256919

lambda2_4<-predict(model_new2, list(weight = 1500, distance = 15, age = 25, carage = 4, gender ="male",cluster = "4", exposure = 1), type="response")
lambda2_4
##          1 
## 0.09256919
Plotting λ vs age (x3) for each cluster

par(mfrow = c(2, 2))
y_1<-predict(model_new2, list(age=xage,
                               weight=rep(1500,length(xage)),
                               distance=rep(15,length(xage)),
                               carage=rep(4,length(xage)),
                               gender=rep("male",length(xage)),
                               cluster=rep("1",length(xage)),
                               exposure=rep(1,length(xage))), type="response")
plot(xage,y_1,type = "l",main = "Intensity versus Age For Cluster 1", xlab="age",ylab="intensity", col = "red",lwd = 2)

y_2<-predict(model_new2, list(age=xage,
                               weight=rep(1500,length(xage)),
                               distance=rep(15,length(xage)),
                               carage=rep(4,length(xage)),
                               gender=rep("male",length(xage)),
                               cluster=rep("2",length(xage)),
                               exposure=rep(1,length(xage))),
                               type="response")
plot(xage,y_2,type = "l",main = "Intensity versus Age For Cluster 2", xlab="age",ylab="intensity", col = "green",lwd = 2)

y_3<-predict(model_new2, list(age=xage,
                               weight=rep(1500,length(xage)),
                               distance=rep(15,length(xage)),
                               carage=rep(4,length(xage)),
                               gender=rep("male",length(xage)),
                               cluster=rep("3",length(xage)),
                               exposure=rep(1,length(xage))),
                               type="response")
plot(xage,y_3,type = "l",main = "Intensity versus Age For Cluster 3", xlab="age",ylab="intensity", col = "orange",lwd = 2)

y_4<-predict(model_new2, list(age=xage,
                               weight=rep(1500,length(xage)),
                               distance=rep(15,length(xage)),
                               carage=rep(4,length(xage)),
                               gender=rep("male",length(xage)),
                               cluster=rep("4",length(xage)),
                               exposure=rep(1,length(xage))),
                               type="response")
plot(xage,y_4,type = "l",main = "Intensity versus Age For Cluster 4", xlab="age",ylab="intensity", col = "purple",lwd = 2)


Question e)

Fit a Poisson Regression Tree without x6

Reg_T<- rpart(cbind(exposure,Counts)~ 
                            weight+distance+age+carage+gender,
                            data=Assg,
                            method="poisson")
Find the optimal cp

cp.select <- function(tree){
  min.x <- which.min(tree$cp[, 4])
  for(i in 1:nrow(tree$cp)){
    if(tree$cp[i, 4] < tree$cp[min.x, 4] 
       + tree$cp[min.x, 5]){
      return(tree$cp[i, 1])
    }
  }
}
cp.best <-cp.select(Reg_T)
cp.best
## [1] 0.01
Prune and Plot - The optimal size of tree is 2 with 3 leaves

prune <- prune(Reg_T, cp=cp.best)
rpart.plot(prune)


As a benchmark, λ = 0.0395817

lambda_Prune <- predict(prune,list(weight = 1500, distance = 15, age = 25, carage = 4, gender ="male",exposure = 1), type = "vector")
lambda_Prune
##         1 
## 0.0395817
Plot λ versus x3 (age), exposure = 1

xage<-seq(min(Assg$age),max(Assg$age),0.5)
y_prune<-predict(prune,list(age=xage,
                               weight=rep(1500,length(xage)),
                               distance=rep(15,length(xage)),
                               carage=rep(4,length(xage)),
                               gender=rep("male",length(xage)),
                               exposure=rep(1,length(xage)),
                               type ="vector"))
plot(xage,y_prune,main = "Intensity versus Age", xlab="age",ylab="intensity", col = "darkgreen",lwd = 2)


Question f)

Split 90% into train dataset, 10% is validation dataset

set.seed(10)
split <- sample(c(1:nrow(Assg)),0.9 * nrow(Assg), replace = FALSE)
train <- Assg[split, ]
val <- Assg[-split, ]
Initialize λ(1) = 1, first I define a potential range for K - no of boosting step, max_depth - size of tree, shrinkage - shrinkage parameter.

train$fit <- train$exposure
val$fit <- val$exposure
K_set <- seq(20, 50, by = 10)
max_depth_set <- 1:5                   
shrinkage_set <- seq(0.1, 0.5, by = 0.1)   
Tuning hyperparameters via gridsearch

set.seed(10)
# Define hyperparameter grid
grid <- expand.grid(K = K_set,
                    max_depth = max_depth_set,
                    shrinkage = shrinkage_set)

# Fit the model
fit_model <- function(K, max_depth, shrinkage) {
  
  # Reset fit values to initial exposure before each fit for tuning
  val$fit <- val$exposure
  
  for (k in 1:K) {
    boosting_step <- rpart(cbind(fit, Counts) ~ weight + distance + carage + age + gender,
                           data = train,
                           method = "poisson",
                           control = rpart.control(maxsurrogate = 0, maxdepth = max_depth, 
                                                   xval = 1, cp = 0.00001, minbucket = 10000))
    
    val$fit <- val$fit * (predict(boosting_step, newdata = val))^shrinkage
  }
  
  # Calculate validation error
  val_error <- 2 * (sum(log((val$Counts / val$fit)^val$Counts)) - sum(val$Counts) + sum(val$fit)) / nrow(val)
  return(val_error)
}

# Initialize best values
best_val_error <- Inf
best_params <- list()

# Tune hyperparameters
for (i in 1:nrow(grid)) {
  val_error <- fit_model(grid$K[i], grid$max_depth[i], grid$shrinkage[i])
  
  # Choose the min val error and the corresponding parameters
  if (val_error < best_val_error) {
    best_val_error <- val_error
    best_params <- grid[i, ]
  }
}

best_params
##    K max_depth shrinkage
## 9 20         3       0.1
As benchmark, predict λ = 0.001212091

set.seed(10)
benchmark <- data.frame(weight = 1500,
                        distance = 15,
                        age = 25,
                        carage = 4,
                        gender = "male",
                        exposure = 1)
benchmark$fit <- benchmark$exposure

for (k in 1:best_params$K) {
  boost_Reg <- rpart(cbind(fit, Counts) ~ weight + distance + carage + age + gender,
                         data = train,
                         method = "poisson",
                         control = rpart.control(maxsurrogate = 0, maxdepth = best_params$max_depth, xval = 1,
                                                 cp = 0.00001,
                                                 minbucket = 10000))
  benchmark$fit<- benchmark$fit*(predict(boost_Reg,newdata=benchmark))^best_params$shrinkage
}
benchmark$fit
## [1] 0.001212091
Plot λ vs x3 (age)

set.seed(10)
plot <-  data.frame(
  age = xage,
  weight = rep(1500, length(xage)),
  distance = rep(15, length(xage)),
  carage = rep(4, length(xage)),
  gender = rep("male", length(xage)),
  exposure = rep(1, length(xage))
)

plot$fit <- plot$exposure

for (k in 1:best_params$K) {
  boost_Reg <- rpart(cbind(fit, Counts) ~ weight + distance + carage + age + gender,
                         data = train,
                         method = "poisson",
                         control = rpart.control(maxsurrogate = 0, maxdepth = best_params$max_depth, xval = 1,
                                                 cp = 0.00001,
                                                 minbucket = 10000))
plot$fit<- plot$fit*(predict(boost_Reg,newdata=plot))^best_params$shrinkage
}

plot(xage,plot$fit,main = "Intensity versus Age", xlab="age",ylab="intensity", col = "darkgreen",lwd = 2)


Question g)

Since base model is GLM in b), λ(1) is from the best model fitted in b)

train$fit_GLM <- train$exposure*predict(model_new, newdata = train, type = "response")
val$fit_GLM <- val$exposure*predict(model_new, newdata = val, type = "response")
Tuning hyperparameters via gridsearch

set.seed(10)
# Fit the model
fit_GLM_model <- function(K, max_depth, shrinkage) {
  
  # Reset fit values to initial exposure before each fit
  val$fit_GLM <- val$fit_GLM
  
  for (k in 1:K) {
    boosting_step <- rpart(cbind(fit_GLM, Counts) ~ weight + distance + carage + age + gender,
                           data = train,
                           method = "poisson",
                           control = rpart.control(maxsurrogate = 0, maxdepth = max_depth, 
                                                   xval = 1, cp = 0.00001, minbucket = 10000))
    
val$fit_GLM <- val$fit_GLM*(predict(boosting_step, newdata = val))^shrinkage
  }
  
  # Calculate validation error
  val_error_GLM <- 2 * (sum(log((val$Counts / val$fit_GLM)^val$Counts)) - sum(val$Counts) + sum(val$fit_GLM)) / nrow(val)
  return(val_error_GLM)
}

# Initialize best values
best_val_error_GLM <- Inf
best_params_GLM <- list()

# Tune hyperparameters
for (i in 1:nrow(grid)) {
  val_error_GLM <- fit_model(grid$K[i], grid$max_depth[i], grid$shrinkage[i])
  
  # Update best parameters if current error is lower
  if (val_error_GLM < best_val_error_GLM) {
    best_val_error_GLM <- val_error_GLM
    best_params_GLM <- grid[i, ]
  }
}

best_params_GLM
##    K max_depth shrinkage
## 9 20         3       0.1
As benchmark, predict λ = 2.080405e-05

set.seed(10)
benchmark$fit_GLM <- benchmark$exposure*predict(model_new, newdata = benchmark, type = "response")

for (k in 1:best_params_GLM$K) {
  boost_Reg_GLM <- rpart(cbind(fit, Counts) ~ weight + distance + carage + age + gender,
                         data = train,
                         method = "poisson",
                         control = rpart.control(maxsurrogate = 0, maxdepth = best_params_GLM$max_depth, xval = 1,
                                                 cp = 0.00001,
                                                 minbucket = 10000))
  benchmark$fit_GLM<- benchmark$fit_GLM*(predict(boost_Reg_GLM,newdata=benchmark))^best_params_GLM$shrinkage
}
benchmark$fit_GLM
## [1] 2.080405e-05
Plot λ vs x3 (age)

set.seed(10)

# Lambda (1) is predicted by best model in b) based on the plot dataframe

plot$fit_GLM <- plot$exposure*predict(model_new, newdata = plot, type = "response")

for (k in 1:best_params_GLM$K) {
  boost_Reg_GLM <- rpart(cbind(fit, Counts) ~ weight + distance + carage + age + gender,
                         data = train,
                         method = "poisson",
                         control = rpart.control(maxsurrogate = 0, maxdepth = best_params_GLM$max_depth, xval = 1,
                                                 cp = 0.00001,
                                                 minbucket = 10000))
plot$fit_GLM<- plot$fit_GLM*(predict(boost_Reg_GLM,newdata=plot))^best_params_GLM$shrinkage
}

plot(xage,plot$fit_GLM,main = "Intensity versus Age", xlab="age",ylab="intensity", col = "darkorange",lwd = 2)


Question h)

Set up for 10 K-fold CV

set.seed(10)
# Make new dataset to avoid changing in the original dataset
Assg1 <- Assg
Assg1$random <- runif(nrow(Assg1))
Assg1 <- Assg1[order(Assg1$random),]

K <- 10
Assg1$CV <- rep(1:K, length = nrow(Assg1))
For b) CV error = 0.05117425

set.seed(10)
val_error <- 0
for (k in 1:K) {
  Assg1.train <- Assg1[Assg1$CV != k, ]
  Assg1.val <- Assg1[Assg1$CV == k, ]
  
  predictions <- predict(model_new, newdata = Assg1.val, type = "response")
  
  # Calculate validation error using the specified metric
  val_error <- 2 * (sum(log((Assg1.val$Counts / predictions)^Assg1.val$Counts)) -   sum(Assg1.val$Counts) + sum(predictions)) / nrow(Assg1.val)
}

# CV error
val_error/K
## [1] 0.05117425
For d) CV error = 0.05093664

set.seed(10)
val_error <- 0
for (k in 1:K) {
  Assg1.val <- Assg1[Assg1$CV == k, ]
  # Predict on new dataset
  predictions <- predict(model_new2, newdata = Assg1.val, type = "response")
  # Calculate validation error 
  val_error <- 2 * (sum(log((Assg1.val$Counts / predictions)^Assg1.val$Counts)) -   sum(Assg1.val$Counts) + sum(predictions)) / nrow(Assg1.val)
}

# CV error
val_error/K
## [1] 0.05093664
For e) CV error = 0.0513968

set.seed(10)
val_error <- 0

for (k in 1:K) {
  Assg1.val <- Assg1[Assg1$CV == k, ]
  # Predict the best model in e) on new validation dataset
  predictions <- predict(prune, newdata = Assg1.val, type = "vector")
  # Calculate validation error
  val_error <- 2 * (sum(log((Assg1.val$Counts / predictions)^Assg1.val$Counts)) -  sum(Assg1.val$Counts) + sum(predictions)) / nrow(Assg1.val)
}

# CV error
val_error/K
## [1] 0.0513968
For f) CV error = 0.07586909

set.seed(10)
for (k in 1:10) {
  Assg1.val <- Assg1[Assg1$CV == k, ]
  
  #The initial is same when doing tuning parameters
  Assg1.val$fit <- Assg1.val$exposure  

  # Boosting loop, use the fitted model in f) so data is still "train"
  for (i in 1:best_params$K) {
    boosting_step <- rpart(cbind(fit, Counts) ~ weight + distance + carage + age + gender,
                           data = train,
                           method = "poisson",
                           control = rpart.control(maxsurrogate = 0, maxdepth = best_params$max_depth,   xval = 1, cp = 0.00001, minbucket = 10000))
    
    # Predict the best model in f) in new test set Assg1.val
    
    Assg1.val$fit <- Assg1.val$fit * (predict(boosting_step, newdata = Assg1.val))^best_params$shrinkage
  }
  # Calculate validation error of f) in new test set Assg1.val
  val_error <- 2 * (sum(log((Assg1.val$Counts / Assg1.val$fit)^Assg1.val$Counts)) - sum(Assg1.val$Counts) + sum(Assg1.val$fit)) / nrow(Assg1.val)
}

#CV error
val_error/K
## [1] 0.07586909
For g) CV error = 0.05148029

set.seed(10)
for (k in 1:10) {
    Assg1.train <- Assg1[Assg1$CV != k, ]
    Assg1.val <- Assg1[Assg1$CV == k, ]
    
    #The initial is same when doing tuning parameters
    Assg1.val$fit_GLM <- Assg1.val$exposure*predict(model_new, newdata = Assg1.val, type = "response")

  # Boosting loop, use the fitted model in g) so data is still "train"
  for (i in 1:best_params_GLM$K) {
    boosting_step <- rpart(cbind(fit_GLM, Counts) ~ weight + distance + carage + age + gender,
                           data = train,
                           method = "poisson",
                           control = rpart.control(maxsurrogate = 0, maxdepth = best_params_GLM$max_depth, 
                                                   xval = 1, cp = 0.00001, minbucket = 10000))
    
    #Predict the best model in g) in new test set Assg1.val
    
    Assg1.val$fit_GLM <- Assg1.val$fit_GLM * (predict(boosting_step, newdata = Assg1.val))^best_params_GLM$shrinkage
  }
  
  # Calculate validation error g) the new test set Assg1.val
  val_error <- 2 * (sum(log((Assg1.val$Counts / Assg1.val$fit_GLM)^Assg1.val$Counts)) - sum(Assg1.val$Counts) + sum(Assg1.val$fit_GLM)) / nrow(Assg1.val)
}

 #CV error
val_error/K
## [1] 0.05148029
