# Credit Card Default Detection
# Author: Anubhav Gupta
# ----------------------------------------------------------
# Description: This is the final assignment 
# for the Harvard Data Science Professional Program 
# In this capstone project, we
# have to choose a dataset and we have to analyze it and 
# perform our machine learning tasks in complete autonomy 
# without external help.


# Install all required libraries if it is not present

if(!require(tidyverse)) install.packages("tidyverse") 
if(!require(kableExtra)) install.packages("kableExtra")
if(!require(tidyr)) install.packages("tidyr")
if(!require(tidyverse)) install.packages("tidyverse")
if(!require(stringr)) install.packages("stringr")
if(!require(ggplot2)) install.packages("ggplot2")
if(!require(gbm)) install.packages("gbm")
if(!require(dplyr)) install.packages("dplyr")
if(!require(caret)) install.packages("caret")
if(!require(xgboost)) install.packages("xgboost")
if(!require(e1071)) install.packages("e1071")
if(!require(class)) install.packages("class")
if(!require(ROCR)) install.packages("ROCR")
if(!require(randomForest)) install.packages("randomForest")
if(!require(PRROC)) install.packages("PRROC")
if(!require(reshape2)) install.packages("reshape2")
if(!require(lightgbm)) install.packages("lightgbm")

# Loading all required libraries

library(dplyr)
library(tidyverse)
library(kableExtra)
library(tidyr)
library(ggplot2)
library(gbm)
library(caret)
library(xgboost)
library(e1071)
library(class)
library(lightgbm)
library(ROCR)
library(randomForest)
library(PRROC)
library(reshape2)
library(lightgbm)

## Loading the dataset
setwd("C:/Users/Anubhav/Documents/capstone") #Set your own working directory
creditcard <- read.csv("default of credit card clients.csv")
names(creditcard) <- creditcard[1,]
creditcard=creditcard[-1,]
creditcard <- data.frame(lapply(creditcard, function(x) as.integer(as.character(x))))
names(creditcard)[names(creditcard) == "default.payment.next.month"] <- "class"
creditcard$SEX=as.factor(creditcard$SEX)
creditcard$EDUCATION=as.factor(creditcard$EDUCATION)
creditcard$MARRIAGE=as.factor(creditcard$MARRIAGE)

#Data checks

data.frame("Length" = nrow(creditcard), "Columns" = ncol(creditcard)) 

imbalanced <- data.frame(creditcard)
imbalanced$class = ifelse(imbalanced$class == 0, 'Non-default', 'Default') %>% as.factor()

# Visualize the proportion between classes

imbalanced %>%
  ggplot(aes(class)) +
  theme_minimal()  +
  geom_bar() +
  scale_x_discrete() +
  scale_y_continuous(labels = scales::comma) +
  labs(title = "Proportions between Default and Non-default Transactions",
       x = "Class",
       y = "Frequency")

# Find missing values

data.frame(sapply(creditcard, function(x) sum(is.na(x))) )
 
# Credit Age Distribution

creditcard[creditcard$class == 1,] %>%
  ggplot(aes(AGE)) + 
  theme_minimal()  +
  geom_histogram(binwidth = 5) +
  labs(title = "Default Transactions agewise distribution",
       x = "Age(in Years)",
       y = "Frequency")

# Credit Amount Distribution

creditcard[creditcard$class == 1,] %>%
  ggplot(aes(LIMIT_BAL/1000)) + 
  theme_minimal()  +
  geom_histogram(binwidth = 40) +
  labs(title = "Default Transactions Amounts Distributions",
       x = "Amount in dollars('000)",
       y = "Frequency")

# Default distribution by SEX

imbalanced$SEX = ifelse(imbalanced$SEX == 1, 'MALE', 'FEMALE') %>% as.factor()

imbalanced[imbalanced$class == 'Default',] %>%
  ggplot(aes(SEX)) +
  theme_minimal()  +
  geom_bar() +
  scale_x_discrete() +
  scale_y_continuous(labels = scales::comma) +
  labs(title = "Default Transactions by sex",
       x = "SEX",
       y = "Frequency") 

data.frame(creditcard[creditcard$class == 1,] %>%
  group_by(SEX) %>%
  summarise(count = n()) )

# Default distribution by Education

creditcard[creditcard$class == 1,] %>%
  ggplot(aes(EDUCATION)) +
  theme_minimal()  +
  geom_bar() +
  scale_x_discrete() +
  scale_y_continuous(labels = scales::comma) +
  labs(title = "Default Transactions by Education",
       x = "EDUCATION",
       y = "Frequency") 

data.frame(creditcard[creditcard$class == 1,] %>%
             group_by(EDUCATION) %>%
             summarise(count = n()) )

# Default distribution by Marriage

creditcard[creditcard$class == 1,] %>%
  ggplot(aes(MARRIAGE)) +
  theme_minimal()  +
  geom_bar() +
  scale_x_discrete() +
  scale_y_continuous(labels = scales::comma) +
  labs(title = "Default Transactions by Marriage",
       x = "MARRIAGE",
       y = "Frequency") 

data.frame(creditcard[creditcard$class == 1,] %>%
             group_by(MARRIAGE) %>%
             summarise(count = n()) )
  

# Get lower triangle of the correlation matrix

get_lower_tri<-function(cormat){
  cormat[upper.tri(cormat)] <- NA
  return(cormat)
}

# Get upper triangle of the correlation matrix

get_upper_tri <- function(cormat){
  cormat[lower.tri(cormat)]<- NA
  return(cormat)
}

reorder_cormat <- function(cormat){
  # Use correlation between variables as distance
  dd <- as.dist((1-cormat)/2)
  hc <- hclust(dd)
  cormat <-cormat[hc$order, hc$order]
}
creditcard1=creditcard %>% select(-c(SEX,EDUCATION,MARRIAGE,class))
corr_matrix <- round(cor(creditcard1),2)
corr_matrix <- reorder_cormat(corr_matrix)

upper_tri <- get_upper_tri(corr_matrix)

melted_corr_matrix <- melt(upper_tri, na.rm = TRUE)

ggplot(melted_corr_matrix, aes(Var2, Var1, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Pearson\nCorrelation") +
  theme_minimal() + 
  theme(axis.text.x = element_text(angle = 90, vjust = 1, 
                                   size = 9, hjust = 1), 
        axis.text.y = element_text(size = 9),
        axis.title.x = element_blank(),
        axis.title.y = element_blank(),
        panel.grid.major = element_blank(),
        panel.border = element_blank(),
        panel.background = element_blank(),
        axis.ticks = element_blank()) +
  coord_fixed()

# Set seed for reproducibility

set.seed(1234)

# Remove unnecessary columns from the dataset

creditcard$class <- as.factor(creditcard$class)
creditcard <- creditcard %>% select(-c(ID,BILL_AMT2,BILL_AMT3,BILL_AMT4,BILL_AMT5,BILL_AMT6,MARRIAGE))

# Split the dataset into train, test dataset and cv

train_index <- createDataPartition(
  y = creditcard$class, 
  p = .6, 
  list = F
)

train <- creditcard[train_index,]

test_cv <- creditcard[-train_index,]

test_index <- createDataPartition(
  y = test_cv$class, 
  p = .5, 
  list = F)

test <- test_cv[test_index,]
cv <- test_cv[-test_index,]

rm(train_index, test_index, test_cv)


###############################################
#MODELLING
###############################################

# Set seed 123 for reproducibility

set.seed(123)

# Build the model with Class as target and all other variables
# as predictors

naive_model <- naiveBayes(class ~ ., data = train, laplace=1)

# Predict

predictions <- predict(naive_model, newdata=test)

# Compute the AUC and AUCPR for the Naive Model

pred <- prediction(as.numeric(predictions) , test$class)

auc_val_naive <- performance(pred, "auc")

auc_plot_naive <- performance(pred, 'sens', 'spec')

# Make the relative plot


plot(auc_plot_naive, main=paste("AUC:", auc_val_naive@y.values[[1]]))


# Adding the respective metrics to the results dataset

results <-  data.frame(
  Model = "Naive Bayes", 
  AUC = auc_val_naive@y.values[[1]]
)
# Show results on a table

results %>%
  kable() %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed",           "responsive"),
                position = "center",
                font_size = 10,
                full_width = FALSE) 

# Set seed 123 for reproducibility

set.seed(123)

# Build a KNN Model with Class as Target and all other
# variables as predictors. k is set to 5

knn_model <- knn(train[,-30], test[,-30], train$class, k=5, prob = TRUE)

# Compute the AUC and AUCPR for the KNN Model

pred <- prediction(
  as.numeric(as.character(knn_model)),  as.numeric(as.character(test$class))
)

auc_val_knn <- performance(pred, "auc")

auc_plot_knn <- performance(pred, 'sens', 'spec')


# Make the relative plot


plot(auc_plot_knn, main=paste("AUC:", auc_val_knn@y.values[[1]]))

# Adding the respective metrics to the results dataset

results <- results %>% add_row(
  Model = "K-Nearest Neighbors k=5", 
  AUC = auc_val_knn@y.values[[1]]

)

# Show results on a table

results %>%
  kable() %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed",           "responsive"),
                position = "center",
                font_size = 10,
                full_width = FALSE) 

# Set seed 123 for reproducibility

set.seed(123)

# Build a SVM Model with Class as Target and all other
# variables as predictors. The kernel is set to sigmoid

svm_model <- svm(class ~ ., data = train, kernel='sigmoid')

# Make predictions based on this model

predictions <- predict(svm_model, newdata=test)

# Compute AUC and AUCPR

pred <- prediction(
  as.numeric(as.character(predictions)), as.numeric(as.character(test$class))
)

auc_val_svm <- performance(pred, "auc")

auc_plot_svm <- performance(pred, 'sens', 'spec')

# Make the relative plot


plot(auc_plot_svm, main=paste("AUC:", auc_val_svm@y.values[[1]]))


# Adding the respective metrics to the results dataset

results <- results %>% add_row(
  Model = "SVM - Support Vector Machine-Sigmoid Kernel",
  AUC = auc_val_svm@y.values[[1]]
  )

# Show results on a table

results %>%
  kable() %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed",           "responsive"),
                position = "center",
                font_size = 10,
                full_width = FALSE)

# Set seed 123 for reproducibility

set.seed(123)

# Build a SVM Model with Class as Target and all other
# variables as predictors. The kernel is set to linear

svm_model_lk <- svm(class ~ ., data = train, kernel='linear')
# Make predictions based on this model

predictions <- predict(svm_model_lk, newdata=test)

# Compute AUC and AUCPR

pred <- prediction(
  as.numeric(as.character(predictions)), as.numeric(as.character(test$class))
)

auc_val_svm_lk <- performance(pred, "auc")

auc_plot_svm_lk <- performance(pred, 'sens', 'spec')


# Make the relative plot


plot(auc_plot_svm_lk, main=paste("AUC:", auc_val_svm_lk@y.values[[1]]))


# Adding the respective metrics to the results dataset

results <- results %>% add_row(
  Model = "SVM - Support Vector Machine-Linear Kernel",
  AUC = auc_val_svm@y.values[[1]]
)

# Show results on a table

results %>%
  kable() %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed",           "responsive"),
                position = "center",
                font_size = 10,
                full_width = FALSE)

# Set seed 123 for reproducibility

set.seed(123)

# Build a Random Forest Model with Class as Target and all other
# variables as predictors. The number of trees is set to 500

rf_model <- randomForest(class ~ ., data = train, ntree = 500)

# Get the feature importance

feature_imp_rf <- data.frame(importance(rf_model))

# Make predictions based on this model

predictions <- predict(rf_model, newdata=test)

# Compute the AUC and AUPCR

pred <- prediction(
  as.numeric(as.character(predictions)), as.numeric(as.character(test$class))
)

auc_val_rf <- performance(pred, "auc")

auc_plot_rf <- performance(pred, 'sens', 'spec')

# make the relative plot

plot(auc_plot_rf, main=paste("AUC:", auc_val_rf@y.values[[1]]))


# Adding the respective metrics to the results dataset

results <- results %>% add_row(
  Model = "Random Forest",
  AUC = auc_val_rf@y.values[[1]]
  )

# Show results on a table

results %>%
  kable() %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed",           "responsive"),
                position = "center",
                font_size = 10,
                full_width = FALSE)

# Show feature importance on a table

feature_imp_rf %>%
  kable() %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed",           "responsive"),
                position = "center",
                font_size = 10,
                full_width = FALSE)

# Set seed 123 for reproducibility

set.seed(123)

# Build a GBM Model with Class as Target and all other
# variables as predictors. Distribution is bernoully, 
# number of tree is 500

gbm_model <- gbm(as.character(class) ~ .,
                 distribution = "bernoulli", 
                 data = rbind(train, test), 
                 n.trees = 500,
                 interaction.depth = 3, 
                 n.minobsinnode = 100, 
                 shrinkage = 0.01, 
                 train.fraction = 0.7,
)

# Determine the best iteration based on test data

best_iter = gbm.perf(gbm_model, method = "test")

# Make predictions based on this model

predictions = predict.gbm(
  gbm_model, 
  newdata = test, 
  n.trees = best_iter, 
  type="response"
)

# Get feature importance

feature_imp_gbm = summary(gbm_model, n.trees = best_iter)

# Compute the AUC and AUPCR

pred <- prediction(
  as.numeric(as.character(predictions)),as.numeric(as.character(test$class))
)

auc_val_gbm <- performance(pred, "auc")

auc_plot_gbm <- performance(pred, 'sens', 'spec')


# Make the relative plot


plot(auc_plot_gbm, main=paste("AUC:", auc_val_gbm@y.values[[1]]))


# Adding the respective metrics to the results dataset

results <- results %>% add_row(
  Model = "GBM - Generalized Boosted Regression",
  AUC = auc_val_gbm@y.values[[1]]
 )

# Show results on a table

results %>%
  kable() %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed",           "responsive"),
                position = "center",
                font_size = 10,
                full_width = FALSE)

# Show feature importance on a table

feature_imp_gbm %>%
  kable() %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed",           "responsive"),
                position = "center",
                font_size = 10,
                full_width = FALSE) 

# Set seed 123 for reproducibility

set.seed(123)

# Prepare the training dataset
train=train %>% select(-c(SEX,EDUCATION))
test=test %>% select(-c(SEX,EDUCATION))
cv=cv %>% select(-c(SEX,EDUCATION))

xgb_train <- xgb.DMatrix(
  as.matrix(train[, colnames(train) != "class"]), 
  label = as.numeric(as.character(train$class))
)

# Prepare the test dataset

xgb_test <- xgb.DMatrix(
  as.matrix(test[, colnames(test) != "class"]), 
  label = as.numeric(as.character(test$class))
)

# Prepare the cv dataset

xgb_cv <- xgb.DMatrix(
  as.matrix(cv[, colnames(cv) != "class"]), 
  label = as.numeric(as.character(cv$class))
)

# Prepare the parameters list. 

xgb_params <- list(
  objective = "binary:logistic", 
  eta = 0.1, 
  max.depth = 3, 
  nthread = 6, 
  eval_metric = "aucpr"
)

# Train the XGBoost Model

xgb_model <- xgb.train(
  data = xgb_train, 
  params = xgb_params, 
  watchlist = list(test = xgb_test, cv = xgb_cv), 
  nrounds = 500, 
  early_stopping_rounds = 40, 
  print_every_n = 20
)

# Get feature importance

feature_imp_xgb <- xgb.importance(colnames(train), model = xgb_model)

xgb.plot.importance(feature_imp_xgb, rel_to_first = TRUE, xlab = "Relative importance")

# Make predictions based on this model

predictions = predict(
  xgb_model, 
  newdata = as.matrix(test[, colnames(test) != "class"]), 
  ntreelimit = xgb_model$bestInd
)

# Compute the AUC and AUPCR

pred <- prediction(
  as.numeric(as.character(predictions)), as.numeric(as.character(test$class))
)

auc_val_xgb <- performance(pred, "auc")

auc_plot_xgb <- performance(pred, 'sens', 'spec')


# Make the relative plot

plot(auc_plot_xgb, main=paste("AUC:", auc_val_xgb@y.values[[1]]))


# Adding the respective metrics to the results dataset

results <- results %>% add_row(
  Model = "XGBoost",
  AUC = auc_val_xgb@y.values[[1]]
  )

# Show results on a table

results %>%
  kable() %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed",           "responsive"),
                position = "center",
                font_size = 10,
                full_width = FALSE)

# Show feature importance on a table

feature_imp_xgb %>%
  kable() %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed",           "responsive"),
                position = "center",
                font_size = 10,
                full_width = FALSE)

