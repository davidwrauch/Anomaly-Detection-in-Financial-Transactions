install.packages("solitude")   # Isolation Forest implementation
install.packages("caret")      # For train/test split and evaluation
install.packages("xgboost")
install.packages("themis")
install.packages("MLmetrics")
install.packages("isotree")
install.packages("nnet")

library(caret)
library(nnet)
library(isotree)
library(caret)
library(MLmetrics)
library(xgboost)
library(ranger)
library(solitude)
library(caret)
library(themis)   # SMOTE for class balancing
library(randomForest)
library(dplyr)
library(tidyverse)
library(here)
library(lubridate)
library(readxl)
library(dplyr)
library(data.table)
library(stringi)
library(openxlsx)
library(pdftools)
library(Matrix) 
library(plyr) 
library(kableExtra)
library(ranger)

setwd("C:/data exercises/anomaly")

# Load dataset from Kaggle
transactions <- read.csv('metaverse_transactions_dataset.csv') %>% 
  # Drop fields that either leak information (risk_score)
  # or are too high-cardinality to be useful without heavy encoding
  select(-transaction_type, -risk_score)

str(transactions)

# ================================
# Feature Engineering
# ================================

# Convert timestamp string → actual datetime object
transactions$timestamp <- ymd_hms(transactions$timestamp)

# Extract weekday (categorical)
transactions$day_of_week <- weekdays(transactions$timestamp)
transactions$day_of_week <- as.factor(transactions$day_of_week)

# Extract month (helps capture seasonal patterns)
transactions$month <- month(transactions$timestamp)

# Compute time gap between consecutive transactions
# This often reveals unusual bursts of activity
transactions$time_diff <- c(NA, diff(transactions$timestamp))

# Replace the first NA with a reasonable value (median gap)
transactions$time_diff[is.na(transactions$time_diff)] <- 
  median(transactions$time_diff, na.rm = TRUE)

str(transactions)

# ================================
# Data Prep
# ================================

# Convert target to factor for classification
transactions$anomaly <- as.factor(transactions$anomaly)

# Remove identifiers that don't help modeling
transactions_clean <- transactions %>%
  select(-timestamp, -sending_address, -receiving_address)

# Split predictors and target
predictors <- transactions_clean %>% select(-anomaly)
target <- transactions_clean$anomaly

# One‑hot encode categorical predictors
# (Many ML models require numeric-only input)
dummies <- dummyVars(" ~ .", data = predictors)
predictors_encoded <- data.frame(predict(dummies, newdata = predictors))

# Recombine predictors + target
transactions_encoded <- cbind(predictors_encoded, anomaly = target)

# Train/test split (stratified)
set.seed(123)
trainIndex <- createDataPartition(transactions_encoded$anomaly, p = 0.7, list = FALSE)
trainData <- transactions_encoded[trainIndex, ]
testData  <- transactions_encoded[-trainIndex, ]

# ================================
# Baseline Random Forest
# ================================

# Basic RF to see how the model behaves with imbalanced classes
rf_model <- randomForest(anomaly ~ ., data = trainData, ntree = 300, mtry = 5, importance = TRUE)

preds <- predict(rf_model, newdata = testData)
confusionMatrix(preds, testData$anomaly)

# Classic issue:
# High accuracy driven by majority classes, but zero ability to detect high-risk cases.

importance(rf_model)
varImpPlot(rf_model)

# ================================
# SMOTE + Ranger (Weighted RF)
# ================================

# Rebuild dataset (clean slate)
transactions$anomaly <- as.factor(transactions$anomaly)

transactions_clean <- transactions %>%
  select(-timestamp, -sending_address, -receiving_address)

predictors <- transactions_clean %>% select(-anomaly)
target <- transactions_clean$anomaly

dummies <- dummyVars(" ~ .", data = predictors)
predictors_encoded <- data.frame(predict(dummies, newdata = predictors))

transactions_encoded <- cbind(predictors_encoded, anomaly = target)

set.seed(123)
trainIndex <- createDataPartition(transactions_encoded$anomaly, p = 0.7, list = FALSE)
trainData <- transactions_encoded[trainIndex, ]
testData  <- transactions_encoded[-trainIndex, ]

# Use SMOTE inside caret to rebalance classes during training
train_control <- trainControl(
  method = "cv",
  number = 5,
  sampling = "smote",          # oversample minority class
  summaryFunction = multiClassSummary,
  classProbs = TRUE
)

# Small tuning grid for ranger
tuneGrid <- expand.grid(
  mtry = c(3, 5),
  splitrule = "gini",
  min.node.size = 1
)

rf_model <- train(
  anomaly ~ ., data = trainData,
  method = "ranger",
  trControl = train_control,
  tuneGrid = tuneGrid,
  metric = "Mean_F1"           # optimize for balanced performance
)

preds <- predict(rf_model, newdata = testData)
confusionMatrix(preds, testData$anomaly, mode = "prec_recall")

# SMOTE helps a bit, but RF still struggles with high-risk detection.

# ================================
# XGBoost (with class weights)
# ================================

# Compute class weights to counter imbalance
class_counts <- table(trainData$anomaly)
total <- sum(class_counts)
class_weights <- total / (length(class_counts) * class_counts)

# Convert class weights → row weights
row_weights <- class_weights[trainData$anomaly]

# Prepare matrices for xgboost
dummies <- dummyVars(" ~ .", data = trainData[,-ncol(trainData)])
train_matrix <- data.frame(predict(dummies, newdata = trainData[,-ncol(trainData)]))
test_matrix  <- data.frame(predict(dummies, newdata = testData[,-ncol(testData)]))

# Ensure test has same columns as train
missing_cols <- setdiff(names(train_matrix), names(test_matrix))
for (col in missing_cols) test_matrix[[col]] <- 0

extra_cols <- setdiff(names(test_matrix), names(train_matrix))
test_matrix <- test_matrix[ , !(names(test_matrix) %in% extra_cols) ]

test_matrix <- test_matrix[ , names(train_matrix) ]

# Build DMatrix objects
dtrain <- xgb.DMatrix(data = as.matrix(train_matrix), label = as.numeric(trainData$anomaly) - 1)
dtest  <- xgb.DMatrix(data = as.matrix(test_matrix),  label = as.numeric(testData$anomaly) - 1)

params <- list(
  objective = "multi:softmax",
  num_class = length(levels(trainData$anomaly)),
  eval_metric = "mlogloss"
)

xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 200,
  watchlist = list(train = dtrain, test = dtest)
)

preds <- predict(xgb_model, dtest)

preds_factor <- factor(preds,
                       levels = 0:(length(levels(trainData$anomaly))-1),
                       labels = levels(trainData$anomaly))

confusionMatrix(preds_factor, testData$anomaly, mode = "prec_recall")

# XGBoost still fails to meaningfully detect high-risk cases.

# ================================
# Ranger with Manual Class Weights
# ================================

# Assign heavier weights to minority classes
row_weights <- ifelse(trainData$anomaly == "high_risk", 5,
                 ifelse(trainData$anomaly == "moderate_risk", 2, 1))

rf_model <- ranger(
  anomaly ~ ., 
  data = trainData,
  probability = TRUE,
  case.weights = row_weights,
  num.trees = 500,
  importance = "impurity"
)

probs <- predict(rf_model, data = testData)$predictions
preds <- colnames(probs)[max.col(probs)]

confusionMatrix(factor(preds, levels = levels(testData$anomaly)),
                testData$anomaly, mode = "prec_recall")

# Better recall for high-risk, but precision collapses.

# ================================
# Isolation Forest
# ================================

train_matrix <- as.matrix(trainData[,-ncol(trainData)])
test_matrix  <- as.matrix(testData[,-ncol(testData)])

# Fit unsupervised anomaly detector
iso_model <- isolation.forest(train_matrix, ntrees = 500, ndim = 2)

# Higher score = more anomalous
scores <- predict(iso_model, test_matrix, type = "score")

# Flag top 10% as high-risk
threshold <- quantile(scores, 0.90)
preds <- ifelse(scores > threshold, "high_risk", "low_risk")

# Optional: map into 3 classes
preds3 <- cut(scores,
              breaks = c(-Inf, quantile(scores, 0.70), quantile(scores, 0.90), Inf),
              labels = c("low_risk", "moderate_risk", "high_risk"))

ref <- factor(ifelse(testData$anomaly == "high_risk","high_risk","low_risk"),
              levels = c("high_risk","low_risk"))

preds <- factor(preds, levels = c("high_risk","low_risk"))

confusionMatrix(preds, ref, mode = "prec_recall")

confusionMatrix(factor(preds3, levels = levels(testData$anomaly)),
                testData$anomaly, mode = "prec_recall")

# Isolation Forest behaves like a generic anomaly detector:
# lots of false positives, very weak alignment with labeled classes.

# ================================
# Hybrid Model: Logistic Regression + Isolation Forest Score
# ================================

# Add anomaly score as a new feature
trainData$iso_score <- predict(iso_model, train_matrix, type = "score")
testData$iso_score  <- predict(iso_model, test_matrix, type = "score")

# Weighted multinomial logistic regression
row_weights <- ifelse(trainData$anomaly == "high_risk", 5,
                 ifelse(trainData$anomaly == "moderate_risk", 2, 1))

glm_model <- multinom(anomaly ~ ., 
                      data = trainData,
                      weights = row_weights)

preds <- predict(glm_model, newdata = testData)

confusionMatrix(factor(preds, levels = levels(testData$anomaly)),
                testData$anomaly, mode = "prec_recall")

# First model that meaningfully detects high-risk cases.
# Recall becomes extremely high, at the cost of precision.

# ================================
# Threshold Tuning for High-Risk
# ================================

evaluate_threshold <- function(thresh, probs, test_labels) {
  preds <- apply(probs, 1, function(x) {
    if (x["high_risk"] > thresh) "high_risk"
    else colnames(probs)[which.max(x)]
  })
  
  cm <- confusionMatrix(factor(preds, levels = levels(test_labels)),
                        test_labels, mode = "prec_recall")
  
  cm$byClass["Class: high_risk", "F1"]
}

# Sweep thresholds to find best F1 for high-risk
thresholds <- seq(0.05, 0.9, by = 0.05)
results <- sapply(thresholds, function(t) evaluate_threshold(t, probs, testData$anomaly))

best_thresh <- thresholds[which.max(results)]
best_f1 <- max(results, na.rm = TRUE)

cat("Best threshold:", best_thresh, "with F1 =", best_f1, "\n")

plot(thresholds, results, type = "b", pch = 19,
     xlab = "High-risk threshold", ylab = "F1 (high_risk)",
     main = "Threshold tuning for high-risk detection")

# Macro F1 version
evaluate_macroF1 <- function(thresh, probs, test_labels) {
  preds <- apply(probs, 1, function(x) {
    if (x["high_risk"] > thresh) "high_risk"
    else colnames(probs)[which.max(x)]
  })
  cm <- confusionMatrix(factor(preds, levels = levels(test_labels)),
                        test_labels, mode = "prec_recall")
  mean(cm$byClass[,"F1"], na.rm = TRUE)
}

macro_results <- sapply(thresholds, function(t) evaluate_macroF1(t, probs, testData$anomaly))

plot(thresholds, macro_results, type="b", pch=19,
     xlab="Threshold", ylab="Macro F1",
     main="Macro F1 vs Threshold")

predict_with_threshold <- function(probs, threshold = 0.25) {
  apply(probs, 1, function(x) {
    if (x["high_risk"] > threshold) "high_risk"
    else colnames(probs)[which.max(x)]
  })
}

preds_final <- predict_with_threshold(probs, threshold = 0.25)

confusionMatrix(factor(preds_final, levels = levels(testData$anomaly)),
                testData$anomaly, mode = "prec_recall")

# Final takeaway:
# Threshold tuning + weighted logistic regression + iso_score
# gives extremely strong recall for high-risk cases,
# at the cost of more false positives.
