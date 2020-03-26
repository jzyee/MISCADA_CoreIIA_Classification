library("data.table")
library("mlr3verse")
library("caret")
library("mlr3viz")
library("precrec")
library("xgboost")
library("e1071")
library("kknn")
library("DataExplorer")

raw_data <- read.csv("data.csv", header=TRUE)
# raw_data <- raw_data[,-1]
raw_data$y[raw_data$y != 1] <- 0
DataExplorer::plot_boxplot(raw_data, by = "y", ncol = 10)

# positve outcome being epilepsy
raw_data$y[raw_data$y != 1] <- 0
raw_data$y <- as.factor(raw_data$y)

raw_data <- caret::downSample(x= raw_data[, -ncol(raw_data)],
                              y= raw_data$y)

raw_data <- as.data.frame(raw_data[,-1])
DataExplorer::plot_boxplot(raw_data, by = "Class", ncol = 10)
