

#machine learning methods using lr3
library("data.table")
library("mlr3verse")
library("caret")
library("mlr3viz")
library("precrec")
library("xgboost")
library("e1071")
library("kknn")
library("rsample")
library("recipes")
set.seed(44)
#use_session_with_seed(44)

raw_data <- read.csv("data.csv", header=TRUE)
raw_data <- raw_data[,-1]

# positve outcome being epilepsy
raw_data$y[raw_data$y != 1] <- 0
raw_data$y <- as.factor(raw_data$y)


for(i in 0:1){
  print(paste('class',toString(i) ,  sep=' '))
  print(sum(raw_data$y == i))
}



raw_data <- caret::downSample(x= raw_data[, -ncol(raw_data)],
                       y= raw_data$y)

for(i in 0:1){
  print(paste('class',toString(i) ,  sep=' '))
  print(sum(raw_data$Class == i))
}


table(raw_data$Class)


epilepsy_task <- TaskClassif$new(id= 'epilepsy',
                                 backend= raw_data,
                                 target= "Class",
                                 positive= '1')
cv5 <- rsmp("cv", folds = 5)
cv5$instantiate(epilepsy_task)

mlr_learners

lrn_baseline <- lrn("classif.featureless", predict_type = "prob")
lrn_cart     <- lrn("classif.rpart", predict_type = "prob")
lrn_cart_cp  <- lrn("classif.rpart", predict_type = "prob", cp = 0.016, id = "cartcp")
lrn_ranger   <- lrn("classif.ranger", predict_type = "prob")
lrn_xgboost  <- lrn("classif.xgboost", predict_type = "prob")
  pl_xgb <- po("encode") %>>%
    po(lrn_xgboost)
lrn_nb <- lrn("classif.naive_bayes", predict_type = 'prob')
lrn_kknn <- lrn("classif.kknn", predict_type='prob')
lrn_svm <- lrn("classif.svm", predict_type = 'prob')
lrn_log_reg  <- lrn("classif.log_reg", predict_type = "prob")
lrn_lda <- lrn("classif.lda", predict_type = "prob")

res <- benchmark(data.table(
  task       = list(epilepsy_task),
  learner    = list(lrn_baseline,
                    lrn_cart,
                    lrn_cart_cp,
                    pl_xgb,
                    lrn_kknn,
                    lrn_svm,
                    lrn_log_reg,
                    lrn_lda),
  resampling = list(cv5)
), store_models = TRUE)



res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   #msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr"),
                   msr("classif.recall")))

acc_res <- res$aggregate(msr("classif.acc"))
barplot(acc_res$classif.acc,
        names.arg = acc_res$learner_id,
        col = rainbow(nrow(acc_res)))

fnr_res <- res$aggregate(msr("classif.recall"))
barplot(fnr_res$classif.recall,
        names.arg = fnr_res$learner_id,
        col = rainbow(nrow(fnr_res)))


###########################
# boosting to the extreme!#
###########################
library("data.table")
library("mlr3verse")
library("caret")
library("mlr3viz")
library("precrec")
library("xgboost")
library("e1071")
library("kknn")
set.seed(44)

raw_data <- read.csv("data.csv", header=TRUE)
raw_data <- raw_data[,-1]

# positve outcome being epilepsy
raw_data$y[raw_data$y != 1] <- 0
raw_data$y <- as.factor(raw_data$y)

raw_data <- caret::downSample(x= raw_data[, -ncol(raw_data)],
                              y= raw_data$y)




table(raw_data$Class)

set.seed(44)
epilepsy_task <- TaskClassif$new(id= 'epilepsy',
                                 backend= raw_data,
                                 target= "Class",
                                 positive= '1')
cv5 <- rsmp("cv", folds = 5)
cv5$instantiate(epilepsy_task)

#model init
lrn_baseline <- lrn("classif.featureless", predict_type = "prob")
lrn_cart     <- lrn("classif.rpart", predict_type = "prob")
lrn_cart_cp  <- lrn("classif.rpart", predict_type = "prob", cp = 0.016, id = "cartcp")
lrn_ranger   <- lrn("classif.ranger", predict_type = "prob")
lrn_xgboost  <- lrn("classif.xgboost", predict_type = "prob")
  pl_xgb <- po("encode") %>>%
    po(lrn_xgboost)
lrn_nb <- lrn("classif.naive_bayes", predict_type = 'prob')
lrn_kknn <- lrn("classif.kknn", predict_type='prob')
lrn_svm <- lrn("classif.svm", predict_type = 'prob')
lrn_log_reg  <- lrn("classif.log_reg", predict_type = "prob")
lrn_lda <- lrn("classif.lda", predict_type = "prob")

pl_factor <- po("encode")

cv5 <- rsmp("cv", folds = 5)
cv5$instantiate(epilepsy_task)


cv_list <- list(rep(0, length(model_list)))

lrn_svm <- lrn("classif.svm", predict_type = 'prob', id = 'XTREME_svm')
xgboost_svm <- pl_factor %>>% 
  po("learner_cv", lrn_xgboost) %>>% 
  po(lrn_svm)

lda_svm <- pl_factor %>>% 
  po("learner_cv", lrn_lda) %>>% 
  po(lrn_svm)



xgboost_cart <- pl_factor %>>% 
  po("learner_cv", lrn_xgboost) %>>% 
  po(lrn_cart)

lda_cart <- pl_factor %>>% 
  po("learner_cv", lrn_lda) %>>% 
  po(lrn_cart)

res <- benchmark(data.table(
    task = list(epilepsy_task),
    learner = list(xgboost_svm,
                   xgboost_cart,
                   lda_svm,
                   lda_cart),
    resampling = list(cv5)
    
  ),store_models = TRUE
)



res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   #msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr"),
                   msr("classif.recall")))

acc_res <- res$aggregate(msr("classif.acc"))
barplot(acc_res$classif.acc,
        names.arg = acc_res$learner_id,
        col = rainbow(nrow(acc_res)),
        ylim=c(0,1))
abline(h=0.955,lty=2,col='red')
acc_res


rec_res <- res$aggregate(msr("classif.recall"))
barplot(rec_res$classif.recall,
        names.arg = rec_res$learner_id,
        col = rainbow(nrow(rec_res)),
        ylim=c(0,1))
abline(h=0.941,lty=2,col='red')
rec_res


