library("rsample")
library("recipes")
library("keras")
#MLP
use_session_with_seed(44)
set.seed(212)


raw_data <- read.csv("data.csv", header=TRUE)
raw_data <- raw_data[,-1]
# positve outcome being epilepsy
raw_data$y[raw_data$y != 1] <- 0
raw_data$y <- as.factor(raw_data$y)



epil_split <- initial_split(raw_data)
epil_train <- training(epil_split)
# Then further split the training into validate and test
epil_split2 <- initial_split(testing(epil_split), 0.5)
epil_validate <- training(epil_split2)
epil_test <- testing(epil_split2)

for(i in 0:1){
  print(paste('class',toString(i) ,  sep=' '))
  print(sum(epil_train$y == i))
}


epil_train <- caret::downSample(x= epil_train[, -ncol(epil_train)],
                              y= epil_train$y)

for(i in 0:1){
  print(paste('class',toString(i) ,  sep=' '))
  print(sum(epil_train$Class == i))
}

epil_train <- epil_train %>% rename(y = Class)


cake <- recipe(y ~ ., data = raw_data) %>%
  #step_meanimpute(all_numeric()) %>% # impute missings on numeric values with the mean
  #step_center(all_numeric()) %>% # center by subtracting the mean from all numeric features
  step_scale(all_numeric()) %>% # scale by dividing by the standard deviation on all numeric features
  #step_unknown(all_nominal(), -all_outcomes()) %>% # create a new factor level called "unknown" to account for NAs in factors, except for the outcome (response can't be NA)
  #step_dummy(all_nominal(), one_hot = TRUE) %>% # turn all factors into a one-hot coding
  prep(training = epil_train) # learn all the parameters of preprocessing on the training data

epil_train_final <- bake(cake, new_data = epil_train) # apply preprocessing to training data
epil_validate_final <- bake(cake, new_data = epil_validate) # apply preprocessing to validation data
epil_test_final <- bake(cake, new_data = epil_test)


#convert to matrix
epil_train_x <- epil_train_final %>%
  select(-starts_with("y")) %>%
  as.matrix()
epil_train_y <- epil_train_final %>%
  select(y) %>%
  as.matrix()

epil_validate_x <- epil_validate_final %>%
  select(-starts_with("y")) %>%
  as.matrix()
epil_validate_y <- epil_validate_final %>%
  select(y) %>%
  as.matrix()

epil_test_x <- epil_test_final %>%
  select(-starts_with("y")) %>%
  as.matrix()
epil_test_y <- epil_test_final %>%
  select(y) %>%
  as.matrix()


deep.net <- keras_model_sequential() %>%
  layer_dense(units = 89, #activation='relu',
              input_shape = c(ncol(epil_train_x))) %>%
  layer_activation_relu() %>%
  layer_dropout(rate=0.5)%>%
  layer_dense(units = 45, #activation = 'relu'
              ) %>%
  layer_activation_relu() %>%
  layer_dropout(rate=0.5)%>%
  layer_dense(units = 22, activation = 'relu'
  ) %>%
  # #layer_activation_leaky_relu() %>%
  layer_dropout(rate=0.5)%>%
  
  layer_dense(units = 1, activation = "sigmoid")

deep.net %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(),
  metrics = c("accuracy")
)

callbacks <- list(kerasR::EarlyStopping(patience=5))

deep.net %>% fit(
  epil_train_x, epil_train_y,
  epochs = 50, batch_size = 32,
  validation_data = list(epil_validate_x, epil_validate_y),
  callbacks = callbacks
)

pred_test_res <- deep.net %>% predict_classes(epil_test_x)

# Confusion matrix/accuracy/AUC metrics
# (recall, in Lab03 we got accuracy ~0.80 and AUC ~0.84 from the super learner,
# and around accuracy ~0.76 and AUC ~0.74 from best other models)
table(pred_test_res, epil_test_y)

yardstick::accuracy_vec(as.factor(epil_test_y),
                        as.factor(pred_test_res))

yardstick::precision_vec(as.factor(epil_test_y),
                        as.factor(pred_test_res))

yardstick::recall_vec(as.factor(epil_test_y),
                        as.factor(pred_test_res))

MLmetrics::F1_Score(epil_test_y, pred_test_res)

