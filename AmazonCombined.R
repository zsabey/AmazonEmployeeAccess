##Libraries
library(tidyverse)
library(tidymodels)
library(embed)
library(discrim)
library(kernlab)
library(themis)


trainCsv <- read_csv("train.csv")

testCsv <- read_csv("test.csv")

trainCsv <- trainCsv %>%
  mutate(ACTION = as.factor(ACTION))

##Recipe
my_recipe <- recipe(ACTION ~ ., data=trainCsv) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  ###step_other(all_nominal_predictors(), threshold = .001) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_smote(all_outcomes(),neighbors=5)# combines categorical values that occur <5% into an "other" value
#step_dummy(all_nominal_predictors())


penReg_recipe <- recipe(ACTION ~ ., data=trainCsv) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  ##step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur <5% into an "other" value
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_smote(all_outcomes(),neighbors=5)
#%>% # dummy variable encoding
# step_lencode_mixed(all_nominal_predictors(), outcome = vars(target_var)) #target encoding
# also step_lencode_glm() and step_lencode_bayes()


# NOTE: some of these step functions are not appropriate to use together

# apply the recipe to your data
prep <- prep(my_recipe)
baked <- bake(prep, new_data = NULL)
baked

##Logistic Regression workflow
my_mod <- logistic_reg() %>% #Type of model
  set_engine("glm")


logReg_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod) %>%
  fit(data = trainCsv) # Fit the workflow

logReg_predictions <- predict(logReg_workflow,
                              new_data=testCsv,
                              type="prob") # "class" or "prob" (see doc)


##Penalized Logistic Regression
my_mod <- logistic_reg(mixture=tune(), penalty=tune()) %>% #Type of model
  set_engine("glmnet")

penalized_workflow <- workflow() %>%
  add_recipe(penReg_recipe) %>%
  add_model(my_mod)

## Grid of values to tune over
tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 10) ## L^2 total tuning possibilities
tuning_grid

## Split data for CV
folds <- vfold_cv(trainCsv, v = 3, repeats=1)

## Run the CV
CV_results <- penalized_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc, f_meas, sens, recall, spec,
                               precision, accuracy)) #Or leave metrics NULL

## Plot Results (example)
collect_metrics(CV_results) %>% # Gathers metrics into DF
  filter(.metric=="roc_auc") %>%
  ggplot(data=., aes(x=penalty, y=mean, color=factor(mixture))) +
  geom_line()

collect_metrics(CV_results)

CV_results
## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best("roc_auc")
bestTune

## Finalize the Workflow & fit it
final_wf <- penalized_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=trainCsv)

penalized_predictions <- final_wf %>%
  predict(new_data = testCsv,
          type = "prob")


##Prep for submission

Sub1 <- logReg_predictions %>% 
  bind_cols(testCsv) %>% 
  select(id,.pred_1) %>%
  rename(Id= id, Action = .pred_1)


Sub2 <- penalized_predictions %>% 
  bind_cols(testCsv) %>% 
  select(id,.pred_1) %>%
  rename(Id= id, Action = .pred_1)


write_csv(Sub1, "LogRegSubmission.csv")

write_csv(Sub2, "PenRegSubmission.csv")



##SVM

trainCsv <- read_csv("train.csv")

testCsv <- read_csv("test.csv")

trainCsv <- trainCsv %>%
  mutate(ACTION = as.factor(ACTION))

#Create the recipe and bake it

SVM_recipe <- recipe(ACTION ~ ., data=trainCsv) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  ###step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur <5% into an "other" value
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_smote(all_outcomes(),neighbors=5) #%>%
#step_pca(all_predictors(),threshold = .95) #pca addition

prep <- prep(SVM_recipe)
baked <- bake(prep, new_data = NULL)
baked


## SVM model
SVM_model <- svm_rbf(rbf_sigma=tune(), cost=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kernlab")

SVM_wf <- workflow() %>%
  add_recipe(SVM_recipe) %>%
  add_model(SVM_model)

## Tune smoothness and Laplace here


## Set up grid of tuning values
tuning_grid <- grid_regular(rbf_sigma(),
                            cost(),
                            levels = 3) ## L^2 total tuning possibilities

## Set up K-fold CV
folds <- vfold_cv(trainCsv, v = 3, repeats=1)

## Run the CV
## smoothness 1.5, Laplace 0
CV_results <- SVM_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc)) #Or leave metrics NULL

## Find best tuning parameters
collect_metrics(CV_results) %>% # Gathers metrics into DF
  filter(.metric=="roc_auc") %>%
  ggplot(data=., aes(x=rbf_sigma, y=mean, color=factor(cost))) +
  geom_line()

collect_metrics(CV_results)


## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best("roc_auc")
bestTune

## Finalize the Workflow & fit it
final_wf <- SVM_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=trainCsv)

SVM_predictions <- final_wf %>%
  predict(new_data = testCsv,
          type = "prob")

Sub4 <- SVM_predictions %>% 
  bind_cols(testCsv) %>% 
  select(id,.pred_1) %>%
  rename(Id= id, Action = .pred_1)


write_csv(Sub4, "SVMSubmission.csv")

##RF


trainCsv <- read_csv("train.csv")

testCsv <- read_csv("test.csv")

trainCsv <- trainCsv %>%
  mutate(ACTION = as.factor(ACTION))

#Create the recipe and bake it

rf_recipe <- recipe(ACTION ~ ., data=trainCsv) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  ###step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur <5% into an "other" value
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_smote(all_outcomes(), neighbors=5)

prep <- prep(rf_recipe)
baked <- bake(prep, new_data = NULL)
baked



#Set up the model
my_mod <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

## Create a workflow with model & recipe
rf_workflow <- workflow() %>%
  add_recipe(rf_recipe) %>%
  add_model(my_mod)

## Set up grid of tuning values
tuning_grid <- grid_regular(mtry(range = c(1,4)),
                            min_n(),
                            levels = 5) ## L^2 total tuning possibilities

## Set up K-fold CV
folds <- vfold_cv(trainCsv, v = 3, repeats=1)

## Run the CV
CV_results <- rf_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc, f_meas, sens, recall, spec,
                               precision, accuracy)) #Or leave metrics NULL

## Find best tuning parameters
collect_metrics(CV_results) %>% # Gathers metrics into DF
  filter(.metric=="roc_auc") %>%
  ggplot(data=., aes(x=mtry, y=min_n, color=factor(mtry))) +
  geom_line()

collect_metrics(CV_results)

CV_results
## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best("roc_auc")
bestTune

## Finalize the Workflow & fit it
final_wf <- rf_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=trainCsv)

rf_predictions <- final_wf %>%
  predict(new_data = testCsv,
          type = "prob")

Sub3 <- rf_predictions %>% 
  bind_cols(testCsv) %>% 
  select(id,.pred_1) %>%
  rename(Id= id, Action = .pred_1)


write_csv(Sub3, "RFSubmission.csv")


##Naive Bayes

#Read in dataset
trainCsv <- read_csv("train.csv")

testCsv <- read_csv("test.csv")

trainCsv <- trainCsv %>%
  mutate(ACTION = as.factor(ACTION))

#Create the recipe and bake it

nb_recipe <- recipe(ACTION ~ ., data=trainCsv) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  ###step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur <5% into an "other" value
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_pca(all_predictors(),threshold = .95) %>%
  step_smote(all_outcomes(),neighbors=5)#pca addition

prep <- prep(nb_recipe)
baked <- bake(prep, new_data = NULL)
baked


## nb model
nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes") # install discrim library for the naivebayes eng

nb_wf <- workflow() %>%
  add_recipe(nb_recipe) %>%
  add_model(nb_model)

## Tune smoothness and Laplace here


## Set up grid of tuning values
tuning_grid <- grid_regular(Laplace(),
                            smoothness(),
                            levels = 3) ## L^2 total tuning possibilities

## Set up K-fold CV
folds <- vfold_cv(trainCsv, v = 3, repeats=1)

## Run the CV
## smoothness 1.5, Laplace 0
CV_results <- nb_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc)) #Or leave metrics NULL

## Find best tuning parameters
collect_metrics(CV_results) %>% # Gathers metrics into DF
  filter(.metric=="roc_auc") %>%
  ggplot(data=., aes(x=Laplace, y=mean, color=factor(smoothness))) +
  geom_line()

collect_metrics(CV_results)


## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best("roc_auc")
bestTune

## Finalize the Workflow & fit it
final_wf <- nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=trainCsv)

nb_predictions <- final_wf %>%
  predict(new_data = testCsv,
          type = "prob")

Sub4 <- nb_predictions %>% 
  bind_cols(testCsv) %>% 
  select(id,.pred_1) %>%
  rename(Id= id, Action = .pred_1)

#Write it to a csv

write_csv(Sub4, "nbSubmission.csv")

## KNN
trainCsv <- read_csv("train.csv")

testCsv <- read_csv("test.csv")

trainCsv <- trainCsv %>%
  mutate(ACTION = as.factor(ACTION))

#Create the recipe and bake it

knn_recipe <- recipe(ACTION ~ ., data=trainCsv) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  ##step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur <5% into an "other" value
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_pca(all_predictors(),threshold = .9) %>%
  step_smote(all_outcomes(),neighbors=5) #pca addition

prep <- prep(knn_recipe)
baked <- bake(prep, new_data = NULL)
baked


## knn model
knn_model <- nearest_neighbor(neighbors=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kknn")

knn_wf <- workflow() %>%
  add_recipe(knn_recipe) %>%
  add_model(knn_model)

## Fit or Tune Model HERE


## Tune smoothness and Laplace here

## Set up grid of tuning values
tuning_grid <- grid_regular(neighbors(),
                            levels = 3) ## L^2 total tuning possibilities
tuning_grid

## Set up K-fold CV
folds <- vfold_cv(trainCsv, v = 3, repeats=1)

## Run the CV
CV_results <-knn_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc)) #Or leave metrics NULL

## Find best tuning parameters
collect_metrics(CV_results) %>% # Gathers metrics into DF
  filter(.metric=="roc_auc") %>%
  ggplot(data=., aes(x=neighbors, y=mean))+ #, color=factor(smoothness))) +
  geom_line()

collect_metrics(CV_results)


## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best("roc_auc")
bestTune

## Finalize the Workflow & fit it
final_wf <- knn_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=trainCsv)

knn_predictions <- final_wf %>%
  predict(new_data = testCsv,
          type = "prob")

Sub5 <- knn_predictions %>% 
  bind_cols(testCsv) %>% 
  select(id,.pred_1) %>%
  rename(Id= id, Action = .pred_1)


write_csv(Sub5, "knnSubmission.csv")
