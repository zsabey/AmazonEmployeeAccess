##Libraries
library(tidyverse)
library(tidymodels)
library(embed)
library(discrim)
library(kernlab)



trainCsv <- read_csv("train.csv")

testCsv <- read_csv("test.csv")

trainCsv <- trainCsv %>%
  mutate(ACTION = as.factor(ACTION))

#Create the recipe and bake it

SVM_recipe <- recipe(ACTION ~ ., data=trainCsv) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  #step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur <5% into an "other" value
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) #%>%
  #step_pca(all_predictors(),threshold = .95) #pca addition

prep <- prep(SVM_recipe)
baked <- bake(prep, new_data = NULL)
baked


## SVM model
svmRadial <- svm_rbf(rbf_sigma=tune(), cost=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kernlab")

SVM_wf <- workflow() %>%
  add_recipe(SVM_recipe) %>%
  add_model(SVM_model)

## Tune smoothness and Laplace here


## Set up grid of tuning values
tuning_grid <- grid_regular(rbf_sigma(),
                            cost(),
                            levels = 5) ## L^2 total tuning possibilities

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


