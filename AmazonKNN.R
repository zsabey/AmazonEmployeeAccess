##Libraries
library(tidyverse)
library(tidymodels)


trainCsv <- read_csv("train.csv")

testCsv <- read_csv("test.csv")

trainCsv <- trainCsv %>%
  mutate(ACTION = as.factor(ACTION))

#Create the recipe and bake it

knn_recipe <- recipe(ACTION ~ ., data=trainCsv) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur <5% into an "other" value
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))

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
                            levels = 8) ## L^2 total tuning possibilities
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


