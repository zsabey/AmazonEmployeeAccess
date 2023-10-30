##Libraries
library(tidyverse)
library(tidymodels)
library(embed)
library(themis)

trainCsv <- read_csv("train.csv")

testCsv <- read_csv("test.csv")

trainCsv <- trainCsv %>%
  mutate(ACTION = as.factor(ACTION))

#Create the recipe and bake it

rf_recipe <- recipe(ACTION ~ ., data=trainCsv) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  #step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur <5% into an "other" value
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) #%>%
  #step_smote(all_outcomes(), neighbors=5)

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
                            levels = 10) ## L^2 total tuning possibilities

## Set up K-fold CV
folds <- vfold_cv(trainCsv, v = 3, repeats=1)

## Run the CV
CV_results <- rf_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc)) #Or leave metrics NULL

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


