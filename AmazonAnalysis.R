##Libraries
library(tidyverse)
library(tidymodels)
library(embed)

trainCsv <- read_csv("train.csv")

testCsv <- read_csv("test.csv")

trainCsv <- trainCsv %>%
  mutate(ACTION = as.factor(ACTION))

##Recipe
my_recipe <- recipe(ACTION ~ ., data=trainCsv) %>%
step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .001) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))# combines categorical values that occur <5% into an "other" value
  #step_dummy(all_nominal_predictors())


penReg_recipe <- recipe(ACTION ~ ., data=trainCsv) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur <5% into an "other" value
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))
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
