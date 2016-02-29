#Author : Abhishek Singh

#------------------------#------------------------#------------------------
#---------------------# Making Predictions with H20 functions #-------------------
#------------------------#------------------------#------------------------
#Calling the libraries
library(h2o)  ;  library(h2oEnsemble) 

#innitializing the set up
h2o.init(nthreads  =  -1, #All threads
         max_mem_size  =  "4G",
         min_mem_size  =  '2G') #Memory for H20 cloud
h2o.removeAll()  #Clean stat
setwd("~//Dropbox/AdvancedML_Project_/Data/workspace/")  #Chaging directory for result
#Instance : Successfully connected to http://127.0.0.1:54321/ 

#Importing the files  
df  <-  h2o.importFile(  
  path  =  normalizePath("~/Dropbox/AdvancedML_Project_/Data/workspace/trainMetrics.csv"))

#Treating the response variable : Factor
df[,  1]  <-  as.factor(df[,  1])

#Splitting into traning, test & control
splits  <-  h2o.splitFrame(
  df, #The dataframe
  .8, #80% train & 20% test 
  seed   =  10004  #randomization
)

#Training, test & validation sets
supertrain  <-  df  #The Full train
train  <-  h2o.assign(splits[[1]],  "train.hex")
test  <-  h2o.assign(splits[[2]],  "test.hex")




#------------------------#------------------------#------------------------
#---------------------# Making Predictions with Random Forests #-------------------
#------------------------#------------------------#------------------------
rf1  <-  h2o.randomForest(
                          training_frame  =  train,
                          validation_frame  =  test,
                          x  =  2:length(df) , #Predictors
                          y  =  1, #Response
                          ignore_const_cols  =  TRUE, #Remove Consts
                          model_id  =  "h2o_rf_whale", #Instance
                          balance_classes  =  TRUE, #Unbalanced data
                          max_depth  =  15,  #max depth
                          binomial_double_trees  =  TRUE,#Binary Class
                          mtries  =  75,  #mtry variables
                          stopping_metric  =  "AUC", #AUC stop
                          stopping_rounds  =  3, #Stopping criteria
                          stopping_tolerance  =  .0001, #Stopping threshold 
                          score_each_iteration  =  T,#Train & validation for each Tree
                          seed  =  100001)

#checking model performance
summary(rf1) #checking the performance
rf1@model$validation_metrics  #Validation performance
h2o.confusionMatrix(rf1) #Confusion Matrix
h2o.auc(rf1,  train  =  TRUE,  valid  =  TRUE) #AUC  0.9544400 (1st Interaction)
rf1@model$variable_importances[1:50, "variable"] #Important variables






#------------------------#------------------------#------------------------
#---------------------# Making Predictions with GBM  #-------------------
#------------------------#------------------------#------------------------

# Doing a grid search to get the best hyperparameters for GBM 
learn_rate  =  seq(.2,.6,.02) #All possible learning rates
hyper_params  <-  list(learn_rate  =  learn_rate) #Parameter tuning

model_grid  <-  h2o.grid("gbm", #classifier type
                         hyper_params  =  hyper_params, #The linear search 
                         training_frame  =  train,
                         validation_frame  =  test,
                         x  =  2:length(df), #Predictors
                         y  =  1, #Response
                         distribution  =  'bernoulli', #Binomial
                         ntrees  =  50,  #Adding trees
                         ignore_const_cols  =  TRUE, #Remove Consts
                         sample_rate  =  .8,  #Out of box error
                         col_sample_rate  =  .8, #Random Subsampling 
                         balance_classes  =  TRUE,#Binary class 
                         stopping_rounds  =  3,
                         stopping_metric  =  "AUC", #AUC being the evaluation metric
                         stopping_tolerance  =  .0001,
                         score_each_iteration  =  T)  #h20's seed

#fitting the best gbm
gbm1  <-  h2o.gbm(
                  training_frame  =  train,
                  validation_frame  =  test,
                  x  =  2:length(df), #Predictors
                  y  =  1, #Response
                  distribution  =  'bernoulli', #Classification type
                  ntrees  =  50,  #Adding trees
                  ignore_const_cols  =  TRUE, #Remove Consts
                  sample_rate  =  .8,  #Out of box error
                  col_sample_rate  =  .8, #Random Subsampling 
                  balance_classes  =  TRUE,#Binary class 
                  stopping_rounds  =  3,
                  stopping_metric  =  "AUC", #AUC being the evaluation metric
                  stopping_tolerance  =  .0001,
                  score_each_iteration  =  T, 
                  learn_rate  =  .40, #learn rate showing the best results
                  model_id  =  "h2o_gbm_whale",
                  seed  =  2000001)  #h20's seed

summary(gbm1)   #GBM performance
gbm1@model$validation_metrics   #Validation performance
h2o.confusionMatrix(gbm1)
h2o.auc(gbm1,  train  =  TRUE,  valid  =  TRUE) #AUC  
gbm1@model$variable_importances[1:40, "variable"] #Important variables





#------------------------#------------------------#------------------------
#----------# Making Predictions with GBM  (With Template Matching) #----------
#------------------------#------------------------#------------------------
#fitting the best gbm
gbm_tm  <-  h2o.gbm(
  training_frame  =  train,
  validation_frame  =  test,
  x  =  2:151, #Predictors
  y  =  1, #Response
  distribution  =  'bernoulli', #Classification type
  ntrees  =  50,  #Adding trees
  ignore_const_cols  =  TRUE, #Remove Consts
  sample_rate  =  .8,  #Out of box error
  col_sample_rate  =  .8, #Random Subsampling 
  balance_classes  =  TRUE,#Binary class 
  stopping_rounds  =  3,
  stopping_metric  =  "AUC", #AUC being the evaluation metric
  stopping_tolerance  =  .0001,
  score_each_iteration  =  T, 
  learn_rate  =  .40, 
  #  checkpoint  =  model_grid@model_ids[[1]], #The best model from tuning 
  model_id  =  "h2o_gbm_whale",
  seed  =  2000001)  #h20's seed

summary(gbm_tm)   #GBM performance
gbm_tm@model$validation_metrics   #Validation performance
h2o.confusionMatrix(gbm_tm)
h2o.auc(gbm_tm,  train  =  TRUE,  valid  =  TRUE) #AUC  
gbm_tm@model$variable_importances[1:40, "variable"] #Important variables







#------------------------#------------------------#------------------------
#----------# Making Predictions with GBM  (Without Template Matching) #----------
#------------------------#------------------------#------------------------
#fitting the best gbm
gbm_wtm  <-  h2o.gbm(
  training_frame  =  train,
  validation_frame  =  test,
  x  =  152:length(train), #Predictors
  y  =  1, #Response
  distribution  =  'bernoulli', #Classification type
  ntrees  =  50,  #Adding trees
  ignore_const_cols  =  TRUE, #Remove Consts
  sample_rate  =  .8,  #Out of box error
  col_sample_rate  =  .8, #Random Subsampling 
  balance_classes  =  TRUE,#Binary class 
  stopping_rounds  =  3,
  stopping_metric  =  "AUC", #AUC being the evaluation metric
  stopping_tolerance  =  .0001,
  score_each_iteration  =  T, 
  learn_rate  =  .56, 
  #  checkpoint  =  model_grid@model_ids[[1]], #The best model from tuning 
  model_id  =  "h2o_gbm_whale",
  seed  =  2000001)  #h20's seed

summary(gbm_wtm)   #GBM performance
gbm_wtm@model$validation_metrics   #Validation performance
h2o.confusionMatrix(gbm_wtm)
h2o.auc(gbm_wtm,  train  =  TRUE,  valid  =  TRUE) #AUC  
gbm_tm@model$variable_importances[1:40, "variable"] #Important variables





#------------------------#------------------------#------------------------
#---------------------# Clearing the environment #-------------------
#------------------------#------------------------#------------------------
h2o.shutdown(prompt  =  FALSE)
detach("package:h2oEnsemble",  unload  =  TRUE)
detach("package:h2o",  unload  =  TRUE)
rm(list  =  ls())  #Clearing environment
