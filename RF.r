#Input of Data and check of data types:
oneypd_tree <- read.csv("o:/MSFO/Моделирование - машинное обучение и GLM/СТ МЛ 2022/input_data_msb v2+.csv")
library(dplyr)
dplyr::glimpse(oneypd_tree)

#Transorming default criterions to dependent variable:
oneypd_tree<- dplyr::mutate(oneypd_tree,
CI_event = if_else(oneypd_tree$dpastdue_to._date > 90  |
oneypd_tree$restr_event == 1 |
oneypd_tree$writeoff_amount > 0 |
oneypd_tree$other_def_events == 1, 1,0))

CI_char=ifelse(oneypd_tree$CI_event==0,"No","Yes")
oneypd_tree=data.frame(oneypd_tree,CI_char)

dplyr::glimpse(oneypd_tree)

#Choosing of independent variables  
oneypd_tree_sel <- oneypd_tree %>%
dplyr::select("CI_char", "CI_event", "region", "interest_rate_type", "vintage_year", "orig_amount", "oked", "emp_number", "loan_balance",
"credit_type", "original_maturity_d","remaining_days_to_maturity", "loan_commitm", "dpastdue_180", "arrears_bal","business_type","fp_1",
"fp_2","fp_3","fp_4","fp_5","fp_6","fp_7","fp_8","fp_12","fp_16","fp_22","fp_26","fp_27","fp_31","fp_34","fp_41","fp_44","fp_45","fp_58")

dplyr::glimpse(oneypd_tree_sel)
#str(oneypd_tree_sel)

#Creating train and test samples 70%/30%
library(caret)
set.seed(234)
train_index <- caret::createDataPartition(oneypd_tree_sel$CI_event,
p = .7, list = FALSE)
train <- oneypd_tree_sel[ train_index,]
test <- oneypd_tree_sel[-train_index,]

dplyr::glimpse(train)
dplyr::glimpse(test)

#Training of Random Forest model on train sample
library(randomForest)
set.seed(123)
rf_oneypd <- randomForest(CI_char~.-CI_event,
data=oneypd_tree_sel[ train_index,], mtry=4, ntree=100,
importance=TRUE, na.action=na.omit)
rf_oneypd

imp <- importance(rf_oneypd)
varImpPlot(rf_oneypd)

#Validation of RF model on test sample

library(ROCR)
predict_test_orig <- as.matrix(
predict(rf_oneypd,newdata=oneypd_tree_sel[-train_index,],type="prob"))
predict_test <- as.matrix(predict_test_orig[,2])
oneypd_test <- oneypd_tree_sel[-train_index,"CI_char"]
actual_test <- as.matrix(ifelse(oneypd_test=="Yes",1,0))
pred_test<- ROCR::prediction(predict_test,actual_test)
perf_test<- ROCR::performance(pred_test, "tpr", "fpr")

plot(perf_test, main="ROC curve test", colorize=T)
abline(0,1, lty =8, col = "black")

auc_test<- ROCR::performance(pred_test, "auc")
auc_test

ks_test<- max(attr(perf_test,"y.values")[[1]]-
attr(perf_test,"x.values")[[1]])
print(ks_test)

library(optiRum)
gini_test<- optiRum::giniCoef(predict_test,actual_test)
gini_test

#Calibration of RF modeling results to PDs

pred_orig <- as.matrix(predict(rf_oneypd,newdata=oneypd_tree_sel,
type="prob"))
rf_pred <- as.matrix(pred_orig[,2])
rf_db_cal <- as.data.frame(cbind(oneypd_tree_sel$CI_event,
rf_pred))
colnames(rf_db_cal) <- c("CI", "pred")

str(rf_db_cal)

pd_model<- glm(CI~pred, family = binomial(link = "logit"),
data = rf_db_cal)
summary(pd_model)

#Grouping accounts in bands 

rf_db_cal$pd<- predict(pd_model, newdata = rf_db_cal,
type = "response")
library(smbinning)
score_cust<- smbinning.custom(rf_db_cal, y = "CI", x= "pred",
cuts= c(0.001,0.16,0.26,0.33,0.45))

rf_db_cal<- smbinning.gen(rf_db_cal, score_cust, chrname ="band")

write.table(rf_db_cal, file = "o:/MSFO/Моделирование - машинное обучение и GLM/СТ МЛ 2022/rf_db_calibr_msb 6 gr our cuts article w_o reg.csv", sep = ";", row.names = FALSE)

#Final validation of fitted PDs

rf_db_cal_plot<-rf_db_cal%>%
dplyr::group_by(band)%>%
dplyr::summarise(mean_CIr = round(mean(CI),4),
mean_pd = round(mean(pd),4))

rmse<-sqrt(mean((rf_db_cal_plot$mean_CIr - rf_db_cal_plot$mean_pd)^2))
rmse

plot(rf_db_cal_plot$band,rf_db_cal_plot$mean_CIr)
lines(rf_db_cal_plot$band,rf_db_cal_plot$mean_CIr, col="red")
lines(rf_db_cal_plot$band,rf_db_cal_plot$mean_pd, col="blue")
