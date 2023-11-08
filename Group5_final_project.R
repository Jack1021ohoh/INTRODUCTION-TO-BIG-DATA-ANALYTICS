library(dplyr)
library(data.table)
library(ISLR)
library(caret)
library(dplyr)
library(ggplot2)
library(epiDisplay)
library(pROC)

data<-fread("WA_Fn-UseC_-Telco-Customer-Churn.csv", stringsAsFactors = TRUE)

data[is.na(data) | data == "Inf"] <- NA
data[,1] = NULL
data<-na.omit(data)
data2<-na.omit(data)
data$Churn <- relevel(data$Churn, "Yes")
set.seed(500)
train_idx = sample(1:nrow(data), 0.7*nrow(data))
senior = data$SeniorCitizen
data[, 2] = NULL
train = data[train_idx,]
test = data[setdiff(1:nrow(data), train_idx),]

preprocess <- preProcess(train, method = c("center", "scale"))
train <- predict(preprocess, train)
test <- predict(preprocess, test)
train$SeniorCitizen = senior[train_idx]
test$SeniorCitizen = senior[-train_idx]

#單變量
#男女比
data1 <- data %>% 
  group_by(gender) %>% # Variable to be transformed
  count() %>% 
  ungroup() %>% 
  mutate(perc = `n` / sum(`n`)) %>% 
  arrange(perc) %>%
  mutate(labels = scales::percent(perc))

ggplot(data1, aes(x = "", y = perc, fill = gender)) +
  geom_col() +
  geom_label(aes(label = labels),
             position = position_stack(vjust = 0.5),
             show.legend = FALSE) +
  coord_polar(theta = "y")

#流失率
data1 <- data %>% 
  group_by(Churn) %>% # Variable to be transformed
  count() %>% 
  ungroup() %>% 
  mutate(perc = `n` / sum(`n`)) %>% 
  arrange(perc) %>%
  mutate(labels = scales::percent(perc))

ggplot(data1, aes(x = "", y = perc, fill = Churn)) +
  geom_col() +
  geom_label(aes(label = labels),
             position = position_stack(vjust = 0.5),
             show.legend = FALSE) +
  coord_polar(theta = "y")



#month to year-add a now column tenure_year
#show the tenure_year percentage
data1<-data %>%
  mutate(tenure_year = case_when(tenure <= 12 ~ "0-1 year",
                                 tenure > 12 & tenure <= 24 ~ "1-2 years",
                                 tenure > 24 & tenure <= 36 ~ "2-3 years",
                                 tenure > 36 & tenure <= 48 ~ "3-4 years",
                                 tenure > 48 & tenure <= 60 ~ "4-5 years",
                                 tenure > 60 & tenure <= 72 ~ "5-6 years"))
data1<-data1 %>%
  group_by(`tenure_year`)%>%
  summarise(num=n())%>%
  mutate(perc = paste0(sprintf("%4.1f", num / sum(num) * 100), "%"))%>%
  arrange(desc(num))
ggplot(data1, aes(x=tenure_year,y=num,color=tenure_year)) + 
  geom_bar(stat = "identity",fill="white")+
  geom_text(aes(label = perc)) +
  theme_minimal()

# Contract percentage
data1 <- data %>% 
  group_by(Contract) %>% # Variable to be transformed
  count() %>% 
  ungroup() %>% 
  mutate(perc = `n` / sum(`n`)) %>% 
  arrange(perc) %>%
  mutate(labels = scales::percent(perc))
ggplot(data1, aes(x = "", y = perc, fill = Contract)) +
  geom_col() +
  geom_label(aes(label = labels),
             position = position_stack(vjust = 0.5),
             show.legend = FALSE) +
  coord_polar(theta = "y")


#顧客合約形式 
data1 <- data %>% 
  group_by(Contract) %>% # Variable to be transformed
  filter(Churn=="Yes")%>%
  count() %>% 
  ungroup() %>% 
  mutate(perc = `n` / sum(`n`)) %>% 
  arrange(perc) %>%
  mutate(labels = scales::percent(perc))
#colnames(carr)<-c("Size","num","perc")

ggplot(data1, aes(x = Contract, y = perc, fill = Contract)) + 
  geom_bar(stat = "identity")+
  theme_minimal() #流失當中 mointh to month的流失最多


#senior citizen percentage 非次等公民中顧客流失率較低 約1/4 
data1 <- data2 %>% 
  group_by(SeniorCitizen) %>% # Variable to be transformed
  mutate(Churn = factor(Churn, labels = c("No", "Yes")),
         SeniorCitizen = factor(SeniorCitizen))
ggplot(data1, aes(x = SeniorCitizen, fill = Churn)) +
  geom_bar(position = "fill") +
  theme_classic()


#雙變量
chisq.test(data$Churn,data$Partner) # X-squared = 157.5, df = 1, p-value < 2.2e-16
chisq.test(data$Churn,data$Dependents)  # X-squared = 186.32, df = 1, p-value < 2.2e-16
chisq.test(data$Churn,data$gender)   # X-squared = 0.47545, df = 1, p-value = 0.4905 # maybe it is not related to gender
chisq.test(data$Churn,data$PhoneService)# X-squared = 0.87373, df = 1, p-value = 0.3499
chisq.test(data$Churn,data$StreamingTV)# X-squared = 372.46, df = 2, p-value < 2.2e-16
chisq.test(data$Churn,data$StreamingMovies)#X-squared = 374.27, df = 2, p-value < 2.2e-16
chisq.test(data$Churn,data$PaperlessBilling)#X-squared = 256.87, df = 1, p-value < 2.2e-16
chisq.test(data$Churn,data$PaymentMethod)#X-squared = 645.43, df = 3, p-value < 2.2e-16
chisq.test(data$Churn,data2$SeniorCitizen)

tenureData<-aov(tenure~Churn,data=data) 
summary.aov(tenureData)#p< <2e-16 *** 顯著

totalchargesChurn<-aov(TotalCharges~Churn,data=data) 
summary.aov(totalchargesChurn)#<2e-16 ***顯著

#dummy
#train$SeniorCitizen = senior[train_idx]
#train$PaperlessBilling<-ifelse(train$PaperlessBilling=="Yes",1,0)
# train$Churn<-ifelse(train$Churn=="Yes",1,0)
#train$Partner<-ifelse(train$Partner=="Yes",1,0)
#train$gender<-ifelse(train$gender=="Male",1,0)
#train$Dependents<-ifelse(train$Dependents=="Yes",1,0)
#train$PhoneService<-ifelse(train$PhoneService=="Yes",1,0)
#train_Churn = train$Churn
#train$Churn = NULL
#dmy <- dummyVars( ~ ., data = train)
#train <- data.frame(predict(dmy, newdata = train))
#train$Churn = train_Churn


#test$PaperlessBilling<-ifelse(test$PaperlessBilling=="Yes",1,0)
#test$SeniorCitizen = senior[-train_idx]
# test$Churn<-ifelse(test$Churn=="Yes",1,0)
#test$Partner<-ifelse(test$Partner=="Yes",1,0)
#test$gender<-ifelse(test$gender=="Male",1,0)
#test$Dependents<-ifelse(test$Dependents=="Yes",1,0)
#test$PhoneService<-ifelse(test$PhoneService=="Yes",1,0)
#test_Churn = test$Churn
#test$Churn = NULL
#test <- data.frame(predict(dmy, newdata = test))
#test$Churn = test_Churn

# upsample
train_upsample = upSample(train[, -19], train$Churn, yname = "Churn")

#downsample
train_downsample = downSample(train[, -19], train$Churn, yname = "Churn")

#forward selection with logistic regression
full <- glm(Churn~., data = train, family = "binomial")
null <- glm(Churn~1, data = train, family = "binomial")
step(null, 
     scope=list(lower=null, upper=full), 
     direction="forward")

f = formula(Churn ~ Contract + InternetService + tenure + PaymentMethod + 
              MultipleLines + OnlineSecurity + TotalCharges + TechSupport + 
              PaperlessBilling + StreamingTV + SeniorCitizen + Partner + 
              StreamingMovies + MonthlyCharges)

#random forest importance
rf = randomForest(Churn ~ ., data = train, ntree = 100, importance = TRUE)
importance(rf)

df <- data.frame(importance(rf))
df %>%
  arrange(desc(MeanDecreaseAccuracy))

varImpPlot(rf, sort=T, n.var= 19, main= "Historical area vs. currently present", pch = 20)

#10-fold cross validation
trControl <- trainControl(method = 'cv',
                          number = 10,
                          repeats =  1)

#logistic regression
logit.CV <- train(f, data = train,
                  method = 'glm',
                  trControl = trControl,
                  family = 'binomial' )

#adjusted OR
logistic.display(logit.CV$finalModel)

#training evaluation
y_train_pred <- predict(logit.CV, newdata = train)
y_train_probs <- predict(logit.CV, newdata = train, type = "prob")

y_train_pred <- as.factor(y_train_pred)
y_train_pred <- relevel(y_train_pred, "Yes")
confusionMatrix(y_train_pred, train$Churn)

roc_result <- roc(train$Churn, y_train_probs$Yes, print.auc=TRUE)
plot.roc(roc_result, print.thres = "best", print.thres.best.method = "youden", print.auc = T)

#testing evaluation
y_test_pred <- predict(logit.CV, newdata = test)
y_test_probs <- predict(logit.CV, newdata = test, type = "prob")

y_test_pred <- as.factor(y_test_pred)
y_test_pred <- relevel(y_test_pred, "Yes")
confusionMatrix(y_test_pred, test$Churn)

roc_result <- roc(test$Churn, y_test_probs$Yes, print.auc=TRUE)
plot.roc(roc_result, print.thres = "best", print.thres.best.method = "youden", print.auc = T)

#random forest

tunegrid <- expand.grid(.mtry = c(1:10))

rf.CV <- train(f , data = train,
               method = 'rf',
               metric = 'Accuracy',
               trControl = trControl,
               tuneGrid = tunegrid,
               ntree = 100,
               nodesize = 75)

rf.CV

#training evaluation
y_train_pred <- predict(rf.CV, newdata = train)
y_train_probs <- predict(rf.CV, newdata = train, type = "prob")

y_train_pred <- as.factor(y_train_pred)
y_train_pred <- relevel(y_train_pred, "Yes")
confusionMatrix(y_train_pred, train$Churn)

roc_result <- roc(train$Churn, y_train_probs$Yes, print.auc=TRUE)
plot.roc(roc_result, print.thres = "best", print.thres.best.method = "youden", print.auc = T)

#testing evaluation
y_test_pred <- predict(rf.CV, newdata = test)
y_test_probs <- predict(rf.CV, newdata = test, type = "prob")

y_test_pred <- as.factor(y_test_pred)
y_test_pred <- relevel(y_test_pred, "Yes")
confusionMatrix(y_test_pred, test$Churn)

roc_result <- roc(test$Churn, y_test_probs$Yes, print.auc=TRUE)
plot.roc(roc_result, print.thres = "best", print.thres.best.method = "youden", print.auc = T)

#xgboost
tuneGridXGB <- expand.grid(
  nrounds=c(200),
  max_depth = c(2, 3, 4),
  eta = c(0.02),
  gamma = c(1),
  colsample_bytree = c(0.7, 0.8, 0.9),
  subsample = c(0.80),
  min_child_weight = c(1))

xgboost.CV <- train(f, method = "xgbTree", data = train,
                    trControl=trControl, tuneGrid = tuneGridXGB)

xgboost.CV

#training evaluation
y_train_pred <- predict(xgboost.CV, newdata = train)
y_train_probs <- predict(xgboost.CV, newdata = train, type = "prob")

y_train_pred <- as.factor(y_train_pred)
y_train_pred <- relevel(y_train_pred, "Yes")
confusionMatrix(y_train_pred, train$Churn)

roc_result <- roc(train$Churn, y_train_probs$Yes, print.auc=TRUE)
plot.roc(roc_result, print.thres = "best", print.thres.best.method = "youden", print.auc = T)

#testing evaluation
y_test_pred <- predict(xgboost.CV, newdata = test)
y_test_probs <- predict(xgboost.CV, newdata = test, type = "prob")

y_test_pred <- as.factor(y_test_pred)
y_test_pred <- relevel(y_test_pred, "Yes")
confusionMatrix(y_test_pred, test$Churn)

roc_result <- roc(test$Churn, y_test_probs$Yes, print.auc=TRUE)
plot.roc(roc_result, print.thres = "best", print.thres.best.method = "youden", print.auc = T)

#neural network
tunegrid <- expand.grid(layer1 = c(4), layer2 = c(2), layer3 = c(0), 
                        momentum = 0.9, learning.rate = 0.01, 
                        dropout = 0.1, activation = "relu")

nn.CV <- train(f, data = train, method = "mxnet", 
               tuneGrid = tunegrid, num.round = 50)

nn.CV

#training evaluation
y_train_pred <- predict(nn.CV, newdata = train)
y_train_probs <- predict(nn.CV, newdata = train, type = "prob")

y_train_pred <- as.factor(y_train_pred)
y_train_pred <- relevel(y_train_pred, "Yes")
confusionMatrix(y_train_pred, train$Churn)

roc_result <- roc(train$Churn, y_train_probs$Yes, print.auc=TRUE)
plot.roc(roc_result, print.thres = "best", print.thres.best.method = "youden", print.auc = T)

#testing evaluation
y_test_pred <- predict(nn.CV, newdata = test)
y_test_probs <- predict(nn.CV, newdata = test, type = "prob")

y_test_pred <- as.factor(y_test_pred)
y_test_pred <- relevel(y_test_pred, "Yes")
confusionMatrix(y_test_pred, test$Churn)

roc_result <- roc(test$Churn, y_test_probs$Yes, print.auc=TRUE)
plot.roc(roc_result, print.thres = "best", print.thres.best.method = "youden", print.auc = T)

#importance
r = varImp(nn.CV)
plot(r)