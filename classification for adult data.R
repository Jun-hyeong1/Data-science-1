setwd('D:/Find in data/projectsbyJun_hyeong/Data Science no.1')
getwd()

#install.packages(c("dplyr", "ggplot2", "ISLR", "MASS", "glmnet",
#                   "randomForest", "gbm", "rpart", "boot"))         #미리 패키지 다운

library(tidyverse)
library(gridExtra)
library(ROCR)

library(ISLR)
library(MASS)
library(glmnet)
library(randomForest)
library(gbm)
library(rpart)
library(boot)
library(ggplot2)

#분류의 지표로 사용하는 이항편차를 계산하는 함수.

binomial_deviance <- function(y_obs, yhat){
  epsilon = 0.0001
  yhat = ifelse(yhat < epsilon, epsilon, yhat)
  yhat = ifelse(yhat > 1-epsilon, 1-epsilon, yhat)
  a = ifelse(y_obs==0, 0, y_obs * log(y_obs/yhat))
  b = ifelse(y_obs==1, 0, (1-y_obs) * log((1-y_obs)/(1-yhat)))
  return(2*sum(a + b))
}


#파일 불러오기

adult <- read.csv("adult.data", header = FALSE, strip.white = TRUE)
names(adult) <- c('age', 'workclass', 'fnlwgt', 'education',
                  'education_num', 'marital_status', 'occupation',
                  'relationship', 'race', 'sex',
                  'capital_gain', 'capital_loss',
                  'hours_per_week', 'native_country',
                  'wage')

glimpse(adult)

#Rows: 32,561
#Columns: 15

summary(adult)

adult$workclass <- as.factor(adult$workclass)
adult$education <- as.factor(adult$education)
adult$marital_status <- as.factor(adult$marital_status)
adult$occupation <- as.factor(adult$occupation)
adult$relationship <- as.factor(adult$relationship)
adult$race <- as.factor(adult$race)
adult$sex <- as.factor(adult$sex)
adult$native_country<- as.factor(adult$native_country)
adult$wage <- as.factor(adult$wage)
                                                    #문자형 변수를 범주형으로 변경. 
glimpse(adult)

levels(adult$wage)

##########################################범주형 변수를 분리하여 모델행렬화 하기. 
          #레벨 수 k 일 때 k-1개의 변수로 분리한다.

levels(adult$race)
adult$race[1:5]
levels(adult$sex)
adult$sex[1:5]                                      #변수의 범주확인

x <- model.matrix( ~ race + sex + age, adult)
glimpse(x)
colnames(x)                                         #model.matrix의 의미


x_orig <- adult %>% dplyr::select(sex, race, age)   #단순한 변수선택과 비교
View(x_orig)

x_mod <- model.matrix( ~ sex + race + age, adult)   #모형행렬
View(x_mod)


x <- model.matrix( ~ . - wage, adult)
view(x)
dim(x)                                              #범주를 변수화(모형행렬)


x<-as.data.frame(x) 
x<-cbind(x, adult$wage)
dim(x)



# 훈련, 검증, 테스트셋의 구분##############################################################
#원래 adult에 대해서 데이터 슬라이싱
set.seed(0810)
n <- nrow(adult)                                   #총 n개의 행
idx <- 1:n
training_idx <- sample(idx, n * .60)               #n개중에 0.6만큼 추출
idx <- setdiff(idx, training_idx)
validate_idx = sample(idx, n * .20)                #n개중에서 0.2만큼 추출
test_idx <- setdiff(idx, validate_idx)             #나머지 0.2 
length(training_idx)
length(validate_idx)
length(test_idx)
training <- adult[training_idx,]
validation <- adult[validate_idx,]
test <- adult[test_idx,]                           #각 데이터셋에 인덱싱 

#이제부터 분리한 데이터를 다룬다. 
# 시각화##############################################################
#그래프1
training %>%
  ggplot(aes(age, fill=wage)) +                    #x축은 나이, y축은 wage밀도 
  geom_density(alpha=.5)
ggsave("../../plots/8-3.png", width=5.5, height=4, units='in', dpi=600)


#그래프2
training %>%
  filter(race %in% c('Black', 'White')) %>%        #인종을 흑인 백인만 선택
  ggplot(aes(age, fill=wage)) +
  geom_density(alpha=.5) +
  ylim(0, 0.1) +
  facet_grid(race ~ sex, scales = 'free_y')        #인종 성별로 4등분     
ggsave("../../plots/8-4.png", width=5.5, height=4, units='in', dpi=600)

#그래프3
training %>%
  ggplot(aes(`education_num`, fill=wage)) +        #x축은 교육기간, y 축은 wage
  geom_bar()                                       #막대그래프
ggsave("../../plots/8-5.png", width=5.5, height=4, units='in', dpi=600)




# 로지스틱 회귀분석##############################################################
ad_glm_full <- glm(wage ~ ., data=training, family=binomial)    #wage는 2개의 범주를 갖는 반응변수 
#설명변수의 개수가 너무 많거나(모델이 복잡하다)샘플이 부족하거나. 오류발생
#일단 진행해보자. 
#glm함수는 범주형 변수에 대하여 자동으로 모형행렬을 생성한다. 

summary(ad_glm_full)        #모델요약


alias(ad_glm_full)


predict(ad_glm_full, newdata = adult[1:5,], type="response")           #간단하게 예측 시도


# 예측 정확도 지표
y_obs <- ifelse(validation$wage == ">50K", 1, 0)  #데이터셋의 0.2를 활용 

yhat_lm <- predict(ad_glm_full, newdata=validation, type='response')    #QQQQQQQQQQQQQQQQQQQ
#요인 native_country는 Holand-Netherlands개의 새로운 수준들을 가지고 있습니다.

#저 변수를 제거해 버리자. 
training<- subset(training,select= -native_country)
summary(training)

test<- subset(test,select= -native_country)
summary(test)

validation<- subset(validation,select= -native_country)
summary(validation)

#다시 로지스틱 모형 적합
ad_glm_full <- glm(wage ~ ., data=training, family=binomial)


y_obs <- ifelse(validation$wage == ">50K", 1, 0)

#다시 예측
yhat_lm <- predict(ad_glm_full, newdata=validation, type='response') 

#prediction from a rank-deficient fit may be misleading 출력.
#다중공선성 또는 부족한 샘플 숫자 
###########################################################################

#실제로 이 데이터는 education_num과 occupation Transport-moving이 
#다른 변수들의 조합으로 완벽하게 구성된다. 

alias(ad_glm_full)


library(gridExtra)

p1 <- ggplot(data.frame(y_obs, yhat_lm),
             aes(y_obs, yhat_lm, group=y_obs,
                 fill=factor(y_obs))) +
  geom_boxplot()
p2 <- ggplot(data.frame(y_obs, yhat_lm),
             aes(yhat_lm, fill=factor(y_obs))) +
  geom_density(alpha=.5)
grid.arrange(p1, p2, ncol=2)         #p1, p2 두개 동시에 출력 

g <- arrangeGrob(p1, p2, ncol=2)
ggsave("../../plots/8-6.png", g, width=5.5*1.5, height=4, units='in', dpi=600)


#예측 정확도 지표. 
binomial_deviance(y_obs, yhat_lm)               #앞서 만든 함수 (이항편차 계산)

library(ROCR)                                   #ROC곡선 그리는 패키지
pred_lm <- prediction(yhat_lm, y_obs)
perf_lm <- performance(pred_lm, measure = "tpr", x.measure = "fpr")
plot(perf_lm, col='black', main="ROC Curve for GLM")
abline(0,1)

performance(pred_lm, "auc")@y.values[[1]]    # AUC(곡선아래 면적.)   


dev.off()      #현재 그래픽 디바이스 종료. 


# 라쏘와 랜덤포레스트########################################

#glmnet 함수를 통한 라쏘 모형, 능형회귀, 변수선택

xx <- model.matrix(wage ~ .-1, adult)      #-1로 절편항 제거 
x <- xx[training_idx, ]
y <- ifelse(training$wage == ">50K", 1, 0)
dim(x)


#모형적합 
#디폴트로 라쏘모형 (alpha=1)
ad_glmnet_fit <- glmnet(x, y)             #glmnet함수는 모형행렬을 수동으로 입력해야한다.

plot(ad_glmnet_fit)  #상단의 숫자는 모형의 자유도(값이 0이 아닌 모수의 개수)

dev.off()

ad_glmnet_fit        #복잡해지는 순으로 모형이 정렬되어 있음.(lambda기준) 

coef(ad_glmnet_fit, s = c(.1713, .1295))   #특정 복잡도lambda값:c(.1713, .1295)에 해당하는 

#모델의 구조를 보여준다. 

#######################자동으로 모형을 골라보자. 

ad_cvfit <- cv.glmnet(x, y, family = "binomial")   #교차검증 함수
#로지스틱을 적합. 

plot(ad_cvfit)   #상단에 변수의 개수 (왼쪽이 가장 복잡)
#y축은 이항편차로 작을 수록 정확.


#어떤 모델을 쓰냐면, 둘중 하나인데 (그래프에서 점선의 위치)

log(ad_cvfit$lambda.min)    #교차검증 오차의 평균을 가장 작게하는 값(예측유리)
log(ad_cvfit$lambda.1se)    #위의 최소값으로부터 1표준편차내의 가장 간단한모델의 값(해석유리)

coef(ad_cvfit, s=ad_cvfit$lambda.1se)  #해당 모델의 구조를 확인
coef(ad_cvfit, s="lambda.1se")         #위와 같은 코드

length(which(coef(ad_cvfit, s="lambda.min")>0))     #해당 모델의 변수의 개수 
length(which(coef(ad_cvfit, s="lambda.1se")>0))


###################################################################값의 선택

set.seed(1607)
foldid <- sample(1:10, size=length(y), replace=TRUE) 
cv1 <- cv.glmnet(x, y, foldid=foldid, alpha=1, family='binomial')        #alpha=1 모델(라쏘)
cv.5 <- cv.glmnet(x, y, foldid=foldid, alpha=.5, family='binomial')      #alpha=0.5 모델(일레스틱넷)
cv0 <- cv.glmnet(x, y, foldid=foldid, alpha=0, family='binomial')        #alpha=0 모델(릿지)


par(mfrow=c(2,2))                            #세가지 모형에대한 이항편차 그래프
plot(cv1, main="Alpha=1.0")                  #릿지는 너무 복잡하며
plot(cv.5, main="Alpha=0.5")                 #4사분면의 그래프를 보면 모델 성능차이가 없다
plot(cv0, main="Alpha=0.0")
plot(log(cv1$lambda), cv1$cvm, pch=19, col="red",
     xlab="log(Lambda)", ylab=cv1$name, main="alpha=1.0")
points(log(cv.5$lambda), cv.5$cvm, pch=19, col="grey")
points(log(cv0$lambda), cv0$cvm, pch=19, col="blue")
legend("topleft", legend=c("alpha= 1", "alpha= .5", "alpha 0"),
       pch=19, col=c("red","grey","blue"))
dev.off()

# predict(모델, 람다값, 예측할x값, type response는 확률, link는 링크함수값)
predict(ad_cvfit, s="lambda.1se", newx = x[1:5,], type='response')

#glmnet모형의 평가

y_obs <- ifelse(validation$wage == ">50K", 1, 0)
yhat_glmnet <- predict(ad_cvfit, s="lambda.1se", newx=xx[validate_idx,], type='response')
yhat_glmnet <- yhat_glmnet[,1] # change to a vector from [n*1] matrix
binomial_deviance(y_obs, yhat_glmnet)
# [1] 4257.118
pred_glmnet <- prediction(yhat_glmnet, y_obs)
perf_glmnet <- performance(pred_glmnet, measure="tpr", x.measure="fpr")

performance(pred_glmnet, "auc")@y.values[[1]]

#앞서 실시한
#glm과 glmnet모형의 비교(변수26개의 glmnet과 101개의 glm이 비슷한 성능이다.)

plot(perf_lm, col='black', main="ROC Curve")
plot(perf_glmnet, col='blue', add=TRUE)
abline(0,1, col='gray')
legend('bottomright', inset=.1,
       legend=c("GLM", "glmnet"),
       col=c('black', 'blue'), lty=1, lwd=2)
dev.off()


############################################################ 나무모형
library(rpart)
cvr_tr <- rpart(wage ~ ., data = training)
cvr_tr


printcp(cvr_tr)
summary(cvr_tr)



png("../../plots/9-6.png", 5.5, 4, units='in', pointsize=9, res=600)
opar <- par(mfrow = c(1,1), xpd = NA)
plot(cvr_tr)
text(cvr_tr, use.n = TRUE)
par(opar)
dev.off()


yhat_tr <- predict(cvr_tr, validation)
yhat_tr <- yhat_tr[,">50K"]
binomial_deviance(y_obs, yhat_tr)
pred_tr <- prediction(yhat_tr, y_obs)
perf_tr <- performance(pred_tr, measure = "tpr", x.measure = "fpr")
performance(pred_tr, "auc")@y.values[[1]]

png("../../plots/9-7.png", 5.5, 4, units='in', pointsize=9, res=600)
plot(perf_lm, col='black', main="ROC Curve")
plot(perf_tr, col='blue', add=TRUE)
abline(0,1, col='gray')
legend('bottomright', inset=.1,
    legend = c("GLM", "Tree"),
    col=c('black', 'blue'), lty=1, lwd=2)
dev.off()


############################################################## 랜덤 포레스트 

set.seed(1607)
ad_rf <- randomForest(wage ~ ., training)
ad_rf

png("../../plots/9-8.png", 5.5, 4, units='in', pointsize=9, res=600)
plot(ad_rf)
dev.off()

tmp <- importance(ad_rf)
head(round(tmp[order(-tmp[,1]), 1, drop=FALSE], 2), n=10)

png("../../plots/9-9.png", 5.5, 4, units='in', pointsize=9, res=600)
varImpPlot(ad_rf)
dev.off()

predict(ad_rf, newdata = adult[1:5,])

predict(ad_rf, newdata = adult[1:5,], type="prob")


yhat_rf <- predict(ad_rf, newdata=validation, type='prob')[,'>50K']
binomial_deviance(y_obs, yhat_rf)
pred_rf <- prediction(yhat_rf, y_obs)
perf_rf <- performance(pred_rf, measure="tpr", x.measure="fpr")
performance(pred_tr, "auc")@y.values[[1]]

png("../../plots/9-10.png", 5.5, 4, units='in', pointsize=9, res=600)
plot(perf_lm, col='black', main="ROC Curve")
plot(perf_glmnet, add=TRUE, col='blue')
plot(perf_rf, add=TRUE, col='red')
abline(0,1, col='gray')
legend('bottomright', inset=.1,
       legend = c("GLM", "glmnet", "RF"),
       col=c('black', 'blue', 'red'), lty=1, lwd=2)
dev.off()


#######################################################예측확률값 자체의 비교
p1 <- data.frame(yhat_glmnet, yhat_rf) %>%
  ggplot(aes(yhat_glmnet, yhat_rf)) +
  geom_point(alpha=.5) +
  geom_abline() +
  geom_smooth()
p2 <- reshape2::melt(data.frame(yhat_glmnet, yhat_rf)) %>%
  ggplot(aes(value, fill=variable)) +
  geom_density(alpha=.5)
grid.arrange(p1, p2, ncol=2)
g <- arrangeGrob(p1, p2, ncol=2)
ggsave("../../plots/9-11.png", g, width=5.5*1.2, height=4*.8, units='in', dpi=600)


###############################################################부스팅 

set.seed(1607)
adult_gbm <- training %>% mutate(wage=ifelse(wage == ">50K", 1, 0))
ad_gbm <- gbm(wage ~ ., data=adult_gbm,
             distribution="bernoulli",
             n.trees=50000, cv.folds=3, verbose=TRUE)
(best_iter <- gbm.perf(ad_gbm, method="cv"))

ad_gbm2 <- gbm.more(ad_gbm, n.new.trees=10000)
(best_iter <- gbm.perf(ad_gbm2, method="cv"))


png("../../plots/9-12.png", 5.5, 4, units='in', pointsize=9, res=600)
(best_iter <- gbm.perf(ad_gbm2, method="cv"))
dev.off()


predict(ad_gbm, n.trees=best_iter, newdata=adult_gbm[1:5,], type='response')

yhat_gbm <- predict(ad_gbm, n.trees=best_iter, newdata=validation, type='response')
binomial_deviance(y_obs, yhat_gbm)
pred_gbm <- prediction(yhat_gbm, y_obs)
perf_gbm <- performance(pred_gbm, measure="tpr", x.measure="fpr")
performance(pred_gbm, "auc")@y.values[[1]]


png("../../plots/9-13.png", 5.5, 4, units='in', pointsize=9, res=600)
plot(perf_lm, col='black', main="ROC Curve")
plot(perf_glmnet, add=TRUE, col='blue')
plot(perf_rf, add=TRUE, col='red')
plot(perf_gbm, add=TRUE, col='cyan')
abline(0,1, col='gray')
legend('bottomright', inset=.1,
    legend=c("GLM", "glmnet", "RF", "GBM"),
    col=c('black', 'blue', 'red', 'cyan'), lty=1, lwd=2)
dev.off()



################################# 모형 비교, 최종 모형 선택, 일반화 성능 평가 


# 모형의 예측확률값의 분포 비교
# example(pairs) 에서 따옴
panel.cor <- function(x, y, digits = 2, prefix = "", cex.cor, ...){
  usr <- par("usr"); on.exit(par(usr))
  par(usr = c(0, 1, 0, 1))
  r <- abs(cor(x, y))
  txt <- format(c(r, 0.123456789), digits = digits)[1]
  txt <- paste0(prefix, txt)
  if(missing(cex.cor)) cex.cor <- 0.8/strwidth(txt)
  text(0.5, 0.5, txt, cex = cex.cor * r)
}

png("../../plots/9-14.png", 5.5, 4, units='in', pointsize=9, res=600)
pairs(data.frame(y_obs=y_obs,
                yhat_lm=yhat_lm,
                yhat_glmnet=c(yhat_glmnet),
              yhat_rf=yhat_rf,
                yhat_gbm=yhat_gbm),
    lower.panel=function(x,y){ points(x,y); abline(0, 1, col='red')},
    upper.panel = panel.cor)
dev.off()


# 테스트셋을 이용한 일반화능력 계산
y_obs_test <- ifelse(test$wage == ">50K", 1, 0)
yhat_gbm_test <- predict(ad_gbm, n.trees=best_iter, newdata=test, type='response')
binomial_deviance(y_obs_test, yhat_gbm_test)
pred_gbm_test <- prediction(yhat_gbm_test, y_obs_test)
performance(pred_gbm_test, "auc")@y.values[[1]]

######################캐럿 (caret) 패키지
install.packages("caret", dependencies = c("Depends", "Suggests"))



# This is for the earlier ROC curve example. ---
{
  png("../../plots/8-1.png", 5.5*1.2, 4*.8, units='in', pointsize=9, res=600)
  opar <- par(mfrow=c(1,2))
  plot(perf_lm, col='black', main="ROC Curve")
  plot(perf_tr, col='blue', add=TRUE)
  abline(0,1, col='gray')
  legend('bottomright', inset=.1,
      legend = c("GLM", "Tree"),
      col=c('black', 'blue'), lty=1, lwd=2)
  plot(perf_lm, col='black', main="ROC Curve")
  plot(perf_glmnet, add=TRUE, col='blue')
  plot(perf_rf, add=TRUE, col='red')
  plot(perf_gbm, add=TRUE, col='cyan')
  abline(0,1, col='gray')
  legend('bottomright', inset=.1,
      legend=c("GLM", "glmnet", "RF", "GBM"),
      col=c('black', 'blue', 'red', 'cyan'), lty=1, lwd=2)
  par(opar)
  dev.off()
}
