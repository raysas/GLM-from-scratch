# ----------------------------------------------------------
# -- LINEAR MODELS

#1
load('data/vulnerability.rdata')
head(vul)

colnames(vul)
str(vul)
summary(vul)

colnames(vul)[-1]
#2
pairs(vul[colnames(vul)[-1]],col='purple')
# ln_fert show positive linear assoc, ln_events show a weak one

#3
# let y = ln_death_risk and x1 = ln_events 
# y = beta1 x1 + beta0 + epsilon <=> y= beta x + epsilon and eps belong to N (0, var I)

y=vul$ln_death_risk
x1<- vul$ln_events
X <- cbind(1,x1)

#model dimensions
dim(X)[2]

#OLS

beta_hat <- solve(t(X) %*% X) %*% t(X) %*% y
class(beta_hat)
beta_0 <- beta_hat[1]
beta_1 <- beta_hat[2]

model <- lm(data=vul,formula = y~x1)
model$coefficients
model_summary <- summary(model)
model_summary$coefficients

# -- significance test
# hypothesis: H0 is beta1=0; H1 beta1 !=0
# hypothesis: H0 is beta0=0; H1 
y_hat <- X %*% beta_hat 

n <- dim(X)[1];n
p<- dim(X)[2]
sigma2 <- sum((y-y_hat)^2)/(n-p)
sigma2

t <- beta_hat /
  sqrt(sigma2 * diag(solve(t(X) %*% X)))
t
model_summary$coefficients[,'t value']

pval <- 2* (1- pt(abs(t),df=n-p))
pval <- 2 * pt(-abs(t), df=n-p)
pval
model_summary$coefficients[,'Pr(>|t|)']
# with significance level = 0.05, bet0 and beta1 are significant i.e. 

#model comparison
M_2 <- lm (data=vul, formula=y~x1)
M_1 <- lm(data=vul, formula=y~1)
summary(M_1)

anova(M_1, M_2)
# f is high, p val sig, we reject H0
# H0: M1 is better; sigma2=sigma2*
# H1: M2 is better; sigma2 < sigma2*

#r2
rss <- sum(
  (y-y_hat)^2
)
tss <- sum(
  (y-mean(y))^2
)
r2 <- 1 - (rss/tss)
r2
model_summary$r.squared

adj_r2_joelle <- (
  (1-r2)/
    (n-p)
)
ajd_r2_rayane <-(1-
  ((n-1)*(1-r2)/
    (n-(p)))
)
ajd_r2_rayane
model_summary$adj.r.squared

# predict for ln_events=3.4

cbind(1,3.4) %*% beta_hat
class(cbind(1,3.4))
class(t(c(1,3.4)))

predict(model, newdata=data.frame(x1=3.4))

library(HH)
ci.plot(model)
# pred interval: we are 95% confident that the real y is within y_hat+- pred_error (this interval is the 2 red hashd lines)
# conf int: 95% beta belongs to beta_hat +- error => real_beta X (true regression line depicting real relationship)

#4
colnames(vul)
#stat model: y = beta= + ln_urb beta_1 + ln_events beta2 + ln_fert beta3 + hdi beta_4 + ln_pop beta5


Xmat <- cbind(1,vul[c(-1,-7,-8)])
Xmat
n<-dim(Xmat)[1]
q <- dim(Xmat)[2]

data_without_death <- vul[c(-1,-8)]
data_without_death
str(data_without_death)
full_model <- lm(data=data_without_death, formula=ln_death_risk~. )
full_model$rank

Xmat <- as.matrix(Xmat)
beta <- (solve(t(Xmat) %*%  Xmat)) %*% t(Xmat) %*% y
beta
full_model$coefficients

# -- validity:

# --- error terms
# - independent
# - identically dist (homoscedastic)
# - N(0, sigma2 I)
# - E(error)=0

plot(model, which=1) #independent
plot(model, which=2) #normality
plot(model, which=3) #homo
plot(full_model)


# --- X full rank

R <- cor(X[,-1])
eigen(R)
# no eigan=0 => full rank
ev <- eigen(R)$values
sort(ev)
ev
lambda1 <- ev[1]
lambda1
lambda5 <- ev[length(ev)]
ratio = lambda1/lambda5
ratio
# no multicolinearity

library(HH)
vif(full_model)

rsq <- summary(full_model)$r.squared
rsq
vif <- 1 /
  (1 - rsq)
vif


x1<- vul['ln_events']
x1
x2 <- vul$ln_urb
x3 <- vul$ln_fert
x4 <- vul$hdi
x5 <- vul$ln_pop

events_model <- lm(x1~x2+x3+x4+x5, data=X)

events_model <- lm(ln_events~ln_urb+ln_fert+ln_pop+hdi, data= vul)

rsq<-summary(events_model)$r.squared
rsq
vif_events <- 1/(1-rsq)
vif_events
vif(full_model)


#ex2

data(toxicity, package="robustbase")
str(toxicity)
summary(toxicity)
pairs(toxicity, col='blue')
model <- lm(data=toxicity, formula= toxicity ~.)
model
summary(model)
library(MASS)
best_model <- stepAIC(model)
best_model
summary(best_model)

x = data.frame(
  logKow=2
  ,pKa=0.8
  ,ELUMO=4.2
  ,Ecarb=17.6
  ,Emet=5
  ,RM=35
  ,IR=4.45
  ,Ts=38
  ,P=4
)
x
predict(model,x)




# -----------------------
load('../data/vulnerability.rdata')
str(vul)
pairs(vul[-1])
# ln_fert and hdi are linearly related
#hdi ln_pop or ln_death risk

# stat model: 
#let y=ln_death risk and X=(1 ln_events)
#model: y= X beta + epsilon
attach(vul)
y=ln_death_risk
x1=ln_events
X=cbind(1, x1)
dim(X)
n<- dim(X)[1]
q <- dim(X)[2]

beta <- solve( t(X) %*% X) %*% t(X) %*% y
dim(beta)
beta


lr_model <- lm(y~x1)
sum <- summary(lr_model)
sum$coefficients


y_hat <- X %*% beta
y_hat

#H0: beta1=0, x1 does not explain the variability in y



s2 <- sum((y-y_hat)^2)/
  (n-q)
s2
t_val <- beta /
  sqrt(s2 * diag(solve(t(X)%*%X)))
t_val

pval <- 2* pt(-abs(t_val),df=n-q)
pval
# model comparison:
null_model <- lm(y~1)
summary(null_model)

anova(null_model, lr_model)
#H0: Mq' better than Mq, sigma*=sigma
#H1: Mq better than Mq', in other words the explicative variable x1 helps explain teh variability in y


# --
rss <- sum((y-y_hat)^2)/n
tss <- sum((y-mean(y))^2)/n
rsq <- 1 - rss/tss
rsq
adj_rsq <- 1-((n-1)*(1-rsq))/
              (n-(q))

sum$r.squared            
sum$adj.r.squared
adj_rsq

predict(lr_model) #this is yhat
predict()

beta %*% cbind(1, 3.4)
c(1,3.4)
t(c(1,3.4)) %*% beta
beta_0 + 3.4 * beta_1

predict(lr_model, data.frame(x1=3.4))
sum

library(HH)
ci.plot(lr_model)        

full <- lm(data=vul[-1], formula=y~.)
summary(full)

# y = X beta + epsilon = ln_  ... + epsilon
# model validity

plot(full, which=1)
#line near 0 => independent
plot(full, which=2) #normal dist
plot(full, which=3) #hetersocedasticity
plot(full, which =4)
#some influential points

# now full rank
vif(full)

# try to get ln_urb vif
model_urb <- lm(data=vul, ln_urb ~ ln_events+ ln_fert + hdi + ln_pop + ln_death_risk)
1/(1-summary(model_urb)$r.squared)

# -- var selection

stepAIC(full)
anova (stepAIC(full), full)

R <- cor(vul[c(-1,-7)])
R
lambda <- eigen(R)$values
lambda
spect <- sort(lambda)
spect[length(spect)]/spect[length(1)]

# ---------------------------------------------------------
# -- LOGISTIC MODELS

library(lbreg)
data(Evans)

?Evans
#CDH is the response y, we want to be able to predict it or explain it by .....
str(Evans)

library(dplyr)
Evans <- mutate(Evans, CDH=factor(CDH),
                SMK=factor(SMK),
                ECG=factor(ECG),
                CAT=factor(CAT),
                HPT=factor(HPT))
str(Evans)

pairs(Evans)

plot(Evans$CDH, main='CDH dist')

for (i in names(Evans)){
 if(class(Evans[,i])=="factor"){
   barplot(Evans$CDH,Evans[,i])
 }
}  

barplot(table(Evans$CDH, Evans$CAT))
table(Evans$CDH, Evans$CAT)


# --
chisq.test(table(CDH,SMK), correct=F)
chisq.test(table(SMK, CDH), correct=F)

reg <- glm(CDH~SMK, family='binomial')
summary(reg)
#significant

OR <- exp(reg$coefficients[2])
OR
#OR greater than 1, there is a positive association between SMK and CDH and as ... this ~ risk, menaing that for everyone 1 unit increase in SMK there is nearly an increase in 2 CDH

library(Epi)
twoby2(SMK,CDH)
twoby2(CDH,SMK)

predict(reg, type='response')
y_hat <- predict(reg, type='response')>0.5
y_hat

full_model <- glm(data=Evans, formula=CDH~., family='binomial')
summary(full_model)

y_hat<-(predict(full_model, type='response') >0.5)
y_hat
full_model$fitted.values

table(CDH, y_hat)
library(ROCR)
prediction <- prediction(full$fitted.values, CDH)
names(prediction)

# ---------------------------------------------------------
# -- POISSON MODELS

library(ISwR)
data(eba1977)

str(eba1977)
summary(eba1977)
attach(eba1977)

hist(cases)
?eba1977

plot(city, cases)
boxplot(cases~age)

plot(pop,cases)
barplot(cases~pop)

model <- glm(cases~age+city, family='poisson')
summary(model)
head(eba1977)

model_1 <- glm(cases~age+city+pop, family = 'poisson')
model_2 <- glm(cases~age+city,offset=log(pop), family='poisson')
summary(model_2)
beta_age60<-1.5186
ir<- exp(beta_age60)

predict(model_2, newdata=data.frame(city='Kolding',age='60-64',pop=895),type='response')
model_2$coefficients

# ---------------------------------------------------------
# -- REGULARIZATION

load('data/HIV.rdata')
HIV_qual<-HIV
for (i in colnames(HIV)[-1]){
  HIV_qual[,i]<- as.factor(HIV[,i])
}
summary(HIV)
str(HIV)

for (i in colnames(HIV)[-1]){
  plot(HIV[,i], HIV$CVinc, main=i)
}
for (i in colnames(HIV)[-1]){
  plot(HIV_qual[,i], HIV$CVinc, main=i)
}
hist(HIV$CVinc)
cor_mat <- cor(HIV[-1])
diag(cor_mat)<-0
summary(cor_mat)
heatmap(cor_mat)

model_quant <- lm(data=HIV, formula=CVinc~.)
model_qual<- lm(data=HIV_qual, formula=CVinc~.)
summary(model_qual)
# let y=cvinc and X.. (qualt)
# y = beta X + epsilon where X= 
# { {SNP1_a SNP1_b SNP2_a SNP2_b ...}
# 
# }

length(model_quant$coefficients)
length(model_qual$coefficients)
#notice # of param almost doubles => varance increase, stat tests less powerful


summary_qual <- summary(model_qual)
summary_qual$coefficients[,"t value"]==summary_qual$coefficients[,3]
#just etsting

sort(p.adjust(summary_qual$coefficients[,4],method='BH'))[1:10]

summary_quant <- summary(model_quant)
sort(p.adjust(summary_quant$coefficients[,4], method='BH'))[1:10]

summary_qual$coefficients[,'Pr(>|t|)']



library(glmnet)
ridge <- glmnet(HIV[,-1],HIV[,1],alpha=0)
coef(ridge)
names(ridge)
plot(ridge)
dim(coef(ridge))\

cv_ridge<-cv.glmnet(as.matrix(HIV[,-1]), as.numeric(HIV[,1]), nfolds=5, alpha=0)
names(cv_ridge)
cv_ridge$lambda
plot(cv_ridge)
