load('test/vulnerability.rdata')
# attach(vul)

# -- all teh features
# X<-vul[-c(1,7,8)] # --removing the 1st col (useless), 7th is the response and 8th is derived from it
y<-vul$death_risk

# -- only one feature
X<- vul$ln_events

X_mat <- as.matrix(cbind(1, X)) # --adding a column of 1s to the matrix and converting to numeric matrix
p<-dim(X)[2]
q<-dim(X_mat)[2] # including the beta0 as an estimator
n<-dim(X)[1]

# -- model: OLS and E(y), H0: beta_j =0
beta_hat <- solve(t(X_mat) %*% X_mat) %*% t(X_mat) %*% y
beta_hat

y_hat <- X_mat %*% beta_hat 
dim(y_hat)

# --metrics
RSS<- sum(
    (y-y_hat)^2
)
RSS

TSS<- sum(
    (y-mean(y))^2
)
TSS

R2 <- 1 - RSS/TSS;R2

adj_R2 <- 1 - (
    (1-R2)* (n-1)/
    (n-p-1) #p is number of predictors, but tehres p+1 predictors (q=p+1)
)
adj_R2

# --model
model <-lm(formula=y ~ as.matrix(X) )
summary(model)
predict(model)[1:5]
y_hat[1:5]

sigma_2 <-(
    sum((y-y_hat)^2)/
    n-p-1
)
t_test <- (
    beta_hat/
    sqrt(sigma_2 * diag(solve(t(X_mat) %*% X_mat)))
)
t_test

pval <- 2*(pt(-abs(t_test), df=n-p-1))
pval

write.csv(vul, 'vul.csv')
?write