# -- equivalent code in R for which we're trying to replicate using the python lib

vul <- read.csv("glm/example/vul.csv", sep=" ")
model <- lm(data=vul, ln_death_risk ~ ln_events)
summary(model)

Call:
lm(formula = ln_death_risk ~ ln_events, data = vul)

# Residuals:
#     Min      1Q  Median      3Q     Max 
# -3.8631 -1.3077 -0.1099  0.6930  5.2887 

# Coefficients:
#             Estimate Std. Error t value Pr(>|t|)    
# (Intercept)  -1.6048     0.4220  -3.802 0.000212 ***
# ln_events     0.5174     0.1434   3.607 0.000429 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

# Residual standard error: 1.681 on 142 degrees of freedom
# Multiple R-squared:  0.08392,   Adjusted R-squared:  0.07747 
# F-statistic: 13.01 on 1 and 142 DF,  p-value: 0.0004286