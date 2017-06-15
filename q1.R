# Q1
library('caret')
library('glmnet')
library('glmnetUtils')
library('parallel')

q1_data <- read.table('uscrime.txt', header = TRUE)

q1_data$So <- factor(q1_data$So)

n <- nrow(q1_data)

set.seed(123)
inTrain <- sample(1:n, size = ceiling(n*0.9))

q1.train <- q1_data[inTrain,]
q1.test <- q1_data[-inTrain,]

# Hack
q1.test <- rbind(q1.train[1,], q1.test)
q1.test <- q1.test[-1,]

# Preprocessing
preProc1 <- preProcess(q1.train[,-c(2, 16)], method = 'corr')
q1.train.preproc1 <- predict(preProc1, q1.train)
preProc2 <- preProcess(q1.train.preproc1[,-15], method = c('BoxCox', 'center', 'scale'))
q1.train.preproc2 <- predict(preProc2, q1.train.preproc1)

# Center & normalize response
#train.resp.scaled <- scale(q1.train.preproc2$Crime, center = TRUE, scale = TRUE)
#q1.train.preproc2$Crime <- train.resp.scaled

# Function for making prediction on unprocessed data.
# q1_predict <- function(model, data.test) {
#     
#     # Pre processing test data
#     test.preproc <- predict(preProc2, predict(preProc1, data.test))
#     
#     pred <- predict(model, test.preproc[-15])
#     
#     pred.unscaled <- pred*attr(train.resp.scaled, 'scaled:scale') + 
#         attr(train.resp.scaled, 'scaled:center')
#     
#     return(pred.unscaled)
#     
# }

# Initial Model for Backwards Selection
model.init <- lm(Crime ~ ., data = q1.train.preproc2)

# Stepwise Selection
model.stepboth <- step(model.init, direction = 'both')

# http://statweb.stanford.edu/~tibs/sta305files/Rudyregularization.pdf

# Lasso
model.lasso <- cv.glmnet(Crime ~., data = q1.train.preproc2, 
                      family = 'gaussian', alpha = 1, type.measure = 'mse', 
                      keep = TRUE, nfolds = 10)

plot(model.lasso$lambda, model.lasso$cvm, xlab = 'Regularization Parameter (T)', ylab = 'Estimated MSE')
lambda <- model.lasso$lambda[which.min(model.lasso$cvm)]


pred.lasso <- predict(model.lasso, predict(preProc2, predict(preProc1, q1.test)))

rmse.lasso <- sqrt(ModelMetrics::mse(q1.test$Crime, pred.lasso))

show_coeffs <- function(model) {
    coeffs <- coef(model, s = "lambda.min")    
    df <- data.frame(name = dimnames(coeffs)[[1]][coeffs@i+1], 
                     coefficient = coeffs@x)
    return(df)
}

print(show_coeffs(model.lasso))

# alpha comparison plots
cv0 <- cv.glmnet(Crime ~., data = q1.train.preproc2, alpha = 0, foldid = model.lasso$foldid)
cv05 <- cv.glmnet(Crime ~., data = q1.train.preproc2, alpha = 0.5, foldid = model.lasso$foldid)
cv1 <- cv.glmnet(Crime ~., data = q1.train.preproc2, alpha = 1, foldid = model.lasso$foldid)

# Elastic Net
plot(log(cv1$lambda),cv1$cvm,pch=19,col="red",xlab="log(Lambda)",ylab=cv1$name)
points(log(cv05$lambda),cv05$cvm,pch=19,col="grey")
points(log(cv0$lambda),cv0$cvm,pch=19,col="blue")
legend("topleft",legend=c("alpha= 1","alpha= .5","alpha 0"),pch=19,col=c("red","grey","blue"))

lambda_range <- c(min(model.lasso$lambda), max(model.lasso$lambda))

lambdas <- seq(from = lambda_range[2], to = lambda_range[1], length.out = 100)

# Grid search
q1_grid_search <- function(alpha_range = c(0, 1), grid_n = 1001, foldid) {

    alphas <- seq(from = alpha_range[1], to = alpha_range[2],
                  length.out = grid_n)

    mse_vals <- rep(0, grid_n)

    q1_helper <- function(alpha, data, foldid) {

        test_model <- glmnetUtils::cv.glmnet(Crime ~., data = data,
                                             alpha = alpha, foldid = foldid)
        
        min_mse <- min(test_model$cvm)

        return(min_mse)
    }

    cl <- makePSOCKcluster(detectCores(logical = FALSE))
    clusterExport(cl = cl, varlist = list('alphas', 'q1.train.preproc2'),
                  envir = environment())

    mse_vals <- parSapply(cl = cl, X = alphas, FUN = q1_helper, 
                          data = q1.train.preproc2, foldid = foldid)

    stopCluster(cl)

    ans = list(min_ind = min(mse_vals), alpha = alphas[which.min(mse_vals)], 
               mses = mse_vals)
    
    return(ans)
}

alpha_search <- q1_grid_search(foldid = model.lasso$foldid)

model.elastic <- cv.glmnet(Crime ~., data = q1.train.preproc2, 
                           alpha = alpha_search$alpha, foldid = model.lasso$foldid)

plot(log(model.elastic$lambda), model.elastic$cvm)
points(log(model.lasso$lambda), model.lasso$cvm, col = 'red')

pred.elastic <- predict(model.elastic, predict(preProc2, predict(preProc1, q1.test)))

rmse.elastic <- sqrt(ModelMetrics::mse(q1.test$Crime, pred.elastic))

print(show_coeffs(model.elastic))
