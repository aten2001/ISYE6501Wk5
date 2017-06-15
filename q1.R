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
preProc1 <- preProcess(q1.train[,-16], method = 'corr')
q1.train.preproc1 <- predict(preProc1, q1.train)
preProc2 <- preProcess(q1.train.preproc1[,-15], method = c('BoxCox', 'center', 'scale'))
q1.train.preproc2 <- predict(preProc2, q1.train.preproc1)

# Center & normalize response
train.resp.scaled <- scale(q1.train.preproc2$Crime, center = TRUE, scale = TRUE)
q1.train.preproc2$Crime <- train.resp.scaled

# Function for making prediction on unprocessed data.
q1_predict <- function(model, data.test) {
    
    # Pre processing test data
    test.preproc <- predict(preProc2, predict(preProc1, data.test))
    
    pred <- predict(model, test.preproc[-15])
    
    pred.unscaled <- pred*attr(train.resp.scaled, 'scaled:scale') + 
        attr(train.resp.scaled, 'scaled:center')
    
    return(pred.unscaled)
    
}

# Initial Model for Backwards Selection
model.init <- lm(Crime ~ ., data = q1.train.preproc2)

# Stepwise Selection
model.stepboth <- step(model.init, direction = 'both')

# http://statweb.stanford.edu/~tibs/sta305files/Rudyregularization.pdf

# Lasso
model.lasso <- cv.glmnet(Crime ~., data = q1.train.preproc2, 
                      family = 'gaussian', alpha = 1, type.measure = 'mse', 
                      keep = TRUE, nfolds = 10, standardize = FALSE)

plot(model.lasso$lambda, model.lasso$cvm, xlab = 'Regularization Parameter (T)', ylab = 'Estimated MSE')
lambda <- model.lasso$lambda[which.min(model.lasso$cvm)]

coeffs <- coef(model.lasso, s = "lambda.min")
data.frame(name = dimnames(coeffs)[[1]][-1][coeffs@i[-1]], coefficient = coeffs@x[-1])

pred.lasso <- q1_predict(model.lasso, q1.test)

rmse.lasso <- sqrt(ModelMetrics::mse(q1.test$Crime, pred.lasso))

# Elastic Net

# Grid search
q1_grid_search <- function(alpha_range = c(0, 1), grid_n = 1001) {
    folds <- createFolds(q1.train.preproc2$Crime, k = 10)
    
    alphas <- seq(from = alpha_range[1], to = alpha_range[2], length.out = grid_n)
    
    q1_helper <- function(alpha, data, folds) {

        temp_mse <- rep(0, length(folds))
        print(alpha)

        for (i in 1:length(folds)) {
            cv.train <- q1.train.preproc2[-folds[[i]],]
            cv.test <- q1.train.preproc2[folds[[i]],]

            test_model <- glmnetUtils::glmnet(Crime ~., data = cv.train, 
                                              alpha = alpha, standardize = FALSE)

            pred <- predict(test_model, cv.test)

            temp_mse[i] <- ModelMetrics::mse(cv.test$Crime, pred)
        }

        return(mean(temp_mse))
    }

    cl <- makePSOCKcluster(4)
    clusterExport(cl = cl, varlist = list('alphas', 'q1.train.preproc2'),
                  envir = environment())

    mse_vals <- parSapply(cl = cl, X = alphas, FUN = q1_helper,
                                data = q1.train.preproc2, folds = folds)


    stopCluster(cl)
    
    return(mse_vals)
}

mse_vals <- q1_grid_search()