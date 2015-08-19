# Load librairies
library(randomForest)
library(foreach)
library(doSNOW)

# Make cluster
NB_NODES <- 2
registerDoSNOW(makeCluster(NB_NODES, type="SOCK"))

# Load train set and test set
train <- read.csv("train.csv", header = TRUE)
test <- read.csv("test.csv", header = TRUE)

train <- data.frame(Category=train$Category, PdDistrict=train$PdDistrict)
test <- data.frame(Id=test$Id, PdDistrict=test$PdDistrict)

# Apply the Random Forest Algorithm
rf <- foreach(ntree=rep(50, NB_NODES), .combine=combine, .packages='randomForest') %dopar% {
  randomForest(as.factor(Category) ~ PdDistrict,
               data=train, importance=TRUE, ntree=ntree)
}

# Make prediction using the test set
prediction <- predict(rf, test)
numeric_prediction <- as.numeric(prediction)

# Create solution
ncols <- nlevels(train$Category)
nrows <- nrow(test)
solution <- matrix(0, nrows, ncols)
colnames(solution) <- levels(train$Category)
for(x in test$Id)
{
  solution[x+1, numeric_prediction[x+1]] <- 1
}
solution <- cbind(format(test$Id, scientific = FALSE), solution)
colnames(solution)[1] <- "Id"

# Write solution
write.csv(solution, file = "solution.csv", row.names = FALSE, quote = FALSE)
