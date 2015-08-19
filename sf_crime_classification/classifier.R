# Load librairies
library(randomForest)
library(foreach)
library(doSNOW)

# Load train set and test set
train <- read.csv("train.csv", header = TRUE)
test <- read.csv("test.csv", header = TRUE)

train <- data.frame(Category=train$Category, PdDistrict=train$PdDistrict)
test <- data.frame(Id=test$Id, PdDistrict=test$PdDistrict)

# Start cluster
NB_NODES <- 2
cl <- makeCluster(NB_NODES, type="SOCK")
registerDoSNOW(cl)

# Apply the Random Forest Algorithm
NB_TREES <- 500
NB_TREES_PER_NODE <- 50
NB_CHUNKS <- NB_TREES / NB_TREES_PER_NODE
rf <- foreach(ntree=rep(NB_TREES_PER_NODE, NB_CHUNKS), .combine=combine, .packages='randomForest') %dopar% {
  randomForest(as.factor(Category) ~ PdDistrict,
               data=train, importance=TRUE, ntree=ntree)
}

# Stop cluster
stopCluster(cl)

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
