# Load in the package
library(randomForest)

# Load train set and test set
train <- read.csv("train.csv", header = TRUE)
test <- read.csv("test.csv", header = TRUE)

str(train)
str(test)

# Apply the Random Forest Algorithm
forest <- randomForest(as.factor(Survived) ~ Sex,
                       data=train, importance=TRUE, ntree=1000)

# Make prediction using the test set
prediction <- predict(forest, test)

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
solution <- data.frame(PassengerId = test$PassengerId, Survived = prediction)

# Write solution away to a csv file with the name solution.csv
write.csv(solution, file = "solution.csv", row.names = FALSE)
