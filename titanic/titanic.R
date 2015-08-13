# Load in the package
library(randomForest)
library(rpart)

# Load train set and test set
train <- read.csv("train.csv", header = TRUE)
test <- read.csv("test.csv", header = TRUE)

str(train)
str(test)

# Passenger on row 62 and 830 do not have a value for embarkment.
# Since many passengers embarked at Southampton, we give them the value S.
train[train$Embarked=="",]
train$Embarked[c(62,830)] = "S"

# Predict missing passenger ages
predicted_age <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked,
                       data=train[!is.na(train$Age),], method="anova")
train$Age[is.na(train$Age)] <- predict(predicted_age, train[is.na(train$Age),])


# Apply the Random Forest Algorithm
forest <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age+ SibSp + Parch + Fare + Embarked,
                       data=train, importance=TRUE, ntree=1000)

# Make prediction using the test set
prediction <- predict(forest, test)

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
solution <- data.frame(PassengerId = test$PassengerId, Survived = prediction)

# Write solution away to a csv file with the name solution.csv
write.csv(solution, file = "solution.csv", row.names = FALSE)
