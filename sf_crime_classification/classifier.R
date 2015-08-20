# Load librairies
library(randomForest)
library(foreach)
library(doSNOW)

# Load train set and test set
train <- read.csv("train.csv", header = TRUE)
test <- read.csv("test.csv", header = TRUE)

# Extract data from train set
DATE_REGEXP = "([0-9]{4})-([0-9]{2})-[0-9]{2} ([0-9]{2}):[0-9]{2}:[0-9]{2}"
date_extractor <- function(date, idx) {
  gsub(DATE_REGEXP, paste("\\",as.character(idx), sep = ""), date)
}
PART_OF_DAY = c("MORNING", "AFTERNOON", "EVENING", "NIGHT")

year_extractor <- function(date) date_extractor(date, 1)
month_extractor <- function(date) date_extractor(date, 2)
hour_extractor <- function(date) date_extractor(date, 3)
part_of_day_extractor <- function(date) {
  hour <- as.numeric(hour_extractor(date))
  part_of_day <- hour
  part_of_day[hour>=7 & hour<13] <- PART_OF_DAY[1]
  part_of_day[hour>=13 & hour<17] <- PART_OF_DAY[2]
  part_of_day[hour>=17 & hour<21] <- PART_OF_DAY[3]
  part_of_day[hour>=21 | hour<7] <- PART_OF_DAY[4]
  as.factor(part_of_day)
}

train$Year <- year_extractor(train$Date)
test$Year <- year_extractor(test$Date)

train$Month <- month_extractor(train$Date)
test$Month <- month_extractor(test$Date)

train$PartOfDay <- part_of_day_extractor(train$Date)
test$PartOfDay <- part_of_day_extractor(test$Date)

# Start cluster
NB_NODES <- 2
cl <- makeCluster(NB_NODES, type="SOCK")
registerDoSNOW(cl)

# Apply the Random Forest Algorithm
NB_TREES <- 500
NB_TREES_PER_NODE <- 50
NB_CHUNKS <- NB_TREES / NB_TREES_PER_NODE
rf <- foreach(ntree=rep(NB_TREES_PER_NODE, NB_CHUNKS), .combine=combine, .packages='randomForest') %dopar% {
  randomForest(as.factor(Category) ~ PdDistrict + DayOfWeek + Year + PartOfDay,
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
