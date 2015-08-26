import pandas as pd
from sklearn import preprocessing
from sklearn.naive_bayes import BernoulliNB

DAYS_OF_WEEK = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
PD_DISTRICT = ['BAYVIEW', 'CENTRAL', 'INGLESIDE', 'MISSION', 'NORTHERN', 'PARK', 'RICHMOND', 'SOUTHERN', 'TARAVAL', 'TENDERLOIN']
HOURS_IN_DAY = [x for x in range(0, 24)]

def convert_data(data):
    # Convert categorical variables (weekdays, districts, and hours) to binarized arrays
    days = pd.get_dummies(data.DayOfWeek)
    district = pd.get_dummies(data.PdDistrict)
    hour = data.Dates.dt.hour
    hour = pd.get_dummies(hour)

    return pd.concat([hour, days, district], axis=1)


# Load data
train = pd.read_csv('train.csv', parse_dates=['Dates'])
test = pd.read_csv('test.csv', parse_dates=['Dates'])

# Convert crime labels to numerical labels
crime_label_encoder = preprocessing.LabelEncoder()
crime_label = crime_label_encoder.fit_transform(train.Category)

# Build train data
train_data = convert_data(train)
train_data['crime'] = crime_label

# Build test data
test_data = convert_data(test)

#Classification model based on the day, the district and the time
features = DAYS_OF_WEEK + PD_DISTRICT + HOURS_IN_DAY

model = BernoulliNB()
model.fit(train_data[features], train_data['crime'])
predicted = model.predict_proba(test_data[features])

result = pd.DataFrame(predicted, columns=crime_label_encoder.classes_)
result.to_csv('solution.csv', index=True, index_label='Id')
