import pandas as pd
from sklearn import preprocessing
from sklearn.naive_bayes import BernoulliNB

YEARS = [x for x in range(2003, 2016)]
MONTHS = [x for x in range(-1, -13, -1)]
HOURS = [x for x in range(0, 24)]
DAYS_OF_WEEK = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
PD_DISTRICT = ['BAYVIEW', 'CENTRAL', 'INGLESIDE', 'MISSION', 'NORTHERN', 'PARK', 'RICHMOND', 'SOUTHERN', 'TARAVAL', 'TENDERLOIN']

def convert_data(data):
    # Convert categorical variables (yeras, months, hours, weekdays, and districts) to binarized arrays
    year = data.Dates.dt.year
	year = pd.getDummies(year)
	month = -data.Dates.dt.month
	month = pd.get_dummies(month)
	hour = data.Dates.dt.hour
    hour = pd.get_dummies(hour)
	day = pd.get_dummies(data.DayOfWeek)
    district = pd.get_dummies(data.PdDistrict)
    return pd.concat([year, month, hour, day, district], axis=1)

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

#Classification model based on the year, month, time, day and the district
features = YEARS + MONTHS + HOURS + DAYS_OF_WEEK + PD_DISTRICT

model = BernoulliNB()
model.fit(train_data[features], train_data['crime'])
predicted = model.predict_proba(test_data[features])

result = pd.DataFrame(predicted, columns=crime_label_encoder.classes_)
result.to_csv('solution.csv', index=True, index_label='Id')
