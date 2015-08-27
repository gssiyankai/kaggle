from types import NoneType
import pandas as pd
from sklearn import preprocessing
from sklearn.naive_bayes import BernoulliNB
import matplotlib.pyplot as plt

YEARS = [x for x in range(2003, 2016)]
MONTHS = [x for x in range(-1, -13, -1)]
HOURS = [x for x in range(0, 24)]
DAYS_OF_WEEK = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
PD_DISTRICT = ['BAYVIEW', 'CENTRAL', 'INGLESIDE', 'MISSION', 'NORTHERN', 'PARK', 'RICHMOND', 'SOUTHERN', 'TARAVAL', 'TENDERLOIN']
NX = 1000
NY = 1000
COORDS = ['B'+str(x) for x in range(0, NX*NY)]

def convert_data(data, limits):
    # Convert categorical variables (yeras, months, hours, weekdays, and districts) to binarized arrays
    year = data.Dates.dt.year
    year = pd.get_dummies(year)
    month = -data.Dates.dt.month
    month = pd.get_dummies(month)
    hour = data.Dates.dt.hour
    hour = pd.get_dummies(hour)
    day = pd.get_dummies(data.DayOfWeek)
    district = pd.get_dummies(data.PdDistrict)
    coord = get_coordinates_variable(data, limits)
    coord = pd.get_dummies(coord)
    return pd.concat([year, month, hour, day, district, coord], axis=1)

def get_coordinates_variable(data, limits):
    coordinates = [None] * len(data.X)
    x_min = limits[0]
    x_max = limits[1]
    x_step = (x_max - x_min) / NX
    y_min = limits[2]
    y_max = limits[3]
    y_step = (y_max - y_min) / NY
    for i in range(0, len(data.X)):
        x = data.X[i]
        y = data.Y[i]
        xn = (x - x_min) / x_step
        yn = (y - y_min) / y_step
        coordinates[i] = 'B' +  str(int(xn) + max(0, (int(yn) - 1)) * NX)
    return coordinates

def compute_coordinates_limits(train, test):
    return min(min(train.X), min(test.X)),\
           max(max(train.X), max(test.X)),\
           min(min(train.Y), min(test.Y)),\
           max(max(train.Y), max(test.Y))

# Load data
train = pd.read_csv('train.csv', parse_dates=['Dates'])
test = pd.read_csv('test.csv', parse_dates=['Dates'])

# Plot map
# plt.figure(1)
# plt.plot(train.X, train.Y, 'ro')
# plt.figure(2)
# plt.plot(test.X, test.Y, 'ro')
# plt.show()

# Convert crime labels to numerical labels
crime_label_encoder = preprocessing.LabelEncoder()
crime_label = crime_label_encoder.fit_transform(train.Category)

# Compute coordinates limits
coord_limits = compute_coordinates_limits(train, test)

# Build train data
train_data = convert_data(train, coord_limits)
train_data['crime'] = crime_label

# Build test data
test_data = convert_data(test, coord_limits)

#Classification model based on year, month, time, day and district
COORDS = [x for x in COORDS if train_data.keys().__contains__(x) and test_data.keys().__contains__(x)]
features = YEARS + MONTHS + HOURS + DAYS_OF_WEEK + PD_DISTRICT + COORDS

model = BernoulliNB()
model.fit(train_data[features], train_data['crime'])
predicted = model.predict_proba(test_data[features])

result = pd.DataFrame(predicted, columns=crime_label_encoder.classes_)
result.to_csv('solution.csv', index=True, index_label='Id')