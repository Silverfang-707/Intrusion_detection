import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv('data.csv')

X=data.drop('label',axis=1)
y=data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

ip_address = input('Enter an IP address: ')
prediction = model.predict([ip_address])

if prediction[0] == 1:
    print('The IP address is familiar.')
else:
    print('The IP address is suspicious.')