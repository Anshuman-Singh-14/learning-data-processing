






import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder

# Weather Dataset: Outlook, Temp, Humidity, Windy, Play
data = [
    ['Sunny', 'Hot', 'High', 'Weak', 'No'], ['Sunny', 'Hot', 'High', 'Strong', 'No'],
    ['Overcast', 'Hot', 'High', 'Weak', 'Yes'], ['Rain', 'Mild', 'High', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Weak', 'Yes'], ['Rain', 'Cool', 'Normal', 'Strong', 'No'],
    ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'], ['Sunny', 'Mild', 'High', 'Weak', 'No']
]
df = pd.DataFrame(data, columns=['Outlook', 'Temp', 'Humidity', 'Windy', 'Play'])

# Encode categorical data to numbers
le = LabelEncoder()
for col in df.columns: df[col] = le.fit_transform(df[col])

X, y = df.drop('Play', axis=1), df['Play']
clf = DecisionTreeClassifier(criterion='entropy').fit(X, y)
print("ID3 Decision Tree Rules:\n", export_text(clf, feature_names=list(X.columns)))



