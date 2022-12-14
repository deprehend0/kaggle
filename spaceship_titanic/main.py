import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("spaceship_titanic/data/train.csv")
data[["Deck", "Num", "Side"]] = data["Cabin"].str.split("/", expand=True)
# data[["Passenger_number", "Passenger_group"]] = data["PassengerId"].str.split("_", expand=True)
data = data.drop(columns=["Cabin"])
# print(data.info())
# print(data.head())

home_encoder = LabelEncoder()
home_encoder.fit(data["HomePlanet"])
data["HomePlanet"] = home_encoder.transform(data["HomePlanet"])

dest_encoder = LabelEncoder()
dest_encoder.fit(data["Destination"])
data["Destination"] = dest_encoder.transform(data["Destination"])

deck_encoder = LabelEncoder()
deck_encoder.fit(data["Deck"])
data["Deck"] = deck_encoder.transform(data["Deck"])

side_encoder = LabelEncoder()
side_encoder.fit(data["Side"])
data["Side"] = side_encoder.transform(data["Side"])

# group_encoder = LabelEncoder()
# group_encoder.fit(data["Passenger_group"])
# data["Passenger_group"] = group_encoder.transform(data["Passenger_group"])

data = data.dropna()

y = data["Transported"]
# X = data.drop(columns=["Transported", "Name", "PassengerId", "Passenger_number"])
X = data.drop(columns=["Transported", "Name", "PassengerId"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

ss_train = StandardScaler()
X_train = ss_train.fit_transform(X_train)
ss_test = StandardScaler()
X_test = ss_test.fit_transform(X_test)

clf = RandomForestClassifier()# max_depth=10, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc}")
