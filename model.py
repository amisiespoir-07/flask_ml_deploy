import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# load the csv file
df = pd.read_csv('iris.csv')

print(df.head)

#Select Independant and depandent variables
X = df[["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width",]]
y = df['Class']

#Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

#Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Initialize the model
classifier = RandomForestClassifier()

#fit the model
classifier.fit(X_train, y_train)

# Make pickle of our model
pickle.dump(classifier, open("model.pkl", "wb"))


