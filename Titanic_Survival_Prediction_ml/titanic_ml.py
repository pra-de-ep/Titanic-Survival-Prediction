import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv(r"C:\Users\pradeep m\termal\train.csv")
print(df.head())  

print("\nMissing values in each column:")
print(df.isnull().sum())
df['Has_Cabin'] = df['Cabin'].notna().astype(int)  # 1 if Cabin is present, 0 if NaN
df.drop(columns=['Cabin'], inplace=True)  # Drop original Cabin colum

df['Age'].fillna(df['Age'].median(), inplace=True)
print("\nMissing values in each column:")
print(df.isnull().sum())
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
print("\nMissing values in each column:")
print(df.isnull().sum())
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)


# Select features you want to use for training
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Has_Cabin', 'Embarked_Q', 'Embarked_S']
X = df[features]
y = df['Survived']  # This is what we want to predict

# Split the data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Shape of training data:", X_train.shape)
print("Shape of test data:", X_test.shape)


# Initialize the model
model = LogisticRegression(max_iter=1000000000000000000)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of the Logistic Regression model:", accuracy*100)


# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))



