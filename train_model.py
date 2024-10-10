import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset from the 'data' folder
df = pd.read_csv('data/drug200.csv')

# Print column names and inspect the first few rows
print(df.columns)
print(df.head())

# Drop the target column 'Drug' to form the feature set X
X = df.drop(columns=['Drug'])
y = df['Drug']

# List of categorical columns that need to be encoded
categorical_columns = ['Sex', 'BP', 'Cholesterol']  # Update based on your dataset

# Encoding categorical variables
label_encoder = LabelEncoder()
for col in categorical_columns:
    X[col] = label_encoder.fit_transform(X[col])

# Print the shape of X to verify the number of features
print(f"Shape of feature matrix X: {X.shape}")  # Should show (n_samples, 5)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a random forest classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Evaluate the model
print(f"Model accuracy: {clf.score(X_test, y_test)}")

# Save the model to a file
joblib.dump(clf, 'model.pkl')