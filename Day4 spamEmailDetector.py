import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Load the dataset
print("Loading dataset...")
data = pd.read_csv('spambase.csv')

# 2. Split into Features (X) and Target (y)
# The last column 'spam' is the target (1 for spam, 0 for not spam)
X = data.iloc[:, :-1] # All columns except the last one
y = data.iloc[:, -1]  # Only the last column

# 3. Split into Training and Testing sets
# We use 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize the Model (Naive Bayes)
# MultinomialNB is a classic algorithm for text classification and spam detection
model = MultinomialNB()

# 5. Train the Model
print("Training model...")
model.fit(X_train, y_train)

# 6. Make Predictions
print("Making predictions...")
y_pred = model.predict(X_test)

# 7. Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("-" * 30)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("-" * 30)
print("\nConfusion Matrix:")
print(conf_matrix)
print("-" * 30)
