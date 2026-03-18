import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# 1. Load the dataset
# Make sure 'spam.csv' is in the same folder as this script!
df = pd.read_csv('spam.csv', encoding='latin-1')

# 2. Clean the data (Keeping only the Message and the Label)
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# 3. Convert labels to numbers (ham = 0, spam = 1)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# 4. Split data into Training (80%) and Testing (20%)
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# 5. Convert text into numbers (Vectorization)
cv = CountVectorizer()
X_train_vec = cv.fit_transform(X_train)
X_test_vec = cv.transform(X_test)

# 6. Train the Model (Naive Bayes)
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 7. Test the Accuracy
predictions = model.predict(X_test_vec)
print(f"Model Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")

# 8. Try it yourself!
sample_msg = ["Congratulations! You won a $1000 gift card. Click here to claim."]
sample_vec = cv.transform(sample_msg)
result = model.predict(sample_vec)

if result[0] == 1:
    print("Result: This is a SPAM message!")
else:
    print("Result: This is a REAL message.")