import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1. Load the data
df = pd.read_csv('complaints.csv')

# 2. Setup the logic
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['text'])
y = df['category']

# 3. Train the model
model = MultinomialNB()
model.fit(X, y)

print("--- Customer Care AI is Ready ---")
# 4. Ask the user for input
user_input = input("Enter the customer's complaint: ")

# 5. Predict and show result
prediction = model.predict(tfidf.transform([user_input]))
print(f"Result: Please route this to the [{prediction[0]}] department.")