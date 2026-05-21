import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
data = pd.read_csv(url, sep='\t', names=['label', 'message'])

data['label'] = data['label'].map({'ham': 0, 'spam': 1})

X = data['message']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer = CountVectorizer(lowercase=True, stop_words='english')

X_train = vectorizer.fit_transform(X_train)  # learn + convert
X_test = vectorizer.transform(X_test)        # only convert

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n----- Model Evaluation -----")
print("Accuracy: {:.2f}%".format(accuracy * 100))

print("\n----- Test Your Own Message -----")
msg = input("Enter a message: ")

msg_transformed = vectorizer.transform([msg])
prediction = model.predict(msg_transformed)

print("\nPrediction Result:")
if prediction[0] == 1:
    print("Spam")
else:
    print("Not Spam")