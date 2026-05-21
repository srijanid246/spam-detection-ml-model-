import pickle

# Load model
with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Take input
msg = input("Enter message: ")

msg_transformed = vectorizer.transform([msg])
prediction = model.predict(msg_transformed)

if prediction[0] == 1:
    print("Spam 🚫")
else:
    print("Not Spam ✅")