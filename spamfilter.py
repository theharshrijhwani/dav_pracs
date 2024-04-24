from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

X = [
    "Hi there, this is just a friendly reminder",
    "Congratulations! You've won a free vacation",
    "Reminder: Your appointment is tomorrow",
    "Click here for exclusive offers",
    "Don't forget to pick up milk on your way home",
    "Urgent: Claim your prize now!",
    "Your Amazon order has been shipped",
    "Limited time offer: Get 50% off today",
    "Meeting canceled, see you next week",
    "You've been selected for a special discount"
]

y = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

X_processed = [(" ").join(TextBlob(text).words.lower()) for text in X]

vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X_processed)

classifier = MultinomialNB()
classifier.fit(X_vectorized, y)

new_text = ["Meet next week?"] 

nt_processed = [(" ").join(TextBlob(text).words.lower()) for text in new_text]

nt_vectorized = vectorizer.transform(nt_processed)

predicted_label = classifier.predict(nt_vectorized)[0]

if(predicted_label==1):
    print("SPAM")
else:
    print("NOT SPAM")