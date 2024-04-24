import random
from textblob import TextBlob

texts = [
    "I love this movie, it's fantastic!",
    "The weather is so beautiful today.",
    "I'm feeling great after my morning workout.",
    "This restaurant has amazing food!",
    "I had a wonderful time at the party last night.",
    "Today was a productive day at work.",
    "I'm so excited about my upcoming vacation!",
    "I can't wait to see my friends this weekend.",
    "The book I'm reading is captivating.",
    "I'm feeling inspired to start a new project.",
    "This new song is my favorite!",
    "I'm grateful for all the opportunities in my life.",
    "The sunset was breathtakingly beautiful.",
    "I'm enjoying learning new things every day.",
    "Spending time with family always makes me happy.",
    "I'm feeling optimistic about the future.",
    "The food at this restaurant is terrible.",
    "I'm disappointed with the service here.",
    "I'm frustrated with the traffic.",
    "The customer support was unhelpful and rude.",
    "I'm tired of dealing with this problem.",
    "I'm stressed about my upcoming exams.",
    "I'm feeling overwhelmed with work.",
    "I'm not looking forward to this meeting.",
    "The movie I watched last night was awful.",
    "I'm upset about the recent events.",
    "I'm feeling lonely.",
    "I'm annoyed with all the noise outside.",
    "The delay in the project is frustrating.",
    "I'm worried about the future.",
    "The news is depressing.",
    "I'm feeling anxious about the presentation.",
    "I'm disappointed with myself.",
    "I'm feeling down today.",
    "The loss of my pet has left me heartbroken.",
    "I'm struggling with my mental health.",
    "I'm feeling hopeless.",
    "I'm overwhelmed with grief.",
    "I'm exhausted from all the stress.",
    "I'm feeling isolated.",
    "I'm battling with depression.",
    "I'm feeling defeated.",
    "The constant negativity is draining.",
    "I'm struggling to find motivation.",
    "I'm feeling worthless.",
    "I'm stuck in a rut.",
    "I'm in a dark place right now."
]

for text in texts:
    sentiment = TextBlob(text).sentiment
    if sentiment.polarity > 0:
        print(f"Text: {text}\nSentiment: Positive\n")
    elif sentiment.polarity < 0:
        print(f"Text: {text}\nSentiment: Negative\n")
    else:
        print(f"Text: {text}\nSentiment: Neutral\n")
    print()