import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import text2emotion as te
import emoji

# nltk.download('vader_lexicon')


with open('ex2_positive_review.txt', 'r', encoding='utf-8') as file1:
    positive_review_text = file1.read()

with open('ex2_negative_review.txt', 'r', encoding='utf-8') as file2:
    negative_review_text = file2.read()

# Initializing the SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

# Analysing the sentiment

positive_scores = sid.polarity_scores(positive_review_text)
negative_scores = sid.polarity_scores(negative_review_text)

# Show the results
print("Positive Review:")
print(positive_scores)
print("Negative Review:")
print(negative_scores) # Adding a few sentences of hate speech increased the neg level and decreased the neu one



# Getting emotions
# positive_emotions = te.get_emotion(positive_review_text)
# negative_emotions = te.get_emotion(negative_review_text)

# Show the results
# print("Emotions in the positive review:")
# print(positive_emotions)
# print("Emotions in the negative review:")
# print(negative_emotions)
