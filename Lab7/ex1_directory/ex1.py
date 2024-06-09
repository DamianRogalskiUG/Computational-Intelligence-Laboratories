import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from wordcloud import WordCloud


nltk.download('stopwords')
nltk.download('wordnet')


# Load article
with open('article.txt', 'r', encoding='utf-8') as file:
    article_text = file.read()


# tokenization
tokens = word_tokenize(article_text)
sentences = sent_tokenize(article_text)
print(f"Number of words: {len(tokens)}")
print(f"Number of sentences: {len(sentences)}")


# Deleting stop words
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
print(f"Number of words after deleting stop words: {len(filtered_tokens)}")


# Deleting additional stop words
additional_stop_words = [".", ',', '“', '”']
filtered_tokens = [word for word in filtered_tokens if word.lower() not in additional_stop_words]
print("Liczba słów po usunięciu dodatkowych stop words:", len(filtered_tokens))


# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
print(f"Number of words after lemmatization {len(lemmatized_tokens)}")

# Stemming
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
print(f"Number of words after stemming: {len(stemmed_tokens)}")


# Words vectors

# Generating words frequency
freq_dist = FreqDist(filtered_tokens)

# Show 10 most common words
most_common_words = freq_dist.most_common(10)
words, counts = zip(*most_common_words)

# Show the plot
plt.figure(figsize=(10, 6))
plt.bar(words, counts)
plt.title('10 najczęściej występujących słów')
plt.xlabel('Słowa')
plt.ylabel('Liczba wystąpień')
plt.xticks(rotation=45)
plt.savefig("plots/ex1_plot.jpg")
plt.show()


# Generate tags cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(article_text)

# Show the tags cloud
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Tags cloud')
plt.savefig("plots/ex1_tags_cloud.jpg")
plt.show()