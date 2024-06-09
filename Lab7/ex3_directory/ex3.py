import praw
from dotenv import load_dotenv
import os

# load environmental variables
load_dotenv()

# Pobierz zmienne środowiskowe
client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")
user_agent = os.getenv("USER_AGENT")


# Reddit object for my app
reddit = praw.Reddit(client_id=client_id,
                     client_secret=client_secret,
                     user_agent=user_agent)


# function to get and return 100 posts
def get_reddit_posts(topic, num_posts=100):
    posts = []
    subreddit = reddit.subreddit(topic)
    for post in subreddit.hot(limit=num_posts):
        posts.append(f'{post.title}\n{post.url}\n')
    return posts


title = "cooking"

# Scraping 100 posts on a given topic
reddit_posts = get_reddit_posts(title, num_posts=100)

# Save posts to the file
with open("reddit_AI_posts.txt", "w", encoding="utf-8") as file:
    file.writelines(reddit_posts)

print(f"Successfully scraped and saved a 100 posts on a topic: {title} from Reddit to the file 'reddit_posts.txt'.")
