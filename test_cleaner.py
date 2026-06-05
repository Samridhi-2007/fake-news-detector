from src.preprocessing.cleaner import clean_text

sample_news = """
BREAKING NEWS!!!

Scientists discovered a new vaccine.

Visit https://abc.com for details.
"""

print(clean_text(sample_news))