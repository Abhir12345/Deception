import os
import pandas as pd
import nltk
import swifter
import spacy
from textstat import textstat
from empath import Empath

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
lex = Empath()

# Your existing feature extraction function
def extract_title_features(df):
    df = df.copy()

    # Tokenize
    df['title_tokens'] = df['title'].swifter.apply(nltk.word_tokenize)
    df['title_word_count'] = df['title_tokens'].swifter.apply(len)

    # ALLCAPS ratio
    df['title_allcaps_ratio'] = df['title_tokens'].swifter.apply(
        lambda toks: sum(1 for w in toks if w.isupper()) / max(1, len(toks))
    )

    # Punctuation counts
    df['title_exclaim_count'] = df['title'].str.count('!')
    df['title_question_count'] = df['title'].str.count(r'\?')
    df['title_quote_count'] = df['title'].str.count(r'"')

    # Pronoun usage
    def pronoun_freq(text):
        doc = nlp(text)
        return sum(1 for t in doc if t.tag_ in ('PRP', 'PRP$')) / max(1, len(doc))

    df['title_pronoun_ratio'] = df['title'].swifter.apply(pronoun_freq)

    # Readability
    df['title_fk_grade'] = df['title'].swifter.apply(textstat.flesch_kincaid_grade)
    df['title_gunning_fog'] = df['title'].swifter.apply(textstat.gunning_fog)

    # Empath scores
    em_scores = df['title'].swifter.apply(lambda t: lex.analyze(t, normalize=True))
    df['title_negemo'] = em_scores.swifter.apply(lambda s: s.get('negative_emotion', 0))
    df['title_posemo'] = em_scores.swifter.apply(lambda s: s.get('positive_emotion', 0))

    return df.drop(columns=['title_tokens'])
