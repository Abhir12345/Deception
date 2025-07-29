import os
import pandas as pd
import swifter
import spacy
from textstat import textstat
from empath import Empath

# Load spacy model without NER/parser
try:
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

lex = Empath()

def extract_title_features(df):
    df = df.copy()

    # Tokenize using .split() — avoids nltk dependencies
    df['title_tokens'] = df['title'].swifter.apply(lambda t: t.split())
    df['title_word_count'] = df['title_tokens'].swifter.apply(len)

    df['title_allcaps_ratio'] = df['title_tokens'].swifter.apply(
        lambda toks: sum(1 for w in toks if w.isupper()) / max(1, len(toks))
    )

    df['title_exclaim_count'] = df['title'].str.count('!')
    df['title_question_count'] = df['title'].str.count(r'\?')
    df['title_quote_count'] = df['title'].str.count(r'"')

    def pronoun_freq(text):
        doc = nlp(text)
        return sum(1 for t in doc if t.tag_ in ('PRP', 'PRP$')) / max(1, len(doc))

    df['title_pronoun_ratio'] = df['title'].swifter.apply(pronoun_freq)
    df['title_fk_grade'] = df['title'].swifter.apply(textstat.flesch_kincaid_grade)
    df['title_gunning_fog'] = df['title'].swifter.apply(textstat.gunning_fog)

    em_scores = df['title'].swifter.apply(lambda t: lex.analyze(t, normalize=True))
    df['title_negemo'] = em_scores.swifter.apply(lambda s: s.get('negative_emotion', 0))
    df['title_posemo'] = em_scores.swifter.apply(lambda s: s.get('positive_emotion', 0))

    return df.drop(columns=['title_tokens'])

def process_and_save_in_batches(df, batch_size=5000, out_dir="processed_title_chunks"):
    os.makedirs(out_dir, exist_ok=True)
    total_batches = (len(df) + batch_size - 1) // batch_size

    for i in range(total_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(df))
        print(f"Processing batch {i+1}/{total_batches} — Rows {start} to {end}")
        chunk = df.iloc[start:end]
        processed_chunk = extract_title_features(chunk)

        chunk_path = os.path.join(out_dir, f"title_features_batch_{i+1}.csv")
        processed_chunk.to_csv(chunk_path, index=False)
        print(f"Saved: {chunk_path}")
