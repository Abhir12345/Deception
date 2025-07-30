import os
import pandas as pd
import swifter
import spacy
from textstat import textstat
from empath import Empath
from empath import Empath
from textblob import TextBlob
import nltk
nltk.download('punkt')

try:
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

lex = Empath()

def extract_title_features(df):
    df = df.copy()

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

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
nlp_ner = spacy.load("en_core_web_sm")  # NER-enabled pipeline
lex = Empath()

def extract_text_features(df):
    df = df.copy()

    em_scores = df['text'].swifter.apply(lambda t: lex.analyze(t, normalize=True))
    df['text_analytic'] = em_scores.swifter.apply(lambda s: s.get('analytical', 0))
    df['text_cognitive'] = em_scores.swifter.apply(lambda s: s.get('cause', 0) + s.get('know', 0))
    df['text_negemo'] = em_scores.swifter.apply(lambda s: s.get('negative_emotion', 0))
    df['text_posemo'] = em_scores.swifter.apply(lambda s: s.get('positive_emotion', 0))
    df['text_authenticity'] = em_scores.swifter.apply(lambda s: s.get('honesty', 0) + s.get('self', 0))
    df['text_social'] = em_scores.swifter.apply(lambda s: s.get('social_media', 0) + s.get('friends', 0) + s.get('children', 0))
    df['text_perceptual'] = em_scores.swifter.apply(lambda s: s.get('see', 0) + s.get('hear', 0) + s.get('feel', 0))

    def pronoun_ratios(text):
        doc = nlp(text)
        tokens = [t for t in doc]
        total = len(tokens)
        i_count = sum(1 for t in doc if t.text.lower() == "i")
        we_count = sum(1 for t in doc if t.text.lower() == "we")
        they_count = sum(1 for t in doc if t.text.lower() == "they")
        return pd.Series({
            'text_ratio_I': i_count / max(1, total),
            'text_ratio_we': we_count / max(1, total),
            'text_ratio_they': they_count / max(1, total),
        })

    df[['text_ratio_I', 'text_ratio_we', 'text_ratio_they']] = df['text'].swifter.apply(pronoun_ratios)

    df['text_subjectivity'] = df['text'].swifter.apply(lambda t: TextBlob(t).sentiment.subjectivity)

    def count_named_entities(text):
        doc = nlp_ner(text)
        return sum(1 for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE'])

    df['text_verifiable_ents'] = df['text'].swifter.apply(count_named_entities)

    df['text_quote_count'] = df['text'].str.count(r'"')
    df['text_exclaim_count'] = df['text'].str.count('!')
    df['text_question_count'] = df['text'].str.count(r'\?')
    df['text_ellipsis_count'] = df['text'].str.count(r'\.\.\.')
    df['text_semicolon_count'] = df['text'].str.count(';')

    df['text_comma_period_ratio'] = df['text'].swifter.apply(lambda t: t.count(',') / max(1, t.count('.')))

    df['text_fk_grade'] = df['text'].swifter.apply(textstat.flesch_kincaid_grade)
    df['text_gunning_fog'] = df['text'].swifter.apply(textstat.gunning_fog)
    df['text_smog'] = df['text'].swifter.apply(textstat.smog_index)
    df['text_ari'] = df['text'].swifter.apply(textstat.automated_readability_index)

    return df

def process_and_save_text_batches(df, batch_size=3000, out_dir="processed_text_chunks"):
    os.makedirs(out_dir, exist_ok=True)
    total_batches = (len(df) + batch_size - 1) // batch_size

    for i in range(total_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(df))
        print(f"Processing batch {i+1}/{total_batches} — Rows {start} to {end}")
        chunk = df.iloc[start:end]
        processed_chunk = extract_text_features(chunk)

        chunk_path = os.path.join(out_dir, f"text_features_batch_{i+1}.csv")
        processed_chunk.to_csv(chunk_path, index=False)
        print(f"Saved: {chunk_path}")
