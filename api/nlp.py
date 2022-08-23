"""
This module defines routines for natural language processing as part of the
SDG Diagnostic Simulator API.
"""

# standard library
import os
import re
import json
from typing import Iterable, List, Dict
from collections import Counter
from heapq import nlargest
from importlib.resources import open_binary

# nlp
import pdfplumber
import spacy

# utils
from tqdm import tqdm


def allowed_file(filename: str) -> bool:
    """
    Checks if a filename is valid .pdf document.
    """
    allowed_extensions = {'pdf'}
    extension = filename.rsplit('.', 1)[-1].lower()
    return '.' in filename and extension in allowed_extensions


def load(filename: str, upload_folder: str, verbose: bool = False) -> str:
    """
    Loads a .pdf document, extracts text and saves it to the disk.
    """
    if verbose: print(f'loading file...')
    filepath_in = os.path.join(upload_folder, filename)
    filepath_out = os.path.join(upload_folder, re.sub('\.pdf$', '.txt', filename))
    texts = list()

    # reading the file page by page
    with pdfplumber.open(filepath_in) as pdf:
        for page in pdf.pages:
            try:
                text = page.extract_text()
                texts.append(text)
            except Exception as e:
                print(e)

    if verbose: print('writting...')
    text = ' '.join(texts)  # always a string but may be an empty one

    # saving the file on disk
    with open(filepath_out, 'w') as file:
        file.write(text)

    return text


def clean(text: str) -> str:
    text = re.sub(r'[.]{2,}', '', text)
    text = re.sub(r'[ ]+', ' ', text)
    # text = re.sub(r'\d{2} \d\.\d+', '\n', text)
    text = re.sub(r'cid:\d+', '\n', text)
    text = re.sub(r'\n', ' ', text)
    # text = re.sub(r'\d+.\d+\W+Goal \d+:', '\nGoal: ', text)
    return text


def rule3(text):
    doc = nlp(text)
    sent = []
    for token in doc:
        # look for prepositions
        if token.pos_=='ADP':
            phrase = ''
            # if its head word is a noun
            if token.head.pos_=='NOUN':
                # append noun and preposition to phrase
                phrase += token.head.text
                phrase += ' '+token.text
                # check the nodes to the right of the preposition
                for right_tok in token.rights:
                    # append if it is a noun or proper noun
                    if (right_tok.pos_ in ['NOUN','PROPN']):
                        phrase += ' '+right_tok.text
                if len(phrase)>2:
                    sent.append(phrase)
    return sent


def search(doc: spacy.tokens.doc.Doc, required: Iterable[str], optional: Iterable[str], stoppers: Iterable[str]) -> int:
    """
    Searches a document to determine if the required terms are present in the text.
    Likewise, at least one optional term and none of the stoppers must be present.
    """
    for i, sent in enumerate(doc.sents):
        st = [s.text.lower() for s in sent]

        # checking the conditions one by one
        c1 = all([w in st for w in required])
        c2 = any([m in st for m in optional])
        c3 = not any([s in st for s in stoppers])
        c4 = len(st) < 100
        if all([c1, c2, c3, c4]):
            return i


def entities(doc: spacy.tokens.doc.Doc, verbose: bool = False) -> List[int]:
    ents = list()
    sdg2queries = json.load(open_binary('api', 'queries.json'))
    if verbose: print('finding entries...')
    for sdg in tqdm(range(1, 18)):
        matches = search(doc, **sdg2queries[f'sdg_{sdg}'])
        ents.append(matches)
    if verbose: print(f'done:{ents}')
    return ents


def summarise(doc: spacy.tokens.doc.Doc) -> List[str]:
    keywords = list()
    pos_tags = {'PROPN', 'ADJ', 'NOUN', 'VERB'}
    for token in doc:
        if not token.is_stop and not token.is_punct and token.pos_ in pos_tags:
            keywords.append(token.text)
    # stopping early
    if len(keywords) == 0:
        return list()

    freq_word = Counter(keywords)
    max_freq = max(freq_word.values())
    freq_word = {k: v / max_freq for k, v in freq_word.items()}
    sent_strength = dict()
    for sent in doc.sents:
        for token in sent:
            if token.text in freq_word.keys():
                sent_strength[sent] = sent_strength.get(sent, 0) + freq_word[token.text]

    summarized_sentences = nlargest(3, sent_strength, key=sent_strength.get)
    final_sentences = [w.text for w in summarized_sentences]
    return final_sentences


def insight(text: str, nlp) -> Dict[str, List[str]]:
    doc = nlp(text)
    pos = entities(doc, verbose=True)
    sdg2insights = dict()
    sents = list()
    for i in tqdm(range(len(pos)-1)):
        label = f'Goal {i+1}'
        start = pos[i]
        if start is None:
            continue

        end = pos[i+1] if (pos[i+1] is not None and pos[i] < pos[i+1]) else start + 50
        if end is None:
            continue

        for idx, sent in enumerate(doc.sents):
            if start <= idx <= end:
                sents.append(sent.text)

        text = ' '.join(sents)  # sensitive to how texts are joined
        doc = nlp(text)
        sdg2insights[label] = summarise(doc)
    return sdg2insights
