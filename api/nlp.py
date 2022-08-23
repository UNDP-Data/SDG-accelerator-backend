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
    """
    Cleans up the input text by removing repetitions, new line characters etc.

    Parameters
    ----------
    text: str
        input text to be processed.

    Returns
    -------
    text: str
        cleaned input text.
    """
    text = re.sub(r'[.]{2,}', '', text)
    text = re.sub(r'[ ]+', ' ', text)
    # text = re.sub(r'\d{2} \d\.\d+', '\n', text)
    text = re.sub(r'cid:\d+', '\n', text)
    text = re.sub(r'\n', ' ', text)
    # text = re.sub(r'\d+.\d+\W+Goal \d+:', '\nGoal: ', text)
    return text


def search_sentence(doc: spacy.tokens.doc.Doc, required: Iterable[str], optional: Iterable[str], stoppers: Iterable[str]) -> int:
    """
    Searches a document to determine if the required terms are present in the text.
    Likewise, at least one optional term and none of the stoppers must be present.

    doc: spacy.tokens.doc.Doc
        input document from a spacy model.
    required: Iterable[str]
        sequence of strings that all must appear in the document.
    optional: Iterable[str]
        sequence of strings of which at least one must appear in the document.
    stoppers: Iterable[str]
        sequence of strings of which none must appear in the document.

    Returns
    -------
    i: int
        integer index of the first matching sentence in the doc.
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


def find_indices(doc: spacy.tokens.doc.Doc, verbose: bool = False) -> List[int]:
    """
    Extracts indices of matchning sentences for each SDG Goal.

    doc: spacy.tokens.doc.Doc
        input document from a spacy model.
    verbose: bool, optional
        flag to indicate if to print any details.

    Returns
    -------
    indices: List[int]
        list of indices that indicating matching sentences, one per SDG.
    """
    indices = list()
    sdg2queries = json.load(open_binary('api', 'queries.json'))
    if verbose: print('finding entries...')
    for sdg in tqdm(range(1, 18)):
        index = search_sentence(doc, **sdg2queries[f'sdg_{sdg}'])
        indices.append(index)
    if verbose: print(f'Found: {indices}')
    return indices


def summarise(doc: spacy.tokens.doc.Doc) -> List[str]:
    """
    Summarises a doc by extracting sentences with most frequent keywords.

    doc: spacy.tokens.doc.Doc
        input document from a spacy model.

    Returns
    -------
    summaries: List[str]
        list of summary sentences from the doc.
    """
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

    key_sentences = nlargest(3, sent_strength, key=sent_strength.get)
    summaries = [sent.text for sent in key_sentences]
    return summaries


def get_insights(text: str, nlp) -> Dict[str, List[str]]:
    """
    Returns insights for a text by providing summaries per each matching SDG.

    text: str
        input text to be analysed.
    nlp: spacy
        an English language model from spacy.

    Returns
    -------
    sdg2insights: Dict[str, List[str]]
        mapping from sdg names to a list of summary sentences.
    """
    doc = nlp(text)
    indices = find_indices(doc, verbose=True)
    sdg2insights = dict()
    sentences = list()
    for i in tqdm(range(len(indices)-1)):
        label = f'Goal {i+1}'
        start = indices[i]
        if start is None:
            continue

        end = indices[i+1] if (indices[i+1] is not None and indices[i] < indices[i+1]) else start + 50
        if end is None:
            continue

        for idx, sent in enumerate(doc.sents):
            if start <= idx <= end:
                sentences.append(sent.text)

        text = ' '.join(sentences)  # sensitive to how texts are joined
        doc = nlp(text)
        sdg2insights[label] = summarise(doc)
    return sdg2insights
