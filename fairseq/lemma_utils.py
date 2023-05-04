import re
from nltk.corpus import wordnet as wn
from spacy.lang.en import English
import spacy

to_spacy_pos = {
    "n": "NOUN",
    "a": "ADJ",
    "v": "VERB",
    "r": "ADV",
    "a.n": "NOUN",
    "n.v": "VERB",
    "n.a": "ADJ",
    "J": "ADJ",
    "V": "VERB",
    "R": "ADV",
    "N": "NOUN",
}

to_wordnet_pos = {
    "n": wn.NOUN,
    "a": wn.ADJ,
    "v": wn.VERB,
    "r": wn.ADV,
    "n.v": wn.VERB,
    "n.a": wn.ADJ,
    "J": wn.ADJ,
    "V": wn.VERB,
    "R": wn.ADV,
    "N": wn.NOUN,
}

def spacy_lemmatize(
    unlem,
    pos_tag,
    verbose=False,
    spacy_version=spacy.__version__,
):
    """
    Lemmatize sequence of words with Spacy lemmatizer.
    Args:
        unlem: sequence of unlemmatized words
        pos_tag: part-of-speech tags of words
            if str than this part-of-speech tag will be used with all words
        verbose: whether to print misc information
        spacy_version: it is necessary to save cache for each version of spacy
    Returns:
        sequence of lemmatized words
    """
    if spacy_version != "2.1.8":
        warnings.warn(f"Your results may depend on the version of spacy: {spacy_version}")

    pattern = re.compile(r"[#\[-]")
    lemmatizer = English.Defaults.create_lemmatizer()
    # lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES, LOOKUP)
    gen = unlem
    if verbose:
        gen = tqdm(unlem, desc="Vocabulary Lemmatization")

    if isinstance(pos_tag, str):
        pos_tag = [to_spacy_pos.get(pos_tag, "NOUN")] * len(unlem)
    else:
        pos_tag = [to_spacy_pos.get(pos_tag_, "NOUN") for pos_tag_ in pos_tag]

    new_vocab = [
        word if pattern.match(word) else lemmatizer(word, pos_tag_)[0]
        for word, pos_tag_ in zip(gen, pos_tag)
    ]
    return new_vocab





def nltk_lemmatize(
    unlem, pos_tag, verbose=False
):
    """
    Lemmatize sequence of words with nltk tokenizer.
    Args:
        unlem: sequence of unlemmatized words
        pos_tag: part-of-speech tags of words
            if str than this part-of-speech tag will be used with all words
        verbose: whether to print misc information
    Returns:
        sequence of lemmatized words
    """
    pattern = re.compile(r"[#\[-]")
    lemmatizer = WordNetLemmatizer()
    gen = unlem
    if verbose:
        gen = tqdm(unlem, desc="Vocabulary Lemmatization")

    # convert to appropriate pos abbreviation
    if isinstance(pos_tag, str):
        pos_tag = [to_wordnet_pos.get(pos_tag, "n")] * len(unlem)
    else:
        pos_tag = [to_wordnet_pos.get(pos_tag_, "n") for pos_tag_ in pos_tag]

    new_vocab = [
        word if pattern.match(word) else lemmatizer.lemmatize(word, pos_tag_)
        for word, pos_tag_ in zip(gen, pos_tag)
    ]
    return new_vocab