"""A simple loader for your CatBoost Classifier model."""
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from json import load as json_load
import re
from typing import Dict, List, Union

from catboost import CatBoostClassifier
from nltk.corpus import stopwords
from numpy import empty, zeros
from pandarallel import pandarallel
from pandas import concat, DataFrame, read_csv, Series
from pymorphy2 import MorphAnalyzer


def safe_json_loader(path: str) -> Union[Dict, List]:
    """Load your dict from JSON.

    Args:
        path: A path to your dict.

    Returns:
        A dict or a list from your json.
    """
    with open(path) as model_file:
        json = json_load(model_file)
        model_file.close()
    return json


def clean_string(string: str) -> str:
    """Clean the junk from the string.

    Args:
        string: a string to clean.

    Returns:
        cleaned string.
    """
    string = re.sub(r'[^0-9a-zA-Zа-яА-ЯёЁ\.,\(\)]+', ' ', string)
    string = re.sub(r'([^\w ])', r' \1', string)
    string = re.sub(r'([^ \w])', r'\1', string)
    string = re.sub(r' +', r' ', string)
    string = re.sub(r'^ ', r'', string)
    string = re.sub(r'[\W_]+', ' ', string)
    return string.lower()


def find_from_dict(searcher: Dict[str, str], string: str) -> List[int]:
    """Find the presence of regex patterns in your string.

    Args:
        searcher: OrderedDict name of regex -> regex to find,
        string: string to search.

    Returns:
        list of ints: 0 if no regexp, 1 else.
    """
    occurrences = []
    for regexp in searcher.values():
        occurrences.append(int(bool(re.search(regexp, string))))
    return occurrences


def occurrences_to_dataframe(
    occurrences: List[int],
    regexes: Dict[str, str],
) -> DataFrame:
    """Helper function to turn your occurrences list to a DataFrame.

    Args:
        occurrences: list of regex patterns presence,
        regexes: OrderedDict name of regex -> regex to find.

    Returns:
        DataFrame: name of regex -> its occurrence.
    """
    return DataFrame((regex for regex in occurrences), columns=regexes.keys())


def replace_from_dict(replacer: Dict[str, str], string: str) -> str:
    """Replace all the regex patterns in your string.

    Args:
        replacer: name of regex -> regex to replace,
        string: string to clean.

    Returns:
        string with replaced regexes.
    """
    for cyrillic, symbol in replacer.items():
        string = re.sub(cyrillic, str(symbol), string)
    return string


@lru_cache(maxsize=100000)
def lemmatizer(word: str, morph) -> str:
    """Get the normal form of the passed word.

    Uses lru_cache to speed up the computations.

    Args:
        word: word to find the normal form,
        morph: MorphAnalyzer,

    Returns:
        normal form of the word.
    """
    return morph.parse(word)[0].normal_form


def process_text(text: str, stopwords_set: set, morph) -> str:
    """Clean and lemmatize your text.

    Args:
        text: text to lemmatize,
        stopwords_set: your stopwords to get rid of,
        morph: MorphAnalyzer, a pymorphy2 class.

    Returns:
        cleaned and lemmatized text without stop words.
    """
    text = clean_string(str(text)).split()
    text = [
        word for word in text
        if word not in stopwords_set
    ]
    return ' '.join(map(lambda word: lemmatizer(word, morph), text))


@dataclass
class CatBoost:
    """CatBoostClassifier wrapper."""

    dataset: DataFrame
    regexes_path: str
    punctuation_path: str
    catboost_path: str
    stopwords_path: str
    predictions: DataFrame = None
    stopwords: set = None
    regexes: Dict[str, str] = None
    model: CatBoostClassifier = None
    punctuation: Dict[str, str] = None

    def load_regexes_and_punctuation(self) -> None:
        """Load your regexes and punctuation dictionaries."""
        self.regexes = safe_json_loader(self.regexes_path)
        self.punctuation = safe_json_loader(self.punctuation_path)

    def get_stopwords(self) -> None:
        """Get your stopwords set."""
        additional_stopwords = safe_json_loader(self.stopwords_path)
        self.stopwords = set(additional_stopwords)
        self.stopwords.update(stopwords.words('russian'))
        self.stopwords.update(stopwords.words('english'))

    def prepare_dataset(self) -> None:
        """Prepare your dataset.

        Clean your dataset, replace punctuation, replace regexes, lemmatize,
        concatenate description and title, remove waste columns.
        """
        morph = MorphAnalyzer()

        regexp_occurrences = self.dataset.description.parallel_apply(
            lambda string: find_from_dict(self.regexes, string)
        )
        regexp_occurrences = occurrences_to_dataframe(regexp_occurrences, self.regexes)
        self.dataset = concat([self.dataset, regexp_occurrences], axis=1)

        self.dataset['title_and_description'] = self.dataset.title.fillna('') + ' ' \
            + self.dataset.description.fillna('')

        self.dataset.title_and_description = \
            self.dataset.title_and_description.parallel_apply(
                lambda string: replace_from_dict(self.punctuation, string)
            )
        self.dataset.title_and_description = \
            self.dataset.title_and_description.parallel_apply(
                lambda string: process_text(string, self.stopwords, morph)
            )

        self.dataset['text'] = self.dataset.title_and_description.parallel_apply(
            lambda string: re.sub('[^A-Za-z0-9\.\@\ \-\_]', ' ', string)
        )
        self.dataset['text'] = self.dataset['text'].parallel_apply(
            lambda string: re.sub(' +', ' ', string)
        )
        self.dataset['numbers'] = self.dataset.title_and_description.parallel_apply(
            lambda string: re.sub('[^0-9\+\(\)\-]', ' ', string)
        )
        self.dataset['numbers'] = self.dataset['numbers'].parallel_apply(
            lambda string: re.sub(' +', ' ', string)
        )

    def load_model(self) -> None:
        """Load the catboost model."""
        self.model = CatBoostClassifier().load_model(self.catboost_path)

    def predict(self) -> None:
        """Simply predict probabilities on given dataset."""
        self.predictions = self.model.predict_proba(self.dataset)[:, 1]
        indices = range(len(self.predictions))
        self.predictions = DataFrame(
            zip(indices, self.predictions),
            columns=['index', 'prediction'],
        )

    def run_model(self) -> DataFrame:
        """Full model pipeline.

        Returns:
            Series: probabilities of is_bad=True prediction.
        """
        # initialize pandarallel to speed up the computation
        pandarallel.initialize(progress_bar=False)
        self.load_regexes_and_punctuation()
        self.load_model()
        self.get_stopwords()
        self.prepare_dataset()
        self.dataset = self.dataset[self.model.feature_names_]
        self.predict()
        return self.predictions


def task1(test: DataFrame) -> DataFrame:
    """Run model on the given config.

    Should've been json reading in here but who cares.

    Args:
        test: a DataFrame we want to infer our models on.

    Returns:
        DataFrame: probabilities of is_bad=True prediction.
    """
    path_to_models = '/app/lib/models'
    catboost = CatBoost(
        dataset=test,
        punctuation_path='{0}/regexps/punctuation.json'.format(path_to_models),
        regexes_path='{0}/regexps/regexp.json'.format(path_to_models),
        catboost_path='{0}/catboost_classifier.cbm'.format(path_to_models),
        stopwords_path='{0}/stopwords.json'.format(path_to_models),
    )
    return catboost.run_model()


def task2():
    """Empty function just to exist."""
    pass
