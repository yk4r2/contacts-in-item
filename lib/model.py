from typing import Tuple, Union
from numpy.typing import NDArray

from pandas import Series
from dataenforce import Dataset

from pandas import read_csv, concat
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack, csr
from sklearn.metrics import roc_auc_score
from sklearn.base import RegressorMixin, TransformerMixin
from datetime import datetime
from stop_words import get_stop_words
from tqdm.notebook import tqdm
from multiprocessing import cpu_count
from json import load as json_load
from numpy import int64, array, logspace, isnan, isfinite, nan_to_num
from joblib import Parallel, delayed
import pickle.load as pickle_load
from dataclasses import dataclass, field
from tabulate import tabulate


def model_loader(path: str) -> RegressorMixin:
    return pickle_load(open(path, "rb"))


def description_cleaner(description: Series[str]) -> Series[str]:
    return description.replace(r'[\W_]+', ' ', regex=True).str.lower()


def description_transformer(transformer: TransformerMixin,
                            description: Series[str],
                            ) -> csr.csr_matrix:
    return transformer.fit_transform(description)


def categories_transformer(transformer: TransformerMixin,
                           categories_list: List[str],
                           dataframe: Dataset["title", ...],
                           ) -> csr.csr_matrix:
    return transformer.fit_transform(dataframe[categories_list].to_dict("cats"))


def regexp_loader(path: str) -> dict:
    with open(path) as json_file:
        return json_load(json_file)


def regexp_transformer(series: Series[str], regexps: dict) -> NDArray[int]:
    columns = array(len(regexps), len(series))
    for index, (_, regexp) in enumerate(regexps.items()):
        columns[index] = series.str.contains(regexp).astype(int)
    return columns


def predict(test: csr.csr_matrix, model: RegressorMixin) -> NDArray:
    return model.predict_proba(test)


def metrics_printer(predictions: NDArray,
                    labels: Series[int],
                    model: RegressorMixin,
                    ) -> None:
    headers = ['Model', 'Metric', 'Value']
    table = []

    model_name = type(model).__name__
    roc_auc = roc_auc_score(labels, predictions)

    table.append([model_name, ds_type, "accuracy", accuracy])
    table.append([model_name, ds_type, "AUC", roc_auc])
    print(tabulate(table, headers=headers, tablefmt='orgtbl'))


@dataclass
class LogReg:
    dataset: Dataset["title", ...]
    tf_idf_path: str
    dict_vectorizer_path: str
    regexp_dict_path: str
    logreg_path: str
    self.tf_idf = filed(default=None, default_factory=TransformerMixin)
    self.dict_vectorizer = field(default=None, default_factory=TransformerMixin)
    self.logreg = field(default=None, default_factory=RegressorMixin)
    self.regexp_dict = field(default=None, default_factory=dict)
    self.categories = field(default=['subcategory', 'category', 'region', 'city'])
    self.data = field(default=None, default_factory=csr.csr_matrix)
    self.labels = field(default=None, default_factory=Series[int])
    self.predictions = field(default=None, default_factory=NDArray)

    def load_models(self) -> None:
        self.tf_idf = pickle_load(self.tf_idf_path)
        self.dict_vectorizer = pickle_load(self.dict_vectorizer_path)
        with open(self.regexp_dict_path) as regexps_json:
            self.regexp_dict = json_load(regexps_json)
        self.logreg = pickle_load(self.logreg_path)

    def prepare_dataset(self) -> None:
        clean_description = description_cleaner(self.dataset["description"])
        description = description_transformer(self.tf_idf, clean_description)
        category = categories_transformer(self.dict_vectorizer,
                                          self.categories,
                                          self.dataset,
                                          )
        self.regexp_dict = regexp_loader(self.regexp_dict_path)
        regexps = regexp_transformer(clean_description, self.regexp_dict)
        self.data = hstack([description, category, regexps])
        self.labels = self.dataset["is_bad"]

    def predict(self) -> None:
        self.predictions = self.logreg.predict_proba(self.data)

    def print_metrics(self) -> None:
        metrics_printer(self.predictions, self.labels, self.logreg)
