"""A simple loader for your Logistic Regression model in scikit-learn."""
from pickle import load as pickle_load
from dataclasses import dataclass
from json import load as json_load
from multiprocessing import cpu_count
from typing import List

from dataenforce import Dataset
from numpy import array
from numpy.typing import NDArray
from pandas import Series
from scipy.sparse import csr, hstack
from sklearn.base import RegressorMixin, TransformerMixin
from sklearn.metrics import roc_auc_score
from tabulate import tabulate


def model_loader(path: str) -> RegressorMixin:
    """Load your model from pickle. Might be insecure.

    Args:
        path: A path to your model.

    Returns:
        A model in sklearn format.
    """
    with open(path, 'rb') as model_file:
        return pickle_load(model_file)


def description_cleaner(description: Series[str]) -> Series[str]:
    """Leave only literals and numerics in your description.

    Args:
        description: Series of dirty descriptions.

    Returns:
        A description with filtered punctuation.
    """
    return description.replace(r'[\W_]+', ' ', regex=True).str.lower()


def description_transformer(transformer: TransformerMixin,
                            description: Series[str],
                            ) -> csr.csr_matrix:
    """Transform your description using pre-loaded TF-iDF transformer.

    Args:
        transformer: TF-iDF transformer,
        description: Series of string-descriptions.

    Returns:
        Numpy's NDArray of TF-iDF vectorized strings.
    """
    return transformer.transform(description)


def categories_transformer(transformer: TransformerMixin,
                           categories_list: List[str],
                           dataframe: Dataset,
                           ) -> csr.csr_matrix:
    """Transform categorial features using DictVectorizer from scipy.

    Args:
        transformer: DictVectorizer,
        categories_list: List of category features,
        dataframe: a pandas Dataframe waiting to be vectorized.

    Returns:
        Numpy's NDArray of DictVectorized category features strings.
    """
    return transformer.fit_transform(dataframe[categories_list].to_dict('cat'))


def regexp_loader(path: str) -> dict:
    """Load the dict with regexps from JSON.

    Args:
        path: regexp's JSON path.

    Returns:
        dict with {regexp name: regexp string}.
    """
    with open(path) as json_file:
        return json_load(json_file)


def regexp_transformer(series: Series[str], regexps: dict) -> NDArray[int]:
    """Find the presence of regexps in your Series.

    Args:
        series: a pandas Series of strings,
        regexps: a dict with {regexp name: regexp string}.

    Returns:
        An NDArray with ints: 0 if there is no such a regexp, 1 else.
    """
    columns = array(len(regexps), len(series))
    for index, (_, regexp) in enumerate(regexps.items()):
        columns[index] = series.str.contains(regexp).astype(int)
    return columns


def auc_printer(predictions: NDArray,
                labels: Series[int],
                model: RegressorMixin,
                ) -> None:
    """Print your model's AUC.

    Args:
        predictions: predicted by model labels,
        labels: ground truth labels,
        model: model that predicted labels. Used to get its name.
    """
    headers = ['Model', 'Metric', 'Value']
    table = []
    model_name = type(model).__name__
    roc_auc = roc_auc_score(labels, predictions)
    table.append([model_name, 'AUC', roc_auc])
    print(tabulate(table, headers=headers, tablefmt='orgtbl'))


@dataclass
class LogReg:
    """Logistic Regression class."""

    dataset: Dataset
    tf_idf_path: str
    dict_vectorizer_path: str
    regexp_dict_path: str
    logreg_path: str
    tf_idf: TransformerMixin = None
    dict_vectorizer: TransformerMixin = None
    logreg: RegressorMixin = None
    regexp_dict: dict = None
    categories: List[str] = None
    features: csr.csr_matrix = None
    labels: Series[int] = None
    predictions: NDArray = None

    def load_models(self) -> None:
        """Load TF-iDF, DictVectorizer, Regexps and LogReg models."""
        with open(self.tf_idf_path, 'rb') as tf_idf_file:
            self.tf_idf = pickle_load(tf_idf_file)
        with open(self.dict_vectorizer_path, 'rb') as dict_vectorizer_file:
            self.dict_vectorizer = pickle_load(dict_vectorizer_file)
        with open(self.regexp_dict_path, 'rb') as regexps_json:
            self.regexp_dict = json_load(regexps_json)
        with open(self.logreg_path, 'rb') as logreg_file:
            self.logreg = pickle_load(logreg_file)
        self.logreg.n_jobs = cpu_count()

    def prepare_dataset(self) -> None:
        """TF-iDF the description, DictVectorize category variables and find regexps."""
        clean_description = description_cleaner(self.dataset['description'])
        description = description_transformer(self.tf_idf, clean_description)
        category = categories_transformer(
            self.dict_vectorizer, self.categories, self.dataset,
        )
        self.regexp_dict = regexp_loader(self.regexp_dict_path)
        regexps = regexp_transformer(clean_description, self.regexp_dict)
        self.features = hstack([description, category, regexps])
        self.labels = self.dataset['is_bad']

    def predict(self) -> None:
        """Simply predict probabilities on given dataset."""
        self.predictions = self.logreg.predict_proba(self.data)

    def print_metrics(self) -> None:
        """Print AUC for the predictions."""
        auc_printer(self.predictions, self.labels, self.logreg)
