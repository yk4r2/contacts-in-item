"""A simple loader for your Logistic Regression model in scikit-learn."""
from dataclasses import dataclass
from json import load as json_load
from multiprocessing import cpu_count
from pickle import load as pickle_load
from typing import List, Union

from dataenforce import Dataset
from numpy import array, empty
from numpy.typing import ArrayLike
from pandas import Series
from scipy.sparse import csr, hstack
from sklearn.base import RegressorMixin, TransformerMixin
from sklearn.metrics import roc_auc_score
from tabulate import tabulate


def pickle_model_loader(path: str) -> Union[RegressorMixin, TransformerMixin]:
    """Load your model from pickle. Might be insecure.

    Args:
        path: A path to your model.

    Returns:
        A model in sklearn format.
    """
    with open(path, 'rb') as model_file:
        return pickle_load(model_file)


def safe_json_loader(path: str) -> dict:
    """Load your dict from JSON.

    Args:
        path: A path to your dict.

    Returns:
        A dict.
    """
    with open(path, 'rb') as model_file:
        return json_load(model_file)


def description_cleaner(description: Series) -> Series:
    """Leave only literals and numerics in your description.

    Args:
        description: Series of dirty descriptions.

    Returns:
        A description with filtered punctuation.
    """
    return description.replace(r'[\W_]+', ' ', regex=True).str.lower()


def description_transformer(transformer: TransformerMixin,
                            description: Series,
                            ) -> csr.csr_matrix:
    """Transform your description using pre-loaded TF-iDF transformer.

    Args:
        transformer: TF-iDF transformer,
        description: Series of string-descriptions.

    Returns:
        Scipy's sparse matrix of TF-iDF vectorized strings.
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
        Scipy's sparse matrix of DictVectorized category features strings.
    """
    return transformer.transform(dataframe[categories_list].to_dict('records'))


def regexp_loader(path: str) -> dict:
    """Load the dict with regexps from JSON.

    Args:
        path: regexp's JSON path.

    Returns:
        dict with {regexp name: regexp string}.
    """
    with open(path) as json_file:
        return json_load(json_file)


def regexp_transformer(series: Series, regexps: dict) -> ArrayLike:
    """Find the presence of regexps in your Series.

    Args:
        series: a pandas Series of strings,
        regexps: a dict with {regexp name: regexp string}.

    Returns:
        An NDArray with ints: 0 if there is no such a regexp, 1 else.
    """
    columns = empty((len(regexps), len(series)), dtype=int)
    for index, (_, regexp) in enumerate(regexps.items()):
        columns[index] = series.str.contains(regexp).astype(int).values
    return columns.T


def auc_printer(predictions: ArrayLike,
                labels: Series,
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
    categories = ['subcategory', 'category', 'region', 'city']
    features: csr.csr_matrix = None
    labels: Series = None
    predictions: ArrayLike = None

    def load_models(self) -> None:
        """Load TF-iDF, DictVectorizer, Regexps and LogReg models."""
        self.tf_idf = pickle_model_loader(self.tf_idf_path)
        self.dict_vectorizer = pickle_model_loader(self.dict_vectorizer_path)
        self.logreg = pickle_model_loader(self.logreg_path)
        self.regexp_dict = safe_json_loader(self.regexp_dict_path)
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
        self.predictions = self.logreg.predict_proba(self.features)

    def print_metrics(self) -> None:
        """Print AUC for the predictions."""
        auc_printer(self.predictions, self.labels, self.logreg)

    def run_model(self) -> Series:
        """Full model pipeline.

        Returns:
            Series[float]: probabilities of is_bad=True prediction.
        """
        self.load_models()
        self.prepare_dataset()
        self.predict()
        return self.predictions


def task1(test: Dataset) -> Series:
    """Run model on the given config.

    Should've been json reading in here but who cares.

    Args:
        test: a DataFrame we want to infer our models on.

    Returns:
        Series[float]: probabilities of is_bad=True prediction.
    """
    path_to_models = '/app/lib/logreg_models'
    logistic_regression_model = LogReg(
        dataset=test,
        tf_idf_path='{0}/text_transformer.pickle'.format(path_to_models),
        dict_vectorizer_path='{0}/cat_transformer.pickle'.format(path_to_models),
        regexp_dict_path='{0}/regexps/regexp.json'.format(path_to_models),
        logreg_path='{0}/logreg.pickle'.format(path_to_models),
    )
    return logistic_regression_model.run_model()


def task2():
    """An empty function just to exist."""
    pass
