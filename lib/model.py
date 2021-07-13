"""A simple loader for your Logistic Regression model in scikit-learn."""
from dataclasses import dataclass
from json import load as json_load
from multiprocessing import cpu_count
from pickle import load as pickle_load
from typing import Dict, List, Tuple, Union

from dataenforce import Dataset
from numpy import empty, zeros
from numpy.typing import ArrayLike
from pandas import Series
from scipy.sparse import csr, hstack
from sklearn.base import RegressorMixin, TransformerMixin
from sklearn.metrics import roc_auc_score
from tabulate import tabulate
from transliterate import translit


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


def description_transformer(
    transformer: TransformerMixin,
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


def categories_transformer(
    transformer: TransformerMixin,
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


def regexp_transformer(series: Series, regexps: dict) -> ArrayLike:
    """Find the presence of regexps in your Series.

    Args:
        series: a pandas Series of strings,
        regexps: a dict with {regexp name: regexp string}.

    Returns:
        ArrayLike (numpy array) with ints: 0 if there is no such a regexp, 1 else.
    """
    columns = empty((len(regexps), len(series)), dtype=int)
    for index, (_, regexp) in enumerate(regexps.items()):
        columns[index] = series.str.contains(regexp).astype(int).values
    return columns.T


def auc_printer(
    predictions: ArrayLike,
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


def transliterate_word(word: str) -> str:
    """Get simple word transliteration.

    Args:
        word: word to transliterate (in cyrillic symbols).

    Returns:
        word transliteration RU -> ENG.
    """
    return translit(word.lower().replace(' ', '_'), 'ru', reversed=True)


def transliterate_list(words: List[str]) -> List[str]:
    """Get transliteration RU -> ENG and replaces some elements.

    Args:
        words: words list to transliterate.

    Returns:
        List of transliterated words.
    """
    return list(map(transliterate_word, words))


def splitted_predictor(
    models: Dict[str, RegressorMixin],
    dataset: Dataset,
    features: csr.csr_matrix,
    categories_names: List[Tuple[str, str]],
) -> List[float]:
    """Predicts by category in dataset.

    Args:
        dataset: dataset to make category mask.
        models: predictors dict.
        features: sample to predict.
        categories_names: transliteration of russian names to english.

    Returns:
        List[float]: probability that label is equal to 1.
    """
    labels_predicted = zeros(len(dataset))
    for category in list(zip(*categories_names))[0]:
        model = models[category]
        category_mask = dataset['category'] == category
        features_used = features[category_mask]
        # [:, 1] Because scikit-learn predict_proba returns probability of 0
        # along with probability of 1, but we need only 1's.
        labels_predicted[category_mask] = model.predict_proba(features_used)[:, 1]
    return labels_predicted


@dataclass
class LogReg:
    """Logistic Regression class."""

    dataset: Dataset
    tf_idf_path: str
    dict_vectorizer_path: str
    regexp_dict_path: str
    logregs_path: str
    tf_idf: TransformerMixin = None
    dict_vectorizer: TransformerMixin = None
    logregs: Dict[str, RegressorMixin] = None
    regexp_dict: dict = None
    categories = ['subcategory', 'category', 'region', 'city']
    features: csr.csr_matrix = None
    labels: Series = None
    predictions: ArrayLike = None
    categories_names: List[Tuple[str, str]] = None

    def load_logregs(self) -> None:
        """Load all the LogReg models."""
        logregs_loaded = {}
        for category, eng_name in self.categories_names:
            model_path = '/{0}/{1}.pickle'.format(self.logregs_path, eng_name)
            logreg = pickle_model_loader(model_path)
            logreg.n_jobs = cpu_count()
            logregs_loaded[category] = logreg
        self.logregs = logregs_loaded

    def categories_with_transliteration(self):
        """Get transliterated categories for dataset. Used for file naming."""
        categories = self.dataset['category'].unique()
        categories_transliterated = transliterate_list(categories)
        self.categories_names = list(zip(categories, categories_transliterated))

    def load_transformers(self) -> None:
        """Load TF-iDF, DictVectorizer, Regexps and LogReg models."""
        self.tf_idf = pickle_model_loader(self.tf_idf_path)
        self.dict_vectorizer = pickle_model_loader(self.dict_vectorizer_path)
        self.regexp_dict = safe_json_loader(self.regexp_dict_path)

    def prepare_dataset(self) -> None:
        """TF-iDF the description, DictVectorize category variables and find regexps."""
        clean_description = description_cleaner(self.dataset['description'])
        description = description_transformer(self.tf_idf, clean_description)
        category = categories_transformer(
            self.dict_vectorizer, self.categories, self.dataset,
        )
        regexps = regexp_transformer(clean_description, self.regexp_dict)
        self.features = hstack([description, category, regexps], format='csr')
        self.categories_with_transliteration()

    def predict(self) -> None:
        """Simply predict probabilities on given dataset."""
        self.predictions = splitted_predictor(
            self.logregs, self.dataset, self.features, self.categories_names,
        )

    def print_metrics(self) -> None:
        """Print AUC for the predictions."""
        auc_printer(self.predictions, self.labels, self.logreg)

    def run_model(self) -> List[float]:
        """Full model pipeline.

        Returns:
            Series[float]: probabilities of is_bad=True prediction.
        """
        self.load_transformers()
        self.prepare_dataset()
        self.load_logregs()
        self.predict()
        return self.predictions


def task1(test: Dataset) -> List[float]:
    """Run model on the given config.

    Should've been json reading in here but who cares.

    Args:
        test: a DataFrame we want to infer our models on.

    Returns:
        Series[float]: probabilities of is_bad=True prediction.
    """
    path_to_models = '/app/lib/models'
    logistic_regression_model = LogReg(
        dataset=test,
        tf_idf_path='{0}/text_transformer.pickle'.format(path_to_models),
        dict_vectorizer_path='{0}/cat_transformer.pickle'.format(path_to_models),
        regexp_dict_path='{0}/regexps/regexp.json'.format(path_to_models),
        logregs_path='{0}/logregs'.format(path_to_models),
    )
    return logistic_regression_model.run_model()


def task2():
    """Empty function just to exist."""
    pass
