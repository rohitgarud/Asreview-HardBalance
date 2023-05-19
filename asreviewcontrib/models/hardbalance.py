import numpy as np
from asreview.models.balance.base import BaseBalance
from asreview.utils import get_random_state
from scipy import sparse
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity


def l2_norm(x):
    return np.sqrt(np.sum(x**2))


def div_norm(x):
    norm_value = l2_norm(x)
    if norm_value > 0:
        return x * (1.0 / norm_value)
    else:
        return x


class HardBalance(BaseBalance):
    """Hard balance strategy (``hard``)

    This selects only hard irrelevant records for each relevant record
    using similarity metrics to undersample the irrelevant class

    Arguments
    ---------
    similarity_metric: str, optional
        Default: 'cosine'
        Similarity metric to find most similar irrelevant records
        for each relevant record

        'cosine': Cosine similarity between relevant and irrelevant records
        'dot_product': Dot product between relevant and irrelevant records
        'euclidean_dist': Euclidean distance between relevant and irrelevant
                          records

    with_replacement: boolean, optional
        Default: False
        Should the same irrelevant record be used as similar record
        for multiple relevant records
    """

    name = "hard"
    label = "Hard (similarity based)"

    def __init__(
        self,
        similarity_metric="cosine",
        with_replacement=False,
        random_state=None,
    ):
        """Initialize the hard balance strategy."""
        super(HardBalance, self).__init__()
        self.similarity_metric = similarity_metric
        self.with_replacement = with_replacement
        self.relevants = None
        self.irrelevants = None
        self._random_state = get_random_state(random_state)
        self.selected_irrelevants = []

    def sample(self, X, y, train_idx):
        """Resample the training data.

        Arguments
        ---------
        X: numpy.ndarray
            Complete feature matrix.
        y: numpy.ndarray
            Labels for all papers.
        train_idx: numpy.ndarray
            Training indices, that is all papers that have been reviewed.

        Returns
        -------
        numpy.ndarray, numpy.ndarray
            X_train, y_train: the resampled matrix, labels.
        """
        # Checking sparsity (TFIDF features)
        # WARNING: TFIDF not recommended with hard balance
        if sparse.isspmatrix_csr(X):
            X = X.toarray()
            print("WARNING: TFIDF not recommended with hard balance")

        self.relevants = X[y == 1, :]
        self.irrelevants = X[y == 0, :]
        if self.similarity_metric == "euclidean_dist":
            self.irrelevants = np.array(
                [
                    div_norm(np.array(self.irrelevants[i, :]))
                    for i in range(len(self.irrelevants))
                ]
            )
        X_train = self.relevants
        y_train = y[y == 1]

        n_one = self.relevants.shape[0]
        n_zero = self.irrelevants.shape[0]

        # If we don't have an excess of 0's, give back all training_samples.
        if ((n_one / n_zero) >= 1.0) and n_zero >= 1:
            return X, y
        else:
            for relevant in self.relevants:
                irrelevant = self._get_most_similar(
                    relevant.reshape(1, -1),
                ).reshape(1, -1)

                X_train = np.append(X_train, irrelevant, axis=0)
                y_train = np.append(y_train, np.array([0]), axis=0)

        return X_train, y_train

    def _get_most_similar(self, relevant):
        if self.similarity_metric == "cosine":
            sim = cosine_similarity(relevant, self.irrelevants).reshape(-1, 1)
            sim_idx = np.argmax(sim)

        elif self.similarity_metric == "dot_product":
            sim = np.array(
                [
                    np.dot(relevant, self.irrelevants[i, :])
                    for i in range(len(self.irrelevants))
                ]
            )
            sim_idx = np.argmax(sim)

        elif self.similarity_metric == "euclidean_dist":
            sim = pairwise_distances(
                self.irrelevants, relevant, metric="euclidean"
            )
            sim_idx = np.argmin(sim)

        else:
            raise ValueError("Unknown similarity metric")

        closest_irrelevant = self.irrelevants[sim_idx]

        if not self.with_replacement:
            self.irrelevants = np.delete(self.irrelevants, sim_idx, 0)

        return closest_irrelevant
