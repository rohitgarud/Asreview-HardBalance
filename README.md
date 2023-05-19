# ASReview HardBalance Extension

This extension adds a new balancer based on the similarity between the features of the relevant records and the irrelevant records. The most similar irrelevant record for each relevant record is selected, which results in undersampling the irrelevant class. This balances the classes by having only as many irrelevant records as the relevant records for training the classifiers.

## Getting started

To install this extension, clone the repository to your system and then run the following command from inside the repository.

```bash
pip install .
```

or you can directly install it from GitHub using

```bash
pip install git+https://github.com/rohitgarud/Asreview-HardBalance.git
```

## Usage
After installation, the HardBalance can be used as any other balancer in the simulation mode using:
```bash
asreview simulate benchmark:van_de_Schoot_2017 -m similarity -e doc2vec -b hard
```
Although the hard balance strategy can be used with TFIDF features, due to similarity measurements and the high dimensionality of TFIDF features, it is prohibitively slow for larger datasets. Hence, other features such as Doc2Vec features are recommended.

There are three different similarity metrics available, `cosine` (cosine similarity), `dot_product` (dot product) and `euclidean_dist` (Euclidean distance) between the feature vectors of the relevant records and irrelevant records.

## License

Apache 2.0 license
