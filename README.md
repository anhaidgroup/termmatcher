# TermMatcher
Homepage of the TermMatcher project

## Paper and Data
A link to our technical report can be found [here](https://pages.cs.wisc.edu/~tingcai/termmatcher_paper/main.pdf). The public NY data used in the paper can be found [here](https://pages.cs.wisc.edu/~tingcai/termmatcher_datasets/).

## Set up environment
* Python 3.8.18
### required packages
* pandas==1.2.0 scikit-learn==1.3.0 numpy==1.24.3 py_stringmtaching==0.4.3 nltk==3.5 dask==2021.11.2 gensim==4.3.2 joblib==1.2.0
### steps to run
* Follow the instructions to download the pre-trained context embeddings
* Change the dataset path in main.py (the gold matches) and bta_cnn_iterator/cnn_initializer.py (the column names and the business terms)
* Run ```python main.py```
