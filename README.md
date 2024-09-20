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
* Change the variables: COL_NAME, BUS_TERM, DESCR, TAB_NAME in main.py according to different datasets
* Change the catalog_path in main.py (for the gold matches) and col_table_path, bt_table_path in bta_cnn_iterator/cnn_initializer.py (for the column names and the business terms) according to different datasets
* Run ```python main.py```
