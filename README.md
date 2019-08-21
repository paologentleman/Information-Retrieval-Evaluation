# Information Retrieval Evaluation

In this project the main goal is to index a collection of documents and improve the search-engine performance by changing its configurations using the provided set of queries and the associated Ground-Truth. For this purpose [Whoosh](https://whoosh.readthedocs.io/en/latest/) API is used.

Our aim is to obtain search engine configurations that have a ***Mean-Reciprocal-Rank*** value MRR(Q) ≥ **0.32**, where Q is the set of the provided queries. For each configuration the program will dislay a ***R-Precision*** distribution table containing the following information relative to the R-precision of each search engine configuration: *mean*, *min*, *1st quartile*, *median*, *3rd quartile*, *max*.

In order to visualize the effectiveness of the search engine, the ***nDCG@k*** (*Discounted Cumulative Gain*) plot is displayed.

![alt text](https://www.microsoft.com/en-us/research/uploads/prod/2018/06/InformationRetrieval_Carousel_06_2018_480x280-800x550.jpg)

# Files

In order to run properly:

1) create 4 folders in the root where you will run the .py called: "Simple", "Stemming", "Standard", "Fancy"
2) open the terminal in the root you've created the 4 directories
3) type run.sh

The **module.py** file contains the corpus of the code and an exemplary executable main .

The [data](https://trec.nist.gov/data/qrels_eng/) is relative to the "Cranfield experiments" which are computer information retrieval experiments conducted by professors at the College of Aeronautics at Cranfield in the 1960s, to evaluate the efficiency of indexing systems. The data represents the prototypical evaluation model of information retrieval systems, and this model has been used in large-scale information retrieval evaluation efforts such as the Text Retrieval Conference (TREC). The evaluation model relies on three components: 1. a document collection (or corpus), 2. a set of queries, 3. a set of relevance judgements, i.e. a file which for each query lists the documents regarded as relevant to answer the given query.
