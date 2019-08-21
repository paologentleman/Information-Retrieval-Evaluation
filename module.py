from __future__ import division, unicode_literals 
import codecs
import os
from bs4 import BeautifulSoup
from os import listdir
from os.path import isfile, join
from whoosh.index import create_in
from whoosh.fields import *
from whoosh.analysis import SimpleAnalyzer, StemmingAnalyzer, StandardAnalyzer, FancyAnalyzer
from whoosh import index
from whoosh.qparser import *
from whoosh import scoring
from whoosh import index
import numpy as np
from numpy import percentile
import math
import csv
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

'''
storing the queries in a dictionary with the following shape
{query_id : "this OR is OR a OR query"}
'''
def initialize_query_dictionary():
    dict_query = {}
    file = open('cran_Queries.tsv')
    flag1 = False
    for line in file:
        if flag1 == False:
            flag1 = True
        else:
            id, query = line.split('\t')
            dict_query[int(id)] = query.strip()
    return dict_query


'''
Storing the ground truth in a dictionary with the query_id as a key
and the list of the relevant doc_id(s) as value
'''
def initialize_gt_dictionery():
    dict_gt = defaultdict(list)
    file = open('cran_Ground_Truth.tsv')
    flag2 = False
    for line in file:
        if flag2 == False:
            flag2 = True
        else:
            query_id, relevant_doc_id = line.split('\t')
            dict_gt[int(query_id)].append(int(relevant_doc_id.strip()))

    return dict_gt


'''
Storing the results of the research in a dictionary, will be used to compute the nDCG value
to be inserted in the respective  array each configuration.
'''
def extrapolate_dictionary_from_search(results, results_dictionary, query):
    for hit in results:
        results_dictionary[query[0]].append(int(hit['id']))


def dcg(k, doc_list, q, dict_gt):
    dcg = 0
    for p in range(k):
        if doc_list[p] in dict_gt[q]:
            dcg += (1/(math.log((p + 2), 2)))
    return dcg

def idcg(k, q, dict_gt):
    idcg = 0
    n = k
    if k > len(dict_gt[q]):
        n = len(dict_gt[q])
    for p in range(n):
        idcg += (1/(math.log((p + 2), 2)))
    return idcg

def compute_search(analyzer, score_function, folder, dictionary_queries, dictionary_ground_truth):

    selected_analyzer = analyzer
    #creating the shema
    schema = Schema(id=ID(stored=True), content_body=TEXT(stored=False, analyzer=selected_analyzer), content_title=TEXT(stored=False, analyzer=selected_analyzer))
    ix = create_in(folder, schema)
    ix = index.open_dir(folder)
    writer = ix.writer(procs=2, limitmb=500)

    #extracting the query from the text in the html files through BeautifulSoup library
    #obtaining also the id of each doc from it's filename

    onlyfilenames = [f for f in listdir('DOCUMENTS') if isfile(join('DOCUMENTS', f))]
    for f in onlyfilenames:
        filename=codecs.open('DOCUMENTS/'+f, 'r', 'utf-8')
        document = BeautifulSoup(filename.read(), "html.parser")
        title = (document.find('title').text).replace('\n', ' ')
        body = (document.find('body').text).replace('\n', ' ')
        idn = os.path.splitext(f)[0].replace('_', '')

        writer.add_document(id = idn, content_body = body, content_title = title)
    writer.commit()

    ix = index.open_dir(folder)
    

    '''
    arrays that will be used to store the values in order to compute the MRR an R-precision values
    min_res ->  MRR, will have to compute the sum of the values inside this array
    r_prec ->   r-precision, for configuration and for each query searchedthis array will be filled with 
                the r-precision value of the query
    '''
    min_res = [] 
    r_prec = [] 
    
    #dictionary that will contain the results of the results, will be used to compute the nDCG values
    results_dictionary = defaultdict(list)

    for q in dictionary_queries.items():

        input_query = q[1]
        prec_count = 0
        
        qp = MultifieldParser(["content_body","content_title"], ix.schema)
        parsed_query = qp.parse(input_query)# parsing the query
        searcher = ix.searcher(weighting=score_function)
        results = searcher.search(parsed_query, limit = 40)
        
        res_dict = extrapolate_dictionary_from_search(results, results_dictionary, q)

        for hit in results:
            #computing MRR
            if int(hit['id']) in dictionary_ground_truth[q[0]]:
                min_res.append(1/(int(hit.rank) + 1))
                break
        mrr = round(sum(min_res)/222, 2)

        ndcg_results = []

        if mrr >= 0.32:

            #Extracting R-precision distribution
            for hit in results:

                if int(hit.rank) == len(dictionary_ground_truth[q[0]]):
                    r_prec.append((prec_count)/(len(dictionary_ground_truth[q[0]])))
                    break

                if int(hit['id']) in dictionary_ground_truth[q[0]]:
                    prec_count += 1

                '''if int(hit['id']) in dictionary_ground_truth[q[0]]:
                                                                    prec_count += 1'''

            
            #obtaining nDCG@k vlues with k [1,10] and filling the array that will be plotted in the main()
            for k in range(1, 11):
                t = 0
                for key in results_dictionary.keys():
                    dcg_val = dcg(k, results_dictionary[key], key, dictionary_ground_truth)
                    idcg_val = idcg(k, key, dictionary_ground_truth)
                    ndcg = dcg_val/idcg_val
                    t += ndcg

                ndcg_results.append(t/len(results_dictionary))

        searcher.close()
    print('MRR for this configuration: ' + str(mrr))


    # obtaining the R-precision distribution informution with the help of numpy
    if len(r_prec) != 0:           
        mean = round(np.mean(r_prec), 3)
        min = round(np.min(r_prec), 3)
        q1 = round(np.percentile(r_prec, .25), 3) 
        median = round(np.median(r_prec), 3)
        q3 = round(np.percentile(r_prec, .75), 3)
        max = round(np.max(r_prec), 3)

        print('R-precision distribution table: ' + '\n')
        print('_____________________________')
        print('| ' + 'mean         ' + '\t' + ' |' + str(mean) + '\t' + '|')
        print('| ' + 'min          ' + '\t' + ' |' + str(min) + '\t' + '|')
        print('| ' + '1st quartile ' + '\t' + ' |' + str(q1) + '\t' + '|')
        print('| ' + 'median       ' + '\t' + ' |' + str(median) + '\t' + '|')
        print('| ' + '3rd quartile ' + '\t' + ' |' + str(q3) + '\t' + '|')
        print('| ' + 'max          ' + '\t' + ' |' + str(max) + '\t' + '|')
        print('_____________________________')
        print('______________________________________________________________')

    else:
        print('______________________________________________________________')

    return ndcg_results

    

    



def main():
    dict_query = initialize_query_dictionary()
    dict_gt = initialize_gt_dictionery()

    analyzers = [SimpleAnalyzer(), StemmingAnalyzer(), StandardAnalyzer(), FancyAnalyzer()]
    scoring_functions = [scoring.TF_IDF(), scoring.Frequency(), scoring.BM25F()]
    directories = ['./Simple', './Stemming', './Standard', './Fancy']

    directory_index = -1

    ndcg_arrays_to_plot = []

    print('______________________________________________________________')
    
    for a in range(0,len(analyzers)):
        directory_index += 1
        
        for score_f in range(0, len(scoring_functions)):
            if a == 0: 
                print('Analyzer : Simple Analyzer')
            if a == 1: 
                print('Analyzer : Stemming Analyzer')
            if a == 2: 
                print('Analyzer : Standard Analyzer')
            if a == 3: 
                print('Analyzer : Fancy Analyzer')
            if score_f == 0:
                print('Scoring function: TF_ID')
            if score_f == 1:
                print('Scoring function: Frequency')
            if score_f == 2:
                print('Scoring function: BM25F')

            vec = compute_search(analyzers[a], scoring_functions[score_f], directories[directory_index], dict_query, dict_gt)

            if len(vec) != 0:

                ndcg_arrays_to_plot.append(vec)

            
    list_of_configs = ['Simple + TF_IDF', 'Simple + Frequency', 'Simple + BM25F', 'Stemming + TF_IDF', 'Stemming + Frequency', 'Stemming + BM25F', 'Standard + TF_IDF', 'Standard + Frequency', 'Standard + BM25F', 'Fancy + TF_IDF', 'Fancy + Frequency', 'Fancy + BM25F']
    list_of_colors = ['yellow', 'green', 'orange', 'blue', 'black', 'red', 'purple', 'brown', 'grey', 'cyan']
    for i in range(10):
        plt.plot(np.arange(1,11,1),ndcg_arrays_to_plot[i],color=list_of_colors[i],label=list_of_configs[i])

        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left",fontsize=10)

        plt.xlabel("k")
        plt.ylabel("avg nDCG over all provided queries")

    plt.show()
        

if __name__ == '__main__':
    main()




