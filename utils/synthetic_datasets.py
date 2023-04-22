import torch
from math import exp
from collections import defaultdict
'''
return value:(qid_docidlist_dict,qid_reltensor_dict)
'''
def uniform(query_num,doc_num,rel):
    if(rel<0 or rel >1):
        rel = 0.5
    qid_docidlist_dict = defaultdict()
    qid_reltensor_dict = defaultdict()
    for j in range(query_num):
        qid_docidlist_dict[j] = [i for i in range(doc_num)]
        qid_reltensor_dict[j] = torch.tensor([rel]*doc_num)
    return (qid_docidlist_dict,qid_reltensor_dict)

def linear(query_num,doc_num):
    qid_docidlist_dict = defaultdict()
    qid_reltensor_dict = defaultdict()
    for j in range(query_num):
        qid_docidlist_dict[j] = [i for i in range(doc_num)]
        qid_reltensor_dict[j] = torch.tensor([i/doc_num for i in range(doc_num)])
    return (qid_docidlist_dict,qid_reltensor_dict)

def exponential(query_num,doc_num):
    qid_docidlist_dict = defaultdict()
    qid_reltensor_dict = defaultdict()
    for j in range(query_num):
        qid_docidlist_dict[j] = [i for i in range(doc_num)]
        qid_reltensor_dict[j] = torch.tensor([exp(-i-1) for i in range(doc_num)])
    #print("rel:",qid_reltensor_dict[0])
    return (qid_docidlist_dict,qid_reltensor_dict)
