import torch
import numpy as np
from collections import defaultdict
import torch.nn.functional as F
def load_with_feature(filepath,feature_size,model):
    '''
    从filepath中加载数据并根据训练好的model生成相关性得分
    '''
    f = open(filepath)
    lines = f.readlines()
    feature_list = []
    docid_list = []
    qid_list = []
    for line in lines:
        tmp = line.split(" ")
        docid_list.append(tmp[0]) # docid:str
        qid_list.append(int(tmp[1])) #qid:int
        feature = torch.zeros(feature_size)
        for i in range(2,len(tmp)):
            feature_pair = tmp[i]
            key_value = feature_pair.split(":")
            feature[int(key_value[0])-1] = float(key_value[1])
        
        feature = feature.unsqueeze(0)
        feature_list.append(feature)
    
    qid_docidlist_dict = defaultdict(list)
    qid_rellist_dict = defaultdict(list)
    rels = model.build(feature_list)
    for i in range(len(lines)):
        rel = rels[i][0][0]
        docid = docid_list[i]
        qid = qid_list[i]
        qid_docidlist_dict[qid].append(docid)
        qid_rellist_dict[qid].append(rel)
        # 直接应用ILP求解出的ranklist中的每一项不是docid，而是doc在docid_list中的下标；需要一步转换，才能得到真正的ranklist
    return (qid_docidlist_dict,get_qid_reltensor_dict(qid_rellist_dict))

def load_with_rel(filepath):
    f = open(filepath)
    lines = f.readlines()
    qid_rellist_dict = defaultdict(list)
    qid_docidlist_dict = defaultdict(list)
    for line in lines:
        tmp = line.split(" ")
        docid= tmp[0]
        qid = int(tmp[1])
        rel = float(tmp[2])
        qid_rellist_dict[qid].append(rel)
        qid_docidlist_dict[qid].append(docid)
    return (qid_docidlist_dict,get_qid_reltensor_dict(qid_rellist_dict))



def get_qid_reltensor_dict(qid_rellist_dict):
    '''
    把list转化为tensor并把相关性softmax
    '''
    qid_reltensor_dict = defaultdict(list)
    for (qid,rellist) in qid_rellist_dict.items():
        reltensor = torch.tensor(rellist)
        print("before sofmax:",reltensor)
        reltensor = F.softmax(reltensor)
        qid_reltensor_dict[qid] = reltensor
    return qid_reltensor_dict

