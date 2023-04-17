
from mip import Model, xsum, minimize, BINARY,OptimizationStatus
import torch
import numpy as np
from queue import PriorityQueue

class ILP:
   
    def __init__(self,qid_reltensor_dict,qid_docidlist_dict,position_bias,fairness_tradeoff):
        '''
    
        position_bias: 
            1D array,
            example: np.array([0.9],[0.6],[0,4],[0,28],[0,2],[0.15],[0.1])

        '''
        self.qid_reltensor_dict = qid_reltensor_dict
        self.qid_docidlist_dict = qid_docidlist_dict
        self.position_bias = position_bias
        self.fairness_tradeoff = fairness_tradeoff
        self.qid2Freq_dict = dict()
        for qid in qid_docidlist_dict:
            self.qid2Freq_dict[qid] = 0
        self.qid2accuRelArray_dict = dict()
        self.qid2accuExpArray_dict = dict()
        for (qid,docid_list) in qid_docidlist_dict.items():
            self.qid2accuRelArray_dict[qid] = np.zeros(len(docid_list))
            self.qid2accuExpArray_dict[qid] = np.zeros(len(docid_list))


    def get_ranking_list(self,qid,ranking_list_length,candidate_num=None):
        '''
        
        '''

        # part1: get preparation
        q_accuRel_array = self.qid2accuRelArray_dict[qid]
        q_accuExp_array = self.qid2accuExpArray_dict[qid]
        qrel_array = self.qid_reltensor_dict[qid].numpy()
        doc_num = q_accuRel_array.shape[0]
        q_freq = self.qid2Freq_dict[qid]
        ranking_list_length,candidate_num,position_bias = self.check_validation(ranking_list_length,candidate_num,doc_num)
        # print("ranking list length:",ranking_list_length)
        # print("candidate_num:",candidate_num)
        # print("position bias:",position_bias)
        # part2: select candidates
        i2idx_dict = dict()
        if(candidate_num!=None):
            
            # select the most relevance k candidates
            q = PriorityQueue()
            for idx in range(doc_num):
                q.put((qrel_array[idx],idx))
            for i in range(ranking_list_length):
                i2idx_dict[i] = idx = q.get()[1]
            
            # select the most unfair n-k candidates
            q = PriorityQueue()
            for idx in range(doc_num):
                q.put((q_accuExp_array[idx]-q_accuRel_array[idx]-qrel_array[idx],idx))

            for i in range(ranking_list_length,candidate_num):
                idx = q.get()[1]
                while(idx in i2idx_dict.values()):
                    idx = q.get()[1]
                i2idx_dict[i] = idx
            q_accuRel_array = np.array([q_accuRel_array[i2idx_dict[i]] for i in range(candidate_num)])
            q_accuExp_array = np.array([q_accuExp_array[i2idx_dict[i]] for i in range(candidate_num)])
            qrel_array = np.array([qrel_array[i2idx_dict[i]] for i in range(candidate_num)])
            position_bias = position_bias[:candidate_num]
            doc_num = candidate_num
        
        # part3: normalize
        position_bias,qrel_array,q_accuExp_array,q_accuRel_array = self.normalize(position_bias,qrel_array,q_accuExp_array,q_accuRel_array)
        
        
        # part4: construct ILP problem
        OBJ = np.abs(q_accuExp_array[:,np.newaxis]*q_freq+position_bias[np.newaxis,:]-q_accuRel_array[:,np.newaxis]*q_freq-qrel_array)
        model = Model()
        model.verbose=0
        X = [[model.add_var(var_type=BINARY) for _ in range(doc_num)] for _ in range(doc_num)]
        model.objective = minimize( xsum(OBJ[i][j]*X[i][j] for i in range(doc_num) for j in range(doc_num)))

        for i in range(doc_num):
            model += xsum(X[i][j] for j in range(doc_num) ) == 1
        for j in range(doc_num):
            model += xsum(X[i][j] for i in range(doc_num) ) == 1  
            
        # 计算idcg
        idcg = np.sum((np.float_power(np.array([2]*ranking_list_length),np.sort(qrel_array, axis=0)[::-1][:ranking_list_length])-np.array([1]*ranking_list_length))*position_bias[:ranking_list_length])
        # 添加ndcg约束
        mat_constr=(np.float_power(np.array([2]*doc_num),qrel_array)-np.array([1]*doc_num))[:,np.newaxis]*position_bias[np.newaxis,:]
        model += xsum(mat_constr[i][j]*X[i][j] for i in range(doc_num) for j in range(doc_num))>=(1-self.fairness_tradeoff)*idcg
            
        status = model.optimize()

        # part5: get solution of ILP problem and get ranking list
        xx = []
        if status == OptimizationStatus.OPTIMAL or status == OptimizationStatus.FEASIBLE:
            print("Utility constrain succeeded")
            for i, var in enumerate(model.vars):
                xx.append(var.x)
            xx=np.array(xx)
            xx=xx.reshape((doc_num,doc_num))
            rankingAllItem=np.argsort(-xx,0)[0,:]
            ranking=rankingAllItem[:ranking_list_length]
        else:
            print("Utility constrain failed.")
            ranking=(-qrel_array).argsort()[:ranking_list_length]
        
        if(candidate_num!=None):
            ranking = np.array([i2idx_dict[i] for i in ranking])
        
        # part6: update accumulativeExp&Rel
        self.update_accuExp(qid,ranking,position_bias)
        self.update_accuRel(qid)
        self.update_query_freq(qid)

        # part7: get the true ranking list
        true_ranking_list = self.get_true_ranking_list(qid,ranking)
        return true_ranking_list
    

    def check_validation(self,ranking_list_length,candidate_num,doc_num):
        '''
        make sure the ranking_list_length<=candidate_num<=doc_num, also padding the position_bias if necessary.
        '''
        # check ranking_list_length
        if(ranking_list_length>doc_num):
            print("Attention:The ranking list length is longer than doc_num. To get a ranking list,we reset ranking_list_length equal to doc_num")
            ranking_list_length = doc_num
        
        # check candidate_num
        if(candidate_num!=None):
            if(candidate_num<ranking_list_length):
                print("Attention:The candidate number is less then ranking list length. To get a ranking list, we reset candidate number to ranking list length.")
                candidate_num = ranking_list_length
            if(candidate_num>doc_num):
                print("Attention:The candidate number is greater then doc_num. To get a ranking list, we reset candidate number to doc_num.")
                candidate_num = doc_num
        
        # padding position_bias
        position_bias = np.zeros(doc_num)
        if(ranking_list_length>self.position_bias.shape[0]):
            print("Attention:The ranking list length is longer than position_bias. We'll padding position_bias with 0.")
            position_bias[:self.position_bias.shape[0]]=self.position_bias[:self.position_bias.shape[0]]
        else:
            position_bias[:ranking_list_length]=self.position_bias[:ranking_list_length]
        return (ranking_list_length,candidate_num,position_bias)
    
   
    def get_true_ranking_list(self,qid,ranking_list):
        '''
        the element of ranking_list is the index of document, but we need the docid
        '''
        return [self.qid_docidlist_dict[qid][idx] for idx in ranking_list]
    
    def update_accuExp(self,qid,ranking_list,position_bias):
        for ranking in range(len(ranking_list)):
            idx = ranking_list[ranking]
            self.qid2accuExpArray_dict[qid][idx] += position_bias[ranking]
        print("accumulateexp is:",self.qid2accuExpArray_dict[qid])
    def update_accuRel(self,qid):
        doc_num = self.qid2accuRelArray_dict[qid].shape[0]
        for i in range(doc_num):
            rel = self.qid_reltensor_dict[qid][i]
            self.qid2accuRelArray_dict[qid][i] += rel
        print("accumulateerel is:",self.qid2accuRelArray_dict[qid])
    def update_query_freq(self,qid):
        self.qid2Freq_dict[qid]+=1  
    def normalize(self,*param):
        return [ i/i.sum() for i in param]
    
