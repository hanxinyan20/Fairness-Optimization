import utils
import numpy as np
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict
from fairness_algorithm import ILP

# fairness_tradeoff = 1.0
# ranklist_length = 1
# iteration_times = 500
# rel_distribution = "uniform"
# position_bias_distribution = "singular"

parser = argparse.ArgumentParser(description='Pipeline commandline argument')
parser.add_argument("--rel_distribution",type=str,default="uniform")
parser.add_argument("--position_bias_distribution",type=str,default="singular")
#parser.add_argument("--fairness_tradeoff",type=float,default=1.0)
parser.add_argument("--ranklist_length",type=int,default=1)
parser.add_argument("--iteration_times",type=int,default=200)


args = parser.parse_args()

if __name__ == "__main__":
    
    #fairness_tradeoff = args.fairness_tradeoff
    ranklist_length = args.ranklist_length
    iteration_times = args.iteration_times
    rel_distribution = args.rel_distribution
    position_bias_distribution = args.position_bias_distribution

    qid_docidlist_dict = defaultdict()
    qid_reltensor_dict = defaultdict()
    position_bias = []
    if(rel_distribution=="uniform"):
        (qid_docidlist_dict,qid_reltensor_dict) = utils.uniform(1,100,0.5)
    elif(rel_distribution=="linear"):
        (qid_docidlist_dict,qid_reltensor_dict) = utils.linear(1,100)
    else:
        (qid_docidlist_dict,qid_reltensor_dict) = utils.exponential(1,100)
    
    if(position_bias_distribution=="singular"):
        position_bias = utils.position_bias.singular(ranklist_length)
    else:
        position_bias = utils.position_bias.geometric(ranklist_length,5,0.5)

    for fairness_tradeoff in [0,0.6,0.8,1.0]:
        fname= "test/"+rel_distribution+" "+position_bias_distribution+" "+"tradeoff:"+str(fairness_tradeoff) + ".txt"
        f = open(fname,"w+")
        ilp = ILP(qid_reltensor_dict,qid_docidlist_dict,position_bias,fairness_tradeoff)

        for _ in range(iteration_times):
            rl = ilp.get_ranking_list(0,ranklist_length)

        #print(ilp.unfairness_list)
        # plt.plot([i for i in range(iteration_times)],ilp.unfairness_list,label="tradeoff:"+str(fairness_tradeoff))
        # plt.xlabel("iteration")
        # plt.ylabel("unfairness")
        # plt.show()
        f.write(rel_distribution+" "+position_bias_distribution+" "+"tradeoff:"+str(fairness_tradeoff))
        #print(ilp.unfairness_list)
        f.writelines([str(unfairness)+"\n" for unfairness in ilp.unfairness_list])
        f.close()

    