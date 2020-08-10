## import modules here
from pyspark import SparkContext, SparkConf
import pickle

########## Question 1 ##########
# do not change the heading of the function
offset=0
def count(hashes_1, hashes_2, offset):
    rdd_1 = hashes_1.map(lambda x: (x[0],[abs(a-b) for a,b in zip(x[1], hashes_2)]))
    #if satisfy the condition, return the data list, else is null
    rdd_2 = rdd_1.map(lambda x:(x[0], sum([1 for c in x[1] if c <= offset])))
    #retrun (key,count)
    return rdd_2

def c2lsh(data_hashes, query_hashes, alpha_m, beta_n):
    offset=0
    while True:
        rdd = count(data_hashes,query_hashes,offset)
        rdd_3 = rdd.filter(lambda x: x[1] >= alpha_m).keys()
        if rdd_3.count() < beta_n:
            offset = offset + 1
        else:
            break
    return rdd_3
    







