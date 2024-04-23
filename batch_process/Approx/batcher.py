import json
import jsonlines
import tiktoken
import time
import numpy as np 
from util import *
from sklearn.cluster import DBSCAN
from hdbscan import HDBSCAN
import copy
import binpacking
from collections import Counter
import matplotlib.pyplot as plt
from batch_process.Approx.correlation import cluster_correlation_search
from batch_process.Approx.utils import get_clusters, transform_edge_weights
from batch_process.Approx.correlation import Loss 
import networkx as nx
import itertools




def set_cover(universe, subsets_ori,costs):
    subsets = copy.deepcopy(subsets_ori)
    cost=0
    elements = set(e for s in subsets for e in s)
    if not universe.issubset(elements):
        return [],[]
    covered = set()
    indexs = []
    cover = []
    while covered != elements:
        max_set = set()
        max_set_index = 0
        max_set_cost = 0
        is_find = False
        for i in range(len(subsets)):
            if len(subsets[i]-covered)/costs[i] > max_set_cost:
                max_set_cost = len(subsets[i]-covered)/costs[i]
                max_set = subsets[i]
                max_set_index = i
                is_find = True
        if not is_find:
            return [],[]
        cover.append(max_set)
        cost+=costs[max_set_index]
        indexs.append(max_set_index)
        covered |= max_set 
    return cover, indexs
   


def compute_efficient(dataset_name, k0, k1,batch_data,demo_data):
    if dataset_name in dataset_for_er:
        sim_type = 'structure'
        batch_feature = get_feature_structure([d['question'] for d in batch_data], dataset_name)
        demo_feature = get_feature_structure([d['question'] for d in demo_data], dataset_name)
        batch_dis_first = get_dis(batch_feature, batch_feature, dataset_name, sim_type)
        for i in range(batch_dis_first.shape[0]):
            for j in range(batch_dis_first.shape[1]):
                if batch_dis_first[i,j]!=0:
                    batch_dis_first[i,j]=1/batch_dis_first[i,j]        
        ori_QGA_num, QGA_num, ori_QDA_num, QDA_num = get_dist_opt_for_batch(batch_feature,demo_feature,k0,k1)
        print('ori_QGA_num:'+str(ori_QGA_num)+'  after optimization,QGA_num:'+str(QGA_num)+'  ratio:'+str(QGA_num/ori_QGA_num))
        print('ori_QDA_num:'+str(ori_QDA_num)+'  after optimization,QDA_num:'+str(QDA_num)+'  ratio:'+str(QDA_num/ori_QDA_num))
    elif dataset_name in dataset_for_code:
        sim_type = 'semantic'
        batch_feature = get_feature_semantic([d['total'] for d in batch_data], dataset_name)
        demo_feature = get_feature_semantic([d['total'] for d in demo_data], dataset_name)
        batch_feature_cpu = batch_feature.cpu()
        demo_feature_cpu = demo_feature.cpu()
        ori_QGA_num, QGA_num, ori_QDA_num, QDA_num = get_dist_opt_for_batch(batch_feature_cpu,demo_feature_cpu,k0,k1)
        print('ori_QGA_num:'+str(ori_QGA_num)+'  after optimization,QGA_num:'+str(QGA_num)+'  ratio:'+str(QGA_num/ori_QGA_num))
        print('ori_QDA_num:'+str(ori_QDA_num)+'  after optimization,QDA_num:'+str(QDA_num)+'  ratio:'+str(QDA_num/ori_QDA_num))
    else:
        print('dataset name must be er here')
        exit(-1)

def batcher_for_code(dataset_name, demo_method, filter_path, output_dir, batch_type='similar',sim_type='semantic', batch_size=8, demo_percentile=0.5, real_batch_data=None, demo_data=None, model='gpt-3.5-turbo-0125'):
    filter_name = filter_path.split('.json')[0]
    filter_name = filter_name.split('/')[-1]
    batch_data_index = []
    filter_real_path = filter_path
    with open(filter_real_path,'r') as filter_file:
        line = filter_file.readline()
        while line:
            json_line = json.loads(line)
            for one_que in json_line['ques']:
                batch_data_index.append(int(one_que))
            line= filter_file.readline()
    batch_data =[]
    for one_index in batch_data_index:
        batch_data.append(real_batch_data[one_index])
            
    encoder = tiktoken.encoding_for_model(model)
    
    cluster_percentile = 1
    min_samples_ = 1
    #取聚类问题的 threshold 
    batch_feature = get_feature_semantic([d['total'] for d in batch_data], dataset_name)
    grpDis = get_dis(batch_feature, batch_feature, dataset_name, sim_type)
    dis_threshold = np.percentile(grpDis, cluster_percentile)
    if dis_threshold <= 0.0:
        dis_threshold = 0.000001
    dbscanF_ = DBSCAN(eps=dis_threshold, min_samples=min_samples_, metric='cosine') 
    preds_ = dbscanF_.fit_predict(batch_feature.cpu())  
    pair2cluster, cluster2pair = {},  {}  
    for pid, cluster in enumerate(preds_):
        pair2cluster[pid] = cluster
        if cluster not in cluster2pair:
            cluster2pair[cluster] = [pid] #pid是新的问题索引
        else:
            cluster2pair[cluster].append(pid)
    #筛选例子
    demo_feature = get_feature_semantic([d['total'] for d in demo_data], dataset_name)
    grpDemoDis = get_dis(batch_feature, demo_feature, dataset_name, sim_type)
    grpdemo_threshol = np.percentile(grpDemoDis, demo_percentile)
    
    
    #组成group
    groups = []
    batch_data_num = len(batch_data)
    copy_cluster2pair = copy.deepcopy(cluster2pair)
    if batch_type == 'diverse':
        while len(copy_cluster2pair) > 0:
            tmp_group = []
            while len(tmp_group)<batch_size:
                cluster_to_remove = []
                for cluster, ques in copy_cluster2pair.items(): #group中的数据是
                    tmp_group.append(ques[0])
                    ques.pop(0) #放入group中的数据 要删除  移除首元素
                    if len(ques)==0: #某个cluster 没有数据要删除
                        cluster_to_remove.append(cluster)
                    if len(tmp_group)==batch_size:
                        break
                for one_cluster in cluster_to_remove:
                    del copy_cluster2pair[one_cluster]
                if len(copy_cluster2pair) == 0:
                    break
            groups.append(tmp_group)
    elif batch_type == 'similar':
        #每个cluster 放到一个group中
        tmp_group = []
        copy_copy_cluster2pair = {}
        for cluster , ques in copy_cluster2pair.items():
            while len(ques)>=batch_size:
                for one_que in ques:
                    tmp_group.append(one_que)
                    if len(tmp_group)==batch_size:
                        groups.append(tmp_group)
                        tmp_group = []
                        break
                ques = ques[batch_size:]
            copy_copy_cluster2pair[cluster] = ques
        assert len(tmp_group)==0
        for cluster, ques in copy_copy_cluster2pair.items():
            for one_que in ques:
                tmp_group.append(one_que)
                if len(tmp_group)==batch_size:
                    groups.append(tmp_group)
                    tmp_group = []
        if len(tmp_group)!=0:
            groups.append(tmp_group)
    else:
        print(batch_type+'  error')
        exit(-1)
    
    if batch_size ==1 and len(groups) != batch_data_num:
        print('group generation error 1')
        exit(-1)
    total_group_num = 0
    for one_group in groups:
        total_group_num+=len(one_group)
        for one_que in one_group:
            if one_que not in pair2cluster.keys():
                print('ques:'+str(one_que)+' not in cluster')
                exit(-1)
    if total_group_num != batch_data_num:
        print('batch_data_num:'+str(batch_data_num)+'  total_group_num:'+str(total_group_num))
        print('group generation error 2')
        exit(-1)
    demo_data_index, demo_coverage_ques_for_each_group = filter_demos_for_code(dataset_name, batch_data, demo_data, grpdemo_threshol, sim_type, groups) 

    #为每个group 选择例子
    queCost = []
    for question in batch_data:
        queCost.append(get_cost(question['total'], encoder, dataset_name))
    demCost = []
    for j in range(len(demo_data)):
        if j in demo_data_index:
            demCost.append(get_cost(demo_data[j]['total'], encoder, dataset_name))
        else:
            demCost.append(-1)
    
    all_groups = []
    prompts = []
    if demo_method == 'set_cover':
        group_index = 0
        for one_group in groups:
            universe = set()
            for q in one_group:
                universe.add(q)
            set_cost = []  #应该是每个例子的 set
            subsets = []
            demo_index_map =[0 for _ in range(len(demo_coverage_ques_for_each_group[group_index]))]
            tmp_index = 0
            for demo, ques_set in demo_coverage_ques_for_each_group[group_index].items():
                demo_index_map[tmp_index]=demo  #demo 索引的对应关系
                tmp_index+=1
                subsets.append(ques_set)
                if demCost[demo]==-1:
                    print('demo:'+str(demo)+'  not in the selected demonstrations')
                    exit(-1)
                set_cost.append(demCost[demo])
            #每个group  的question 和问题做一次 get_dist 
            _, indexes = set_cover(universe, subsets, set_cost)
            #确定 demo
            demos_for_one_group = []
            for one_index in indexes:
                demos_for_one_group.append(demo_index_map[one_index])
            tmp_group = {}
            tmp_group['ques'] = one_group
            tmp_group['demo'] = demos_for_one_group
            group_index += 1
            all_groups.append(tmp_group)  
    elif demo_method == '1demopair':
        #每个问题都选一个最相似的例子
        batch_demo_dis = grpDemoDis.tolist()            
        for one_group in groups:
            demos_for_one_group = set()
            for one_ques in one_group:
                sorted_indices = sorted(range(len(batch_demo_dis[one_ques])), key=lambda k: batch_demo_dis[one_ques][k]) #每个问题最相似的例子排序
                j=0
                while j<len(sorted_indices) and sorted_indices[j] in demos_for_one_group:
                    j+=1
                if j == len(sorted_indices):
                    exit(-1)
                demos_for_one_group.add(sorted_indices[j])  #sorted_indices[j]就是demo的索引
            one_group_demos = []
            for one_demo in demos_for_one_group:
                one_group_demos.append(one_demo)
            tmp_group = {}
            if len(one_group) != len(one_group_demos):
                print('demos and ques are not equal  '+str(len(one_group))+'  '+str(len(one_group_demos)))
            tmp_group['ques'] = one_group
            tmp_group['demo'] = one_group_demos
            all_groups.append(tmp_group)      
    else:
        print('demo_method error:'+demo_method)
    with jsonlines.open(output_dir + '/' + dataset_name +'/'+filter_name+'_'+batch_type+'_'+sim_type+'_'+demo_method+'_'+str(batch_size)+'_'+str(demo_percentile)+'_'+model+'_groups.json', 'w') as f_group:
        f_group.write_all(all_groups)
    
    
    tokens = 0
    for i in range(len(all_groups)):
        demos_data = []
        batches_data = []
        ans = []
        for j in all_groups[i]['demo']:
            demos_data.append(demo_data[j])
        for j in all_groups[i]['ques']:
            batches_data.append(batch_data[j])
            if dataset_name in dataset_for_code:
                ans.append(batch_data[j]['answer'])
            else:
                ans.append(int(batch_data[j]['answer']))
        prompt = get_prompt(dataset_name, demos_data, batches_data)
        tokens += len(encoder.encode(prompt))
        prompts.append([prompt, ans])
    with jsonlines.open(output_dir + '/' + dataset_name +'/'+filter_name+'_'+batch_type+'_'+sim_type+'_'+demo_method+'_'+str(batch_size)+'_'+str(demo_percentile)+'_'+model+'_'+model+'_prompts.json', 'w') as f_prompt:
        f_prompt.write_all(prompts)
    return  prompts, tokens     
        
    