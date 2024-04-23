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
import json

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


        
def overlap_items(dp_keys, cur_i, sets_ori):
    total_items ={}
    for i in range(0, cur_i):
        assert(i!=cur_i)
        for one_item in sets_ori[i]:
            if one_item in total_items:
                total_items[one_item] = 0
            total_items[one_item] += 1
    overlap_items=[]
    for cur_item in sets_ori[cur_i]:
        if cur_item in total_items.keys():
            overlap_items.append(cur_item)
    
    ret_removes = {}
    #dp_keys 是string类型的list 
    for one_del in dp_keys:
        #one_del 是字符串  将其分割 转成
        tmp_total_items = copy.deepcopy(total_items)
        remove_items = one_del.split('_')
        for one_item in remove_items:
            int_item = int(one_item)
            tmp_total_items[int_item]-=1 #最少是1个
            if tmp_total_items[int_item]<=0:
                print('error !!! delete item forever')
        #计算 sets_ori 与前面set的重复情况
        del_items = []
        for one_overlap in overlap_items:
            if tmp_total_items[one_overlap] >=2:
                del_items.append(one_overlap)
        ret_removes[one_del]=del_items
    return ret_removes    

def balance_sets_greedy(sets_ori):
    sets = copy.deepcopy(sets_ori)
    total_item = set()
    for one_set in sets:
        for one_item in one_set:
            total_item.add(one_item)
    total_item_num = len(total_item) #去重后的item总数---> 应该和问题数量相等
    
    avg_items_per_set = int(total_item_num/len(sets)) #每个集合最理想 的平均item 数量
    overlap_items = Counter(item for s in sets for item in s)
    #删除冗余的item
    while True:
        max_overlap_item, max_overlap_count = overlap_items.most_common(1)[0] #先从频率最高的item进行遍历
        if max_overlap_count == 1:
            break
        is_all_small=True
        for s in sets:
            len_s = len(s)
            if len_s > avg_items_per_set:
                to_remove = []
                for item in s:
                    if overlap_items[item] > 1:
                        is_all_small=False
                        to_remove.append(item)
                        overlap_items[item] -= 1 
                        len_s-=1
                        if len_s <= avg_items_per_set:
                            break
                for item in to_remove:
                    s.remove(item)
            if is_all_small:
                break
        if is_all_small:
            break
    while True: #随机删除小于avg_items_per_set中的冗余数据
        max_overlap_item, max_overlap_count = overlap_items.most_common(1)[0] #先从频率最高的item进行遍历
        if max_overlap_count == 1:
            break
        for s in sets:
            if overlap_items[max_overlap_item] == 1:
                break
            if max_overlap_item in s:
                s.remove(max_overlap_item)
                overlap_items[max_overlap_item]-=1  
    return sets


def approx_all_for_testpara(dataset_name, filter_path, tCost, upper_ques, k0, k1, k2, output_dir, batch_type='diverse',sim_type='structure', batch_data=None, demo_data=None, is_dominated=False, model='gpt-3.5-turbo-0125'):
    filter_name = filter_path.split('.json')[0]
    batch_data_index = []
    filter_real_path = 'output/group_assignment/'+filter_path
    with open(filter_real_path,'r') as filter_file:
        line = filter_file.readline()
        while line:
            json_line = json.loads(line)
            for one_que in json_line['ques']:
                batch_data_index.append(int(one_que))
            line= filter_file.readline()
    
    demo_data_index = filter_demos(dataset_name, batch_data, demo_data, k1, upper_ques, sim_type, batch_data_index)

    # 不用加入新的数组中 要保留原始的数据索引
    start_getfeature = time.time()
    new_batch_data_to_old = {}
    new_demo_data_to_old = {}
    new_demo_data=[]
    new_batch_data=[]
    for i in range(len(batch_data_index)):
        new_batch_data_to_old[i]=batch_data_index[i]
        new_batch_data.append(batch_data[batch_data_index[i]])
    for j in range(len(demo_data_index)):
        new_demo_data_to_old[j]=demo_data_index[j]
        new_demo_data.append(demo_data[demo_data_index[j]])
        
    if sim_type == 'structure':
        batch_feature = get_feature_structure([d['question'] for d in new_batch_data], dataset_name)
        demo_feature = get_feature_structure([d['question'] for d in new_demo_data], dataset_name) 
    else:
        if dataset_name in dataset_for_code:
            batch_feature = get_feature_semantic([d['total'] for d in new_batch_data], dataset_name)
            demo_feature = get_feature_semantic([d['total'] for d in new_demo_data], dataset_name)
        else:
            batch_feature = get_feature_semantic([d['question'] for d in new_batch_data], dataset_name)
            demo_feature = get_feature_semantic([d['question'] for d in new_demo_data], dataset_name)
    
    
    #计算每个问题对应的例子集合
    demo_dis = get_dis(batch_feature, demo_feature, dataset_name, sim_type)
    grpDemAff = assign_grpDemAff(demo_dis,upper_ques , k1,dataset_name) 
    
    
    #问题和问题之间的相似度
    #1.使用correlation clustering
    dis_batch = get_dis(batch_feature, batch_feature, dataset_name, sim_type)
    if batch_type == 'diverse':
        grpAff = np.where(dis_batch >= k0, 1, -1)
    else:
        grpAff = np.where(dis_batch <= k0, 1, -1)
    graph = nx.Graph()
    for i in range(len(grpAff)):
        for j in range(0,i):
            if grpAff[i][j] != grpAff[j][i]:
                exit(-1)
            if grpAff[i][j]==1:
                graph.add_edge(i,j,weight=1)
            else:
                graph.add_edge(i,j,weight=-1)    
    clusters, _ = cluster_correlation_search(graph, s = len(grpAff), max_attempts = 100, max_iters = 200)
    
    
    all_groups = []
    prompts = []
    new_batch_data_cost = [-1 for _ in range(len(new_batch_data))]
    new_demo_data_cost = [-1 for _ in range(len(new_demo_data))]
    for each_cluster in clusters.keys():
        ques = clusters[each_cluster] #每个cluster的问题
        demo_ques_list_for_each_cluster = {}
        demo_for_each_cluster = set() #每个cluster 覆盖问题的 例子集合
        for d in ques: #每个问题对应的list集合  d是问题的新索引
            for i in range(len(new_demo_data)): #i是例子的新索引
                if grpDemAff[d][i] == 1:
                    demo_for_each_cluster.add(i)
        for each_demo in demo_for_each_cluster:
            for d in ques:
                if grpDemAff[d][each_demo] == 1:
                    if each_demo not in demo_ques_list_for_each_cluster.keys():
                        demo_ques_list_for_each_cluster[each_demo] = set() 
                    demo_ques_list_for_each_cluster[each_demo].add(d)
        
        encoder = tiktoken.encoding_for_model(model)
        if is_dominated:
            demo_for_each_cluster = filter_dominated(demo_ques_list_for_each_cluster,new_demo_data,encoder,dataset_name)
        
        queCost = []
        for question in ques:
            if new_batch_data_cost[question] == -1:
                if dataset_name in dataset_for_code:
                    one_que_cost = get_cost(new_batch_data[question]['total'], encoder, dataset_name)
                else:
                    one_que_cost = get_cost(new_batch_data[question]['question'], encoder, dataset_name)
                queCost.append(one_que_cost)
                new_batch_data_cost[question] = one_que_cost
            else:
                queCost.append(new_batch_data_cost[question])

        
        demCost = []
        for question in demo_for_each_cluster:
            if new_demo_data_cost[question] == -1:
                if dataset_name in dataset_for_code:
                    one_demo_cost = get_cost(new_demo_data[question]['total'], encoder, dataset_name)
                else:
                    one_demo_cost = get_cost(new_demo_data[question]['question'], encoder, dataset_name)
                demCost.append(one_demo_cost)
                new_demo_data_cost[question] = one_demo_cost
            else:
                demCost.append(new_demo_data_cost[question])
    
        set_cost = []
        subsets = []
        demo_index = 0
        list_demo_for_each_cluster =[]
        for i in demo_for_each_cluster:
            one_set_cost = demCost[demo_index] #每个例子的cost
            demo_index+=1
            one_set = set()
            for j in ques:
                ques_index = 0
                if grpDemAff[j][i] == 1: #search 对应问题的cost
                    one_set_cost+=queCost[ques_index]
                    one_set.add(j)
                ques_index+=1
            set_cost.append(one_set_cost)
            subsets.append(one_set)
            list_demo_for_each_cluster.append(i)
        
        universe = set()
        for q in ques:
            universe.add(q)
        
        covers, indexes = set_cover(universe, subsets, set_cost)
        #2. 分配重复的问题  先求出每个set的平均问题数，使得分配后的每个set的问题数量尽可能相等---->这里考虑的是数量而不是cost
        each_set_cost_after_reassign = balance_sets_greedy(covers)
        
        #3. 调用bin packing的库
        items = {}
        index_bound = {}
        for i in range(len(each_set_cost_after_reassign)):
            if new_demo_data_cost[list_demo_for_each_cluster[indexes[i]]]==-1:
                if dataset_name in dataset_for_code:
                    one_demoCost = get_cost(new_demo_data[list_demo_for_each_cluster[indexes[i]]]['total'], encoder, dataset_name)
                else:
                    one_demoCost = get_cost(new_demo_data[list_demo_for_each_cluster[indexes[i]]]['question'], encoder, dataset_name)
                demo_cost = one_demoCost
                new_demo_data_cost[list_demo_for_each_cluster[indexes[i]]]=one_demoCost
            else:
                demo_cost = new_demo_data_cost[list_demo_for_each_cluster[indexes[i]]]
            que_cost = 0
            indexs=0
            is_bound = False
            reasssign_index = 0
            for each_ques in each_set_cost_after_reassign[i]:
                if (que_cost + demo_cost) > (k2-tCost): #例子对应的问题数太多 超过  阈值  就要split
                    items[str(i)+'_'+str(indexs)]=(demo_cost+que_cost)
                    if i not in index_bound.keys():
                        index_bound[i]=[]
                    index_bound[i].append(reasssign_index) #记录 哪些demo 对应的 ques在哪里 split
                    que_cost = 0
                    is_bound=True
                    indexs+=1
                if new_batch_data_cost[each_ques]==-1:
                    if dataset_name in dataset_for_code:
                        one_queCost = get_cost(new_batch_data[each_ques]['total'],encoder, dataset_name)
                    else:
                        one_queCost = get_cost(new_batch_data[each_ques]['question'],encoder, dataset_name)
                    que_cost += one_queCost
                    new_batch_data_cost[each_ques] = one_queCost
                else:
                    que_cost += new_batch_data_cost[each_ques]
                reasssign_index+=1
            if not is_bound:
                items[i] = demo_cost + que_cost
            if is_bound and que_cost!=0:
                if i not in index_bound.keys():
                    index_bound[i]=[]
                index_bound[i].append(reasssign_index)
                items[str(i)+'_'+str(indexs)]=(demo_cost+que_cost)
        bins = binpacking.to_constant_volume(items, k2-tCost)        
        
        #整合组合后的结果 并且将过滤后的问题 写入文件中
        for one_bin in bins: #one_bin 意味着一个group
            ori_ques_list = []
            ori_demo_list = set()
            for each_demo in one_bin.keys():
                if type(each_demo) == int:
                    ori_demo_list.add(list_demo_for_each_cluster[indexes[each_demo]]) 
                    for each_ques in each_set_cost_after_reassign[each_demo]:
                        ori_ques_list.append(each_ques)
                elif type(each_demo) == str:
                    int_each_demo = int(each_demo.split('_')[0])
                    indexss = int(each_demo.split('_')[1])
                    ori_demo_list.add(list_demo_for_each_cluster[indexes[int_each_demo]])
                    j=0
                    for one_item in range(len(each_set_cost_after_reassign[int_each_demo])):
                        if indexss == 0:
                            if j > index_bound[int_each_demo][indexss]:
                                break
                            else:
                                ori_ques_list.append(one_item)
                        else:
                            if j > index_bound[int_each_demo][indexss]:
                                break
                            elif j<=index_bound[int_each_demo][indexss-1]:
                                pass
                            else:
                                ori_ques_list.append(one_item)
                        j+=1
                            
                else: 
                    print('type error')
                    exit(-1)
        
            #Atlas 数据集加一些 例子
            if dataset_name in dataset_for_code:
                universe_per_group = set()
                for one_ques in ori_ques_list:
                    universe_per_group.add(one_ques)
            final_demo_list = []
            for one_demo in ori_demo_list:
                final_demo_list.append(new_demo_data_to_old[one_demo])
            final_batch_list =[]
            for one_batch in ori_ques_list:
                final_batch_list.append(new_batch_data_to_old[one_batch])
            one_group = {}
            if len(final_batch_list) != 0 and len(final_demo_list)!=0:
                one_group['ques'] = final_batch_list 
                one_group['demo'] = final_demo_list                  
                all_groups.append(one_group)
    with jsonlines.open(output_dir + '/' + dataset_name +'/'+batch_type+'_'+sim_type+'_'+str(k0)+'_'+str(k1)+'_'+str(k2)+'_'+str(upper_ques)+'_'+model+'_testpara_groups.json', 'w') as f_group:
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
        one_token = len(encoder.encode(prompt))
        tokens += one_token
        prompts.append([prompt, ans])
    with jsonlines.open(output_dir + '/' + dataset_name +'/'+batch_type+'_'+sim_type+'_'+str(k0)+'_'+str(k1)+'_'+str(k2)+'_'+str(upper_ques)+'_'+model+'_testpara_prompts.json', 'w') as f_prompt:
        f_prompt.write_all(prompts)
    return  prompts, tokens
        
    