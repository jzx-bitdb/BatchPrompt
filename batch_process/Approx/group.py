import json
import jsonlines
import tiktoken
import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from util import *
from batch_process.Approx.correlation import cluster_correlation_search
from sklearn.cluster import DBSCAN
import networkx as nx

def solve_MIQCP_gurobi_quicksum(n, m, M, grpAff, grpDemAff, queCost, demCost, tCost, k2, grpAff_flag=True):
    model = gp.Model("ILP")
    
    
    # x[i, j]表示问题j是否属于第i组
    x = model.addMVar((n, n), vtype=GRB.BINARY, name="x")
    # y[i, j]表示例子j是否出现在第i组
    y = model.addMVar((n, m), vtype=GRB.BINARY, name="y")
    z = model.addMVar((n, n, m), vtype=GRB.BINARY, name="z")
    u = model.addMVar(n, vtype=GRB.BINARY, name="u")
    
    # 每个问题只在一个group出现一次
    model.addConstrs(gp.quicksum(x[j, i] for j in range(n)) == 1 for i in range(n))
    if grpAff_flag:
        for j in range(n):
            for i in range(n):
                for i_star in range(n):
                    if i_star<i:
                        model.addConstr(x[j,i]*x[j,i_star] <= grpAff[i][i_star])
    # 每个问题至少有一个例子cover
    for i in range(n):
        model.addConstr(gp.quicksum(grpDemAff[i][l]*x[j,i]*y[j,l] for j in range(n) for l in range(m)) >= 1)
    # 每个group少于tcost
    for j in range(n):
        model.addConstr(gp.quicksum(x[j,i]*queCost[i] for i in range(n))+gp.quicksum(y[j,l]*demCost[l] for l in range(m))+u[j]*tCost<=k2)
    for j in range(n):
        # model.addGenConstrIndicator(u[j], True, gp.quicksum(x[j,i] for i in range(n)) + gp.quicksum(y[j,l] for l in range(m)) >= 1)
        model.addConstr(u[j]-(gp.quicksum(x[j,i] for i in range(n)) + gp.quicksum(y[j,l] for l in range(m))) <= 0)
        model.addConstr(M*u[j]-(gp.quicksum(x[j,i] for i in range(n)) + gp.quicksum(y[j,l] for l in range(m)))>=0)
    
    model.setObjective(gp.quicksum(x[j,i]*queCost[i] for j in range(n) for i in range(n)) + gp.quicksum(y[j,l]*demCost[l] for j in range(n) for l in range(m)) + gp.quicksum(u[j]*tCost for j in range(n)), sense=GRB.MINIMIZE)
    
    model.optimize()
    if model.status == GRB.OPTIMAL:
        print("Objective value: {}".format(model.ObjVal))
    groups = {}
    group_num = 0
    flag = 0
    for i in range(n):
        flag = 0
        for j in range(n):
            if x[i][j].X != 0:
                if flag == 0:
                    group_num += 1
                    flag = 1
                    groups[group_num] = []
                groups[group_num].append(j)
    demos = {}
    group_num = 0
    flag = 0
    for i in range(n):
        flag = 0
        for j in range(m):
            if y[i][j].X != 0:
                if flag == 0:
                    group_num += 1
                    flag = 1
                    demos[group_num] = []
                demos[group_num].append(j)
    with open('groups.json', 'w') as f:
        f.write(json.dumps(groups, indent=4))
    with open('demos.json', 'w') as f:
        f.write(json.dumps(demos, indent=4))
    model.write("lp_ex_without96.sol")
    return groups, demos


"""
correlation clustering + ILP
"""
def approx_grouping(dataset_name, tCost, upper_ques, k0, k1, k2, output_dir, batch_type='diverse',sim_type='structure', batch_data=None, demo_data=None, is_dominated=True):
    batch_data_index, demo_data_index, _ = filter_ques_demos(dataset_name, batch_data, demo_data, k1, upper_ques, sim_type)
    new_batch_data = []
    new_demo_data = []
    for i in batch_data_index:
        new_batch_data.append(batch_data[i])
    for j in demo_data_index:
        new_demo_data.append(demo_data[j])
    
    if sim_type == 'structure':
        batch_feature = get_feature_structure([d['question'] for d in new_batch_data], dataset_name)
        demo_feature = get_feature_structure([d['question'] for d in new_demo_data], dataset_name) 
    else:
        batch_feature = get_feature_semantic([d['question'] for d in batch_data], dataset_name)
        demo_feature = get_feature_semantic([d['question'] for d in demo_data], dataset_name)
    
    #计算每个问题对应的例子集合
    demo_dis = get_dis(batch_feature, demo_feature, dataset_name, sim_type)
    grpDemAff = assign_grpDemAff(demo_dis,upper_ques , k1, dataset_name) 
        
    #1.问题和问题之间的相似度
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

    prompts = []
    all_prompts=[]
    #调用每个聚类调用ILP
    for each_cluster in clusters.keys():
        ques = clusters[each_cluster] #每个cluster的问题
        demo_ques_list_for_each_cluster = {}
        demo_for_each_cluster = set() #每个cluster 覆盖问题的 例子集合
        for d in ques: #每个问题对应的list集合  d是问题的索引
            for i in range(len(new_demo_data)):
                if grpDemAff[d][i] == 1:
                    demo_for_each_cluster.add(i)
        for each_demo in demo_for_each_cluster:
            for d in ques:
                if grpDemAff[d][each_demo] == 1:
                    if each_demo not in demo_ques_list_for_each_cluster.keys():
                        demo_ques_list_for_each_cluster[each_demo] = set() 
                    demo_ques_list_for_each_cluster[each_demo].add(d)
                    
        if is_dominated:
            encoder = tiktoken.encoding_for_model("gpt-3.5-turbo-0125")
            demo_for_each_cluster = filter_dominated(demo_ques_list_for_each_cluster,new_demo_data,encoder,dataset_name)
        mini_demo_data = []
        for i in demo_for_each_cluster:
            mini_demo_data.append(new_demo_data[i])
            
        each_cluster_feature = get_feature_structure([new_batch_data[one_q]['question'] for one_q in ques], dataset_name)
        mini_demo_feature = get_feature_structure([d['question'] for d in mini_demo_data], dataset_name)
        demo_dis_mini = get_dis(each_cluster_feature, mini_demo_feature, dataset_name, sim_type)
        grpDemAffEachCluster = assign_grpDemAff(demo_dis_mini,upper_ques , k1)
        queCost = []
        for question in ques:
            queCost.append(get_cost(new_batch_data[question]['question'], encoder, dataset_name))
        
        demCost = []
        for question in mini_demo_data:
            demCost.append(get_cost(question['question'], encoder, dataset_name))
    
        
        #调用ILP
        n = len(ques)
        m = len(mini_demo_data)
        M = 3000000000
    
        start = time.perf_counter()
        groups, demos = solve_MIQCP_gurobi_quicksum(n, m, M, None, grpDemAffEachCluster, queCost, demCost, tCost, k2, grpAff_flag=False)

        end = time.perf_counter()
        print(end - start)
        # prompts = []
        tokens = 0
        for i in range(1, len(groups)+1):
            demos_data = []
            pairs_data = []
            ans = []
            for j in demos[i]:
                demos_data.append(mini_demo_data[j])
            for j in groups[i]:
                pairs_data.append(batch_data[j])
                ans.append(int(batch_data[j]['answer']))
            prompt = get_prompt(dataset_name, demos_data, pairs_data)
            tokens += len(encoder.encode(prompt))
            prompts.append([prompt, ans])
        with jsonlines.open(output_dir + '/' + dataset_name + '_approx_group_prompts.json', 'w') as f_prompt:
            f_prompt.write_all(prompts)
        all_prompts.append(prompts)
    return all_prompts, tokens
    