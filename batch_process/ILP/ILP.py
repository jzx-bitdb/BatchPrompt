import json
import jsonlines
import tiktoken
import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from util_wxl import *
from util import *

def solve_MIQCP_gurobi_quicksum(n, m, M, grpAff, grpDemAff, queCost, demCost, tCost, k2, grpAff_flag=True, group_num = -1, cover_cnt = 1, k0=-1):
    model = gp.Model("ILP")
    
    
    if group_num == -1:
        group_num = n
    # x[i, j]表示问题j是否属于第i组
    x = model.addMVar((group_num, n), vtype=GRB.BINARY, name="x")
    # y[i, j]表示例子j是否出现在第i组
    y = model.addMVar((group_num, m), vtype=GRB.BINARY, name="y")
    u = model.addMVar(group_num, vtype=GRB.BINARY, name="u")
    
    # 每个问题只在一个group出现一次
    model.addConstrs(gp.quicksum(x[j, i] for j in range(group_num)) == 1 for i in range(n))
    if grpAff_flag:
        if k0 == -1:
            for j in range(group_num):
                for i in range(n):
                    for i_star in range(n):
                        if i_star<i:
                            model.addConstr(x[j,i]*x[j,i_star] <= grpAff[i][i_star])
        else:
            for j in range(group_num):
                for i in range(n):
                    for i_star in range(n):
                        if i_star<i:
                            model.addConstr(x[j,i]*x[j,i_star]*grpAff[i][i_star] <= k0)
    # 每个问题至少有一个例子cover
    for i in range(n):
        model.addConstr(gp.quicksum(grpDemAff[i][l]*x[j,i]*y[j,l] for j in range(group_num) for l in range(m)) >= cover_cnt)
    # 每个group少于tcost
    for j in range(group_num):
        model.addConstr(gp.quicksum(x[j,i]*queCost[i] for i in range(n))+gp.quicksum(y[j,l]*demCost[l] for l in range(m))+u[j]*tCost<=k2)
    for j in range(group_num):
        model.addConstr(u[j]-(gp.quicksum(x[j,i] for i in range(n)) + gp.quicksum(y[j,l] for l in range(m))) <= 0)
        model.addConstr(M*u[j]-(gp.quicksum(x[j,i] for i in range(n)) + gp.quicksum(y[j,l] for l in range(m)))>=0)
    
    model.setObjective(gp.quicksum(x[j,i]*queCost[i] for j in range(group_num) for i in range(n)) + gp.quicksum(y[j,l]*demCost[l] for j in range(group_num) for l in range(m)) + gp.quicksum(u[j]*tCost for j in range(group_num)), sense=GRB.MINIMIZE)
    
    model.optimize()
    if model.status == GRB.OPTIMAL:
        print("Objective value: {}".format(model.ObjVal))
    groups = {}
    group_num_real = 0
    flag = 0
    for i in range(group_num):
        flag = 0
        for j in range(n):
            if x[i][j].X != 0:
                if flag == 0:
                    group_num_real += 1
                    flag = 1
                    groups[group_num_real] = []
                groups[group_num_real].append(j)
    demos = {}
    group_num_real = 0
    flag = 0
    for i in range(group_num):
        flag = 0
        for j in range(m):
            if y[i][j].X != 0:
                if flag == 0:
                    group_num_real += 1
                    flag = 1
                    demos[group_num_real] = []
                demos[ group_num_real].append(j)
    return groups, demos

def ILP(dataset_name, tCost, k0, k1, k2, model, batch_type='diverse',sim_type='structure', batch=None, is_dominated=True, is_max_thre=False, group_num=-1, k3= 5, cover_cnt = 1, batch01_flag =True):
    """
    batch 和 demo 分布 分布BatchER 生成每个batch 和demo的索引--->这是baseline
    """
    batch_data = batch
    #data_preprocess.py 生成的数据

    demo_data = []
    with jsonlines.open('./data/demostrations/' + dataset_name + '/' + dataset_name + '.jsonl', 'r') as reader:
        for item in reader:
            demo_data.append(item)
    #问题数量定义
    num = len(batch_data)
    mini_demo = set()
    if sim_type == 'structure':
        batch_feature = get_feature_structure([d['question'] for d in batch_data], dataset_name)
        demo_feature = get_feature_structure([d['question'] for d in demo_data], dataset_name) 
    else:
        if 'Atlas' in dataset_name:
            batch_feature = get_feature_semantic([d['total'] for d in batch_data], dataset_name)
            demo_feature = get_feature_semantic([d['total'] for d in demo_data], dataset_name)
        else:
            batch_feature = get_feature_semantic([d['question'] for d in batch_data], dataset_name)
            demo_feature = get_feature_semantic([d['question'] for d in demo_data], dataset_name)
    
    #例子和问题之间的距离计算
    dis = get_dis(batch_feature, demo_feature, dataset_name, sim_type) 

    # 根据k1筛选demo
    demo_ques_list = {}
    for d in range(num):
        threshold = np.percentile(dis[d], 0.001) #从小到大排列 前0.1%
        max_thre = max(max_thre, threshold)
        indices = np.where(dis[d] <= k1)[0]
        for i in indices:
            mini_demo.add(i)
            if i not in demo_ques_list.keys():
                demo_ques_list[i] = set()
            demo_ques_list[i].add(d)
    
    if is_max_thre:
        demo_ques_list = {}
        mini_demo = set()
        for d in range(num):
            indices = np.where(dis[d] <= max_thre)[0]
            for i in indices:
                mini_demo.add(i)
                if i not in demo_ques_list.keys():
                    demo_ques_list[i] = set()
                demo_ques_list[i].add(d)
        
    dis_batch = get_dis(batch_feature, batch_feature, dataset_name, sim_type)
    if batch01_flag:
        if batch_type == 'diverse':
            grpAff = np.where(dis_batch >= k0, 1, 0)
        else:
            grpAff = np.where(dis_batch <= k0, 1, 0) 
        k0 = -1
        grpAff = grpAff.tolist()
    else:
        if batch_type == 'diverse':
            grpAff = (1-dis_batch).tolist()
            k0 = 1-k0
        else:
            grpAff = dis_batch.tolist()
    encoder = tiktoken.encoding_for_model("gpt-3.5-turbo-0125")
    if is_dominated:
        
        mini_demo = filter_dominated(demo_ques_list, demo_data, encoder, dataset_name)
        
    #mini_demo存的索引值 真正的demo是 demo_data[i]
    mini_demo_data = []
    for i in mini_demo:
        mini_demo_data.append(demo_data[i])
    
    #1.根据 阈值进行一次filter 删除超过阈值的数据--->因为例子是选的 最大阈值  DONE     
    #2.根据例子的dominated性质 ，过滤掉一些例子 
    if sim_type == 'structure':
        mini_demo_feature = get_feature_structure([d['question'] for d in mini_demo_data], dataset_name)
    else:
        if 'Atlas' in dataset_name:
            mini_demo_feature = get_feature_semantic([d['total'] for d in mini_demo_data], dataset_name)
        else:
            mini_demo_feature = get_feature_semantic([d['question'] for d in mini_demo_data], dataset_name)
    demo_dis_mini = get_dis(batch_feature, mini_demo_feature, dataset_name, sim_type)
    # 更改grpDEMAff
    num_rows, num_cols = demo_dis_mini.shape
    top_k_indices = np.argsort(demo_dis_mini, axis=0)[:k3]
    grpDemAff = np.zeros_like(demo_dis_mini)
    grpDemAff[top_k_indices, np.arange(num_cols)] = 1
    
    # 如果某一行1的数量不够cover_cnt 设demo_dis_mini剩余索引中最小的补全
    for i in range(num_rows):
        cnt = np.sum(grpDemAff[i] == 1)
        while cnt < cover_cnt:
            min_index = np.argmin(demo_dis_mini[i])
            grpDemAff[i][min_index] = 1
            demo_dis_mini[i][min_index] = np.inf
            cnt = np.sum(grpDemAff[i] == 1)
        # if cnt < cover_cnt:
        #     # 将这一行最小的值置为1
        #     min_index = np.argmin(demo_dis_mini[i])  
        #     grpDemAff[i][min_index] = 1 
             
    grpDemAff = grpDemAff.tolist()  #问题与例子的相似度
    key = 'total' if 'Atlas' in dataset_name else 'question'
    queCost = []
    for question in batch_data:
        queCost.append(get_cost(question[key], encoder, dataset_name))
        
    demCost = []
    for question in mini_demo_data:
        demCost.append(get_cost(question[key], encoder, dataset_name))
    
    
    
    n = num
    m = len(mini_demo_data)
    M = 30000000
    start = time.perf_counter()
    groups, demos = solve_MIQCP_gurobi_quicksum(n, m, M, grpAff, grpDemAff, queCost, demCost, tCost, k2, group_num=group_num, cover_cnt=cover_cnt, k0=k0)
    
    end = time.perf_counter()
    print("solve_MIQCP time:" + str(end - start))
    prompts = []
    tokens = 0
    for i in range(1, len(groups)+1):
        demos_data = []
        pairs_data = []
        ans = []
        for j in demos[i]:
            demos_data.append(mini_demo_data[j])
        for j in groups[i]:
            pairs_data.append(batch_data[j])
            ans.append(batch_data[j]['answer'])
        prompt = get_prompt(dataset_name, demos_data, pairs_data)
        tokens += len(encoder.encode(prompt))
        prompts.append([prompt, ans])
    return prompts, tokens
    

