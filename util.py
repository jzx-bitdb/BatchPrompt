import Levenshtein
import numpy as np
import pickle
import os
import json
import openai
from sentence_transformers import SentenceTransformer, util
import scipy
import re
import time
import copy
from difflib import SequenceMatcher
from suffix_trees import STree
import bisect

dataset_for_code = ['Atlas_assertEquals','Atlas_assertNotNull','Atlas_assertThat','Atlas_assertTrue']
dataset_for_er = ['AmazonGoogle', 'FodorsZagats','DblpScholar', 'DblpAcm', 'WalmartAmazon', 'AbtBuy', 'iTunesAmazon', 'Beer']

def get_feature_structure(questions, dataset_name):
    if dataset_name in dataset_for_er:
        features = []
        for q in questions:
            feature = []
            for k in q['entity1'].keys():
                sco = Levenshtein.ratio(q['entity1'][k], q['entity2'][k]) #计算编辑距离
                feature.append(sco)
            features.append(feature)
        return np.array(features)


def get_feature_semantic(questions, dataset_name):
    if dataset_name in dataset_for_code:
        model_name="st-codesearch-distilroberta-base"
        encode_model = SentenceTransformer(model_name)
        embeddings = encode_model.encode(questions, convert_to_tensor=True)
        return embeddings
    
def normalized_euclidean(X, Y):
    dis = np.linalg.norm((X - Y))
    if dis==0:
        return 0
    else:
        return 1/dis

def get_dis(batch_feature, demo_feature, dataset_name, sim_type):
    dis = np.zeros((batch_feature.shape[0], demo_feature.shape[0]))
    for i in range(batch_feature.shape[0]):
        dis[i] = np.linalg.norm(np.tile(batch_feature[i], (demo_feature.shape[0], 1))-demo_feature, axis=1)
    return dis
    

def search_range(disr, sorted_indices, end, start):
    #先用顺序搜索吧
    start_index = -1
    end_index = -1
    index = 0
    while index < len(sorted_indices) and (disr[sorted_indices[index]]>=end):
        index+=1
    if index ==len(sorted_indices):
        end_index = index - 1
    else:
        end_index = index
        
    index=len(sorted_indices)-1
    while index>=0 and (disr[sorted_indices[index]]<=start):
        index-=1
    if index==-1:
        start_index=0
    else:
        start_index=index
    
    return start_index, end_index

"""
利用triangle inequality减少计算的次数
<k_para 置为1
>k_para 置为0
要分别计算QDA 和QGA
"""        
def get_dist_opt_for_batch(batch_feature, demo_feature, k0, k1):
    #首先计算 问题和问题之间的similarity
    QGA_num = 0
    ori_QGA_num = len(batch_feature)*len(batch_feature)/2
    QDA_num = 0
    ori_QDA_num = len(batch_feature)*len(demo_feature)
    p = 0 #povit point
    dis_r = [0 for _ in range(len(batch_feature))]
    for i in range(1, len(batch_feature)):
        QGA_num+=1
        dis_r[i] = np.linalg.norm((batch_feature[p]-batch_feature[i]))
    #距离从小到大排序
    for i in range(1, len(batch_feature)):
        dis_ir = dis_r[i] #the distance between reference point
        tmp_disr = [dis_r[j] for j in range(i+1, len(batch_feature))] #point i 需要计算的次数
        tmp_sorted_indices = sorted(range(len(tmp_disr)), key=lambda k: tmp_disr[k])
        #范围是闭区间  所以 left_index和right_index不取等号
        left_index, right_index = search_range(tmp_disr, tmp_sorted_indices, dis_ir-k0, dis_ir+k0)
        if right_index>=left_index:
            QGA_num += (right_index-left_index+1)
    
    #计算 问题和例子之间的相似度
    dis_bathc_demo_r = [0 for _ in range(len(demo_feature))]
    for i in range(0, len(demo_feature)):
        QDA_num+=1
        dis_bathc_demo_r[i] = np.linalg.norm((batch_feature[p]-demo_feature[i]))
    sorted_indices = sorted(range(len(dis_bathc_demo_r)), key=lambda k: dis_bathc_demo_r[k])
    for i in range(1, len(batch_feature)):
        dis_ir = dis_r[i]
        left_index, right_index = search_range(dis_bathc_demo_r, sorted_indices, dis_ir-k1, dis_ir+k1)
        if right_index>=left_index:
            QDA_num += (right_index-left_index+1)
    return ori_QGA_num, QGA_num, ori_QDA_num, QDA_num
            
def get_cost_for_des(data, encoder):
    return len(encoder.encode(data))

def get_cost(data, encoder, dataset_name):
    if dataset_name in dataset_for_er:
        cost = 0
        str_total = ""
        for key in data['entity1'].keys():
            str_total += data['entity1'][key]+' '+data['entity2'][key]
        cost = len(encoder.encode(str_total))
        return cost
    elif dataset_name in dataset_for_code:
        return len(encoder.encode(data))

"""
demo_ques_list是一个以index为key, value是set
输出n个set的包含关系
"""
def filter_dominated(demo_ques_list,demo_data,encoder,dataset_name):
    copy_demo_ques_list = []
    for k,v in demo_ques_list.items():
        copy_demo_ques_list.append((k,v))
    
    min_demo_index = set()
    mini_demo_cost = {}
    for k in demo_ques_list.keys():
        if dataset_name in dataset_for_code:
            mini_demo_cost[k] = get_cost(demo_data[k]['total'], encoder, dataset_name)
        else:
            mini_demo_cost[k] = get_cost(demo_data[k]['question'], encoder, dataset_name)
    
    to_remove = set()
    
    for i in range(len(copy_demo_ques_list)):
        if i in to_remove:
            continue
        for j in range(i+1,len(copy_demo_ques_list)):
            if (copy_demo_ques_list[i][1].issubset(copy_demo_ques_list[j][1])) and (mini_demo_cost[copy_demo_ques_list[i][0]] >= mini_demo_cost[copy_demo_ques_list[j][0]]):   
                to_remove.add(i)                 

    print('len of to_remove:'+str(len(to_remove)))
    print('len of copy_demo_ques_list:'+str(len(copy_demo_ques_list)))
    for i in range(len(copy_demo_ques_list)):
        if i not in to_remove:
            min_demo_index.add(copy_demo_ques_list[i][0])
    return min_demo_index

def get_entity(data):
    entity = ''
    for k in data.keys():
        entity = entity + '"' + k + '": ' + '"' + data[k] + '"' + ', '
    return entity[:-2]

def filter_demos(dataset_name, real_batch_data, demo_data, k1, upper_ques, sim_type, batch_index):
    #只过滤 问题
    batch_data = []
    for i in range(len(batch_index)):
        batch_data.append(real_batch_data[batch_index[i]])
        
    demo_num = len(demo_data)
    demo_data_index = []
    demo_data_filter = [False for _ in range(demo_num)]
    if sim_type == 'structure':
        batch_feature = get_feature_structure([d['question'] for d in batch_data], dataset_name)
        demo_feature = get_feature_structure([d['question'] for d in demo_data], dataset_name)
    else:
        if dataset_name in dataset_for_code:
            batch_feature = get_feature_semantic([d['total'] for d in batch_data], dataset_name)
            demo_feature = get_feature_semantic([d['total'] for d in demo_data], dataset_name)
        else:
            batch_feature = get_feature_semantic([d['question'] for d in batch_data], dataset_name)
            demo_feature = get_feature_semantic([d['question'] for d in demo_data], dataset_name)

    print(len(demo_feature))
    print(len(batch_feature))
    print('before get_dist')
    batch_demo_dis = get_dis(demo_feature, batch_feature, dataset_name, sim_type) 
    print('after get_dist')
    batch_demo_dis = batch_demo_dis.tolist()
    
    #遍历每个例子 找出最相似的upper_ques个问题
    for i in range(demo_num):
        #应该不会存在 问题没有相似例子的情况
        sorted_indices = sorted(range(len(batch_demo_dis[i])), key=lambda k: batch_demo_dis[i][k])
        for j in range(upper_ques):
            if (j < len(sorted_indices)) and (batch_demo_dis[i][sorted_indices[j]] <= k1):
                demo_data_filter[i] = True
    for j in range(demo_num):
        if demo_data_filter[j] is True:
            demo_data_index.append(j)
     
    return demo_data_index

def filter_demos_for_code(dataset_name, batch_data, demo_data, k1, sim_type, groups):
    #首先 过滤例子   然后  确定每个例子对应的set
        
    ques_num = len(batch_data)
    demo_num = len(demo_data)
    demo_data_index = set()
    demo_data_filter = [False for _ in range(demo_num)]
    if sim_type == 'structure':
        batch_feature = get_feature_structure([d['question'] for d in batch_data], dataset_name)
        demo_feature = get_feature_structure([d['question'] for d in demo_data], dataset_name)
    else:
        if dataset_name in dataset_for_code:
            batch_feature = get_feature_semantic([d['total'] for d in batch_data], dataset_name)
            demo_feature = get_feature_semantic([d['total'] for d in demo_data], dataset_name)
        else:
            batch_feature = get_feature_semantic([d['question'] for d in batch_data], dataset_name)
            demo_feature = get_feature_semantic([d['question'] for d in demo_data], dataset_name)

    print(len(demo_feature))
    print(len(batch_feature))
    print('before get_dist')
    batch_demo_dis = get_dis(demo_feature, batch_feature, dataset_name, sim_type) 
    print('after get_dist')
    batch_demo_dis = batch_demo_dis.tolist() #例子-->问题的 距离计算
    
    #过滤出相似的  有相似的例子   不考虑 upper_bounds
    for i in range(demo_num):
        if i%10000 == 0:
            print('index:'+str(i))
        #应该不会存在 问题没有相似例子的情况
        sorted_indices = sorted(range(len(batch_demo_dis[i])), key=lambda k: batch_demo_dis[i][k])
        for j in range(len(sorted_indices)):
            if batch_demo_dis[i][sorted_indices[j]] <= k1:
                demo_data_filter[i] = True
    for j in range(demo_num):
        if demo_data_filter[j] is True:
            demo_data_index.add(j)
    
    demo_coverage_ques_for_each_group = {}
    group_index = 0
    for one_group in groups: #每个 group中  例子覆盖的问题 统计
        demo_coverage_ques_for_each_group[group_index] = {} #key:例子  value:问题   每个group
        #one_group 每个问题的id
        for one_que in one_group:  #每个group中的question
            for j in range(demo_num):  
                if batch_demo_dis[j][one_que]<=k1 and demo_data_filter[j] is True:
                    if j not in demo_coverage_ques_for_each_group[group_index].keys():
                        demo_coverage_ques_for_each_group[group_index][j] = set()
                    demo_coverage_ques_for_each_group[group_index][j].add(one_que)
        group_index += 1
     
    return demo_data_index, demo_coverage_ques_for_each_group

def filter_ques_demos(dataset_name, real_batch_data, demo_data, k1, upper_ques, sim_type, encoder):
    batch_data = []
    batch_data = copy.deepcopy(real_batch_data)
    ques_num = len(batch_data)
    demo_num = len(demo_data)
    batch_data_index =[]
    demo_data_index = []
    batch_data_filter = [False for _ in range(ques_num)]
    demo_data_filter = [False for _ in range(demo_num)]
    if sim_type == 'structure':
        batch_feature = get_feature_structure([d['question'] for d in batch_data], dataset_name)
        demo_feature = get_feature_structure([d['question'] for d in demo_data], dataset_name)
    else:
        if dataset_name in dataset_for_code:
            batch_feature = get_feature_semantic([d['total'] for d in batch_data], dataset_name)
            demo_feature = get_feature_semantic([d['total'] for d in demo_data], dataset_name)
        else:
            batch_feature = get_feature_semantic([d['question'] for d in batch_data], dataset_name)
            demo_feature = get_feature_semantic([d['question'] for d in demo_data], dataset_name)

    batch_demo_dis = get_dis(demo_feature, batch_feature, dataset_name, sim_type) 
    batch_demo_dis = batch_demo_dis.tolist()
    
    
    #遍历每个例子 找出最相似的upper_ques个问题
    for i in range(demo_num):
        sorted_indices = sorted(range(len(batch_demo_dis[i])), key=lambda k: batch_demo_dis[i][k])
        for j in range(upper_ques):
            if (j < len(sorted_indices)) and (batch_demo_dis[i][sorted_indices[j]] <= k1):
                batch_data_filter[sorted_indices[j]] = True
                demo_data_filter[i] = True

    for i in range(ques_num):
        if batch_data_filter[i] is True:
            batch_data_index.append(i)
    for j in range(demo_num):
        if demo_data_filter[j] is True:
            demo_data_index.append(j)
    return batch_data_index, demo_data_index

def assign_grpDemAff(dis_batch_demo,upper_ques, k1,dataset_name):
    dis_batch_demo_list = dis_batch_demo.tolist()
    
    ret_batch_demo_list = [[0 for _ in range(len(dis_batch_demo_list[0]))] for _ in range(len(dis_batch_demo_list))]
    for i in range(len(dis_batch_demo_list[0])): #例子数
        ques_dis_for_one_demo = [row[i] for row in dis_batch_demo_list] #例子 i对应的所有问题
        if i % 10000==0:
            print('index:'+str(i))
        sorted_indices = sorted(range(len(ques_dis_for_one_demo)), key=lambda k: ques_dis_for_one_demo[k]) #同一例子的问题排序
        if dataset_name in dataset_for_code: #代码数据集 Atlas 这里不限制 upper_ques个数  否则会导致每次生成的结果都不一样
            for j in range(len(sorted_indices)):
                if dis_batch_demo_list[sorted_indices[j]][i] <= k1:
                    ret_batch_demo_list[sorted_indices[j]][i]=1
        else:
            for j in range(upper_ques):
                if j<len(sorted_indices) and dis_batch_demo_list[sorted_indices[j]][i] <= k1:
                    ret_batch_demo_list[sorted_indices[j]][i]=1
    
    #验证: 保证每个问题 都有相似的例子
    for i in range(len(ret_batch_demo_list)):
        have_demo = False
        for one_demo in ret_batch_demo_list[i]:
            if one_demo==True:
                have_demo = True
                break
        if not have_demo:
            print('assign_grpDemAff error: question '+str(i)+' do not have similar demonstrations')
            exit(-1)
    return ret_batch_demo_list

def get_prompt(dataset_name, demos, pairs):
    if dataset_name in dataset_for_er:
        item_type = "Product"
        task_description = "When determining whether two Restraunts are the same, you should only focus on critical properties and overlook noisy factors."
        demo_txt = []

        for i, d in enumerate(demos):
            demo_txt.append(
                f'Demonstration {i+1}:\n{item_type} A is {get_entity(d["question"]["entity1"])}\n{item_type} B is {get_entity(d["question"]["entity2"])}')
            if d["answer"] == '1':
                demo_txt.append(
                    f'Yes, {item_type} A and {item_type} B are the same {item_type.lower()}.')
            else:
                demo_txt.append(
                    f'No, {item_type} A and {item_type} B are different {item_type.lower()}s.')

        demo_prompt = '\n'.join(demo_txt)

        pair_txt = []

        for i, p in enumerate(pairs):
            pair_txt.append(
                f'Question {i+1}:\n {item_type} A is {get_entity(p["question"]["entity1"])}\n {item_type} B is {get_entity(p["question"]["entity2"])}')

        pcnt = len(pairs)
        sent = 'question above' if pcnt == 1 else f'above {pcnt} questions'
        pair_txt.append(f'\nUse domain knowledge of {item_type}s to help understand the text and answer the {sent} in the format: For Question i, Yes, {item_type} A and {item_type} B are the same {item_type.lower()}./No, {item_type} A and {item_type} B are different {item_type.lower()}s. For Question i+1, (repeat the above procedures)')
        pair_prompt = '\n'.join(pair_txt)

        prompt = '\n\n'.join([task_description, demo_prompt, pair_prompt])
        return prompt
    elif dataset_name in dataset_for_code:
        task_description = '''You are a professional Java expert at generating meaningful assert statement for test method. 
        Given a test method and a total method , give a assert statement to assess the correctness of the total method.
        <AssertPlaceHolder> is the assert statement being masked.There are several Examples for reference. Don't answer questions in the Examples.
        Output assert statements for each Question below Examples in JSON like this:
        {
            "Question1":...,
            "Question2":...,
            "Question3":...,
            ...
        }
        Examples:
        '''
        prompt = task_description
        for i in range(len(demos)):
            prompt += "testMethod and totalMethod:\n"
            prompt += demos[i]['total']
            prompt += '\n'
            prompt += "assertLine:\n"
            prompt += demos[i]['answer']
            prompt += '\n\n'
        prompt += 'Questions:\n'
        labels = []
        for j in range(len(pairs)):
            prompt += f'Question{j+1}:\n'
            prompt += "testMethod and totalMethod:\n"
            prompt += pairs[j]['total']
            prompt += '\n'
        return prompt
    elif dataset_name in dataset_for_qa:
        task_description = '''Answer common sense questions based on your knowledge, only yes or no.There are several questions below Examples,return answers in JSON like this:
{
    "Question1":yes/no,
    "Question2":yes/no,
    ...
}
Examples:
'''
        prompt = task_description
        for i in range(len(demos)):
            prompt += "Question:"
            prompt += demos[i]['question']
            prompt += '\n'
            prompt += "Answer:"
            prompt += 'yes' if demos[i]['answer'] == True else 'no'
            prompt += '\n'
        prompt += 'Questions:\n'
        for j in range(len(pairs)):
            prompt += f'Question{j+1}:\n'
            prompt += pairs[j]['question']
            prompt += '\n'
        return prompt     
        

def askLLM(prompt, dataset_name, model='gpt-3.5-turbo-0125'):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if dataset_name in dataset_for_er:
        completion = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
    elif dataset_name in dataset_for_code:
        if model == 'gpt-3.5-turbo-0301':
            completion = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,)            
        else:
            completion = openai.ChatCompletion.create(
                model=model,
                response_format={ "type": "json_object" }, # json格式返回
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
    ans = completion.choices[0].message['content']
    return ans


def askLLMandEval(dataset_name, prompts, model):
    if dataset_name in dataset_for_er:
        tp, fp, tn, fn = 0, 0, 0, 0
        for i in range(len(prompts)):
            print('index:'+str(i)+' total:'+str(len(prompts)))
            llm_ans = []
            ans = prompts[i][1]
            res = askLLM(prompts[i][0], dataset_name, model)
            res = res.split('\n')
            for s in res:
                if 'same' in s:
                    llm_ans.append(1)
                elif 'different' in s:
                    llm_ans.append(0)
                else:
                    continue
            for i in range(len(llm_ans)):
                if llm_ans[i] == 1:
                    if i>=len(ans):
                        break
                    if int(ans[i]) == 1:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if i>=len(ans):
                        break
                    if int(ans[i]) == 1:
                        fn += 1
                    else:
                        tn += 1
            print(llm_ans)
            print(ans)
            print(tp, fp, tn, fn)
            print('tp:'+str(tp)+'  fp:'+str(fp)+' tn:'+str(tn)+'  fn:'+str(fn))
            if (tp + fp + tn + fn) != 0:
                print("acc:", str(float(tp + tn)/ (tp + fp + tn + fn)))
            else:
                print("acc:0")
            if (tp + fp) != 0:
                print("precision:", str(float(tp) / (tp + fp)))
            else:
                print("precision: 0")
            if (tp + fn) != 0:
                print("recall:", str(float(tp) / (tp + fn)))
            else:
                print("recall: 0")
            if (2*tp + fp + fn)!=0:
                print("local f1:", str(float(2*tp) / (2*tp + fp + fn)))
            else:
                print("local f1:0")
            
        print('tp:'+str(tp)+'  fp:'+str(fp)+' tn:'+str(tn)+'  fn:'+str(fn))
        if (tp + fp + tn + fn) != 0:
            print("acc:", str(float(tp + tn)/ (tp + fp + tn + fn)))
        else:
            print("acc:0")
        if (tp + fn) != 0:
            print("precision:", str(float(tp) / (tp + fp)))
            print("recall:", str(float(tp) / (tp + fn)))
        else:
            print("precision: 0")
            print("recall: 0")
        if (2*tp + fp + fn)!=0:
            print("total f1:", str(float(2*tp) / (2*tp + fp + fn)))
        else:
            print("total f1:0")
    elif dataset_name in dataset_for_code:
        t, num = 0, 0
        for i in range(len(prompts)):
            ans = prompts[i][1]
            res = askLLM(prompts[i][0], dataset_name, model)
            
            try:
                llm_ans = json.loads(res)
            except:
                continue
            tmp_t = 0 
            for j in range(len(ans)):
                ques = 'Question'+str(j+1)
                if ques not in llm_ans.keys():
                    continue
                
                llm_ans_filter = llm_ans[f'Question{j+1}'].rstrip(';').replace(' ', '')
                ans_filter = ans[j].replace(' ', '')
                if llm_ans_filter == ans_filter:
                    t += 1
                    tmp_t += 1
            num += len(ans)
            print('llm_ans:')
            print(llm_ans)
            print('ans:')
            print(ans)
            print('number of questions:'+str(len(ans)))
            if len(ans)!=0:
                print('local acc:', str(float(tmp_t) / len(ans)))
            if num!=0:
                print('acc: ', str(float(t) / num))
        if num!=0:
            print('total acc: ', str(float(t) / num))