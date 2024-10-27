import Levenshtein
import numpy as np
import json
import openai
import anthropic
import random
from http import HTTPStatus
import dashscope
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import copy
from time import sleep

dataset_for_code = ['Atlas_assertEquals','Atlas_assertNotNull','Atlas_assertThat','Atlas_assertTrue']
dataset_for_er = ['AmazonGoogle', 'FodorsZagats','DblpScholar', 'DblpAcm', 'WalmartAmazon', 'AbtBuy', 'iTunesAmazon', 'Beer']
dataset_for_qa = ['StrategyQA','CommonsenseQA']
dataset_for_jd = ['DXF','FFAJ', 'DXF_ILP', 'FFAJ_ILP']

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
    if dataset_name in dataset_for_code or dataset_name in dataset_for_jd:
        model_name="st-codesearch-distilroberta-base"
        encode_model = SentenceTransformer('./pretrain_model/' + model_name)
        embeddings = encode_model.encode(questions, convert_to_tensor=True)
        return embeddings
    elif dataset_name in dataset_for_qa:
        model_name = 'st-codesearch-distilroberta-base'
        encode_model = SentenceTransformer('./pretrain_model/' + model_name)
        embeddings = encode_model.encode(questions, convert_to_tensor=True)
        return embeddings
        
    
    
def normalized_euclidean(X, Y):
    dis = np.linalg.norm((X - Y))
    if dis==0:
        return 0
    else:
        return 1/dis

def normalized_euclidean_opt(X, Y):
    indexy = Y[-1]
    print(str(indexy)+'  '+str(X[indexy]))
    return X[indexy]

def get_dis(batch_feature, demo_feature, dataset_name, sim_type):
    if (dataset_name in dataset_for_er) and sim_type == 'structure':
        dis = np.zeros((batch_feature.shape[0], demo_feature.shape[0]))
        for i in range(batch_feature.shape[0]):
            dis[i] = np.linalg.norm(np.tile(batch_feature[i], (demo_feature.shape[0], 1))-demo_feature, axis=1)
        return dis
    elif (dataset_name in dataset_for_code  or dataset_name in dataset_for_qa or dataset_name in dataset_for_jd) and sim_type == 'semantic':
        print('for gpu')
        cosine_similarity = cos_sim(batch_feature, demo_feature)
        return (1 - cosine_similarity).cpu().numpy()
    
    
def get_dis_for_batch(batch_feature, dataset_name, sim_type):
    if (dataset_name in dataset_for_er) and sim_type == 'structure':
        dis = np.zeros((batch_feature.shape[0], batch_feature.shape[0]))
        for i in range(batch_feature.shape[0]):
            tmp_dis = np.linalg.norm(np.tile(batch_feature[i], (batch_feature.shape[0], 1))-batch_feature, axis=1)
            if tmp_dis<=0.0:
                dis[i] =0 
            else:
                dis[i] = 1/tmp_dis
        return dis
    

def search_range(disr, sorted_indices, end, start):
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


def get_dist_opt_for_batch(batch_feature, demo_feature, k0, k1):
    QGA_num = 0
    ori_QGA_num = len(batch_feature)*len(batch_feature)/2
    QDA_num = 0
    ori_QDA_num = len(batch_feature)*len(demo_feature)
    p = 0 #povit point
    dis_r = [0 for _ in range(len(batch_feature))]
    for i in range(1, len(batch_feature)):
        QGA_num+=1
        dis_r[i] = np.linalg.norm((batch_feature[p]-batch_feature[i]))
    for i in range(1, len(batch_feature)):
        dis_ir = dis_r[i] #the distance between reference point
        tmp_disr = [dis_r[j] for j in range(i+1, len(batch_feature))] 
        tmp_sorted_indices = sorted(range(len(tmp_disr)), key=lambda k: tmp_disr[k])
        left_index, right_index = search_range(tmp_disr, tmp_sorted_indices, dis_ir-k0, dis_ir+k0)
        if right_index>=left_index:
            QGA_num += (right_index-left_index+1)

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
        for key in data['entity1'].keys():
            cost += len(encoder.encode(data['entity1'][key]))
            cost += len(encoder.encode(data['entity2'][key]))
        return cost
    elif dataset_name in dataset_for_code or dataset_name in dataset_for_qa or dataset_name in dataset_for_jd:
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

def filter_demos_for_er(dataset_name, real_batch_data, demo_data, k1, upper_ques, sim_type, batch_index):
    batch_data = []
    for i in range(len(batch_index)):
        batch_data.append(real_batch_data[batch_index[i]])
        
    ques_num = len(batch_data)
    demo_num = len(demo_data)
    demo_data_index = []
    demo_data_filter = [False for _ in range(demo_num)]
    if sim_type == 'structure':
        batch_feature = get_feature_structure([d['question'] for d in batch_data], dataset_name)
        demo_feature = get_feature_structure([d['question'] for d in demo_data], dataset_name)
    else:
        batch_feature = get_feature_semantic([d['question'] for d in batch_data], dataset_name)
        demo_feature = get_feature_semantic([d['question'] for d in demo_data], dataset_name)

    print(len(demo_feature))
    print(len(batch_feature))
    print('before get_dist')
    batch_demo_dis = get_dis(demo_feature, batch_feature, dataset_name, sim_type) 
    print('after get_dist')
    batch_demo_dis = batch_demo_dis.tolist()
    
    for i in range(demo_num):
        if i%10000 == 0:
            print('index:'+str(i))
        sorted_indices = sorted(range(len(batch_demo_dis[i])), key=lambda k: batch_demo_dis[i][k])
        for j in range(upper_ques):
            if (j < len(sorted_indices)) and (batch_demo_dis[i][sorted_indices[j]] <= k1):
                demo_data_filter[i] = True
    for j in range(demo_num):
        if demo_data_filter[j] is True:
            demo_data_index.append(j)
     
    return demo_data_index

def filter_demos_for_cg(dataset_name, real_batch_data, demo_data, k1, sim_type, batch_index):
    batch_data = []
    for i in range(len(batch_index)):
        batch_data.append(real_batch_data[batch_index[i]])
        
    ques_num = len(batch_data)
    demo_num = len(demo_data)
    demo_data_index = []
    demo_data_filter = [False for _ in range(demo_num)]
    if sim_type == 'structure':
        batch_feature = get_feature_structure([d['total'] for d in batch_data], dataset_name)
        demo_feature = get_feature_structure([d['total'] for d in demo_data], dataset_name)
    else:
        batch_feature = get_feature_semantic([d['total'] for d in batch_data], dataset_name)
        demo_feature = get_feature_semantic([d['total'] for d in demo_data], dataset_name)

    print(len(demo_feature))
    print(len(batch_feature))
    print('before get_dist')
    batch_demo_dis = get_dis(demo_feature, batch_feature, dataset_name, sim_type) 
    print('after get_dist')
    batch_demo_dis = batch_demo_dis.tolist()
    
    for i in range(demo_num):
        if i%10000 == 0:
            print('index:'+str(i))
        sorted_indices = sorted(range(len(batch_demo_dis[i])), key=lambda k: batch_demo_dis[i][k])
        for j in range(len(sorted_indices)):
            if batch_demo_dis[i][sorted_indices[j]] <= k1:
                demo_data_filter[i] = True
    for j in range(demo_num):
        if demo_data_filter[j] is True:
            demo_data_index.append(j)
     
    return demo_data_index


def filter_demos_for_code(dataset_name, batch_data, demo_data, k1, sim_type, groups):        
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
    batch_demo_dis = batch_demo_dis.tolist()
    
    for i in range(demo_num):
        if i%10000 == 0:
            print('index:'+str(i))
        sorted_indices = sorted(range(len(batch_demo_dis[i])), key=lambda k: batch_demo_dis[i][k])
        for j in range(len(sorted_indices)):
            if batch_demo_dis[i][sorted_indices[j]] <= k1:
                demo_data_filter[i] = True
    for j in range(demo_num):
        if demo_data_filter[j] is True:
            demo_data_index.add(j)
    
    demo_coverage_ques_for_each_group = {}
    group_index = 0
    for one_group in groups:
        demo_coverage_ques_for_each_group[group_index] = {} 
        for one_que in one_group:  
            for j in range(demo_num):  
                if batch_demo_dis[j][one_que]<=k1 and demo_data_filter[j] is True:
                    if j not in demo_coverage_ques_for_each_group[group_index].keys():
                        demo_coverage_ques_for_each_group[group_index][j] = set()
                    demo_coverage_ques_for_each_group[group_index][j].add(one_que)
        group_index += 1
    return demo_data_index, demo_coverage_ques_for_each_group

def filter_ques_demos(dataset_name, real_batch_data, demo_data, k1, upper_ques, sim_type, encoder, batch_index=None):
    batch_data = []
    if dataset_name in dataset_for_code: 
        for i in range(len(batch_index)):
            batch_data.append(real_batch_data[batch_index[i]])
    else:
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
        elif dataset_name in dataset_for_er or dataset_name in dataset_for_jd or dataset_name in dataset_for_qa:
            batch_feature = get_feature_semantic([d['question'] for d in batch_data], dataset_name)
            demo_feature = get_feature_semantic([d['question'] for d in demo_data], dataset_name)

    batch_demo_dis = get_dis(demo_feature, batch_feature, dataset_name, sim_type) 
    batch_demo_dis = batch_demo_dis.tolist()
    
    for i in range(demo_num):
        if i%10000 == 0:
            print('index:'+str(i))
        sorted_indices = sorted(range(len(batch_demo_dis[i])), key=lambda k: batch_demo_dis[i][k])
        for j in range(upper_ques):
            if (j < len(sorted_indices)) and (batch_demo_dis[i][sorted_indices[j]] <= k1):
                batch_data_filter[sorted_indices[j]] = True
                demo_data_filter[i] = True

    for i in range(ques_num):
        if batch_data_filter[i] is True:
            if dataset_name in dataset_for_code:
                batch_data_index.append(batch_index[i]) 
            else:
                batch_data_index.append(i)
    for j in range(demo_num):
        if demo_data_filter[j] is True:
            demo_data_index.append(j)
     
    return batch_data_index, demo_data_index

def assign_grpDemAff(dis_batch_demo,upper_ques, k1,dataset_name):
    dis_batch_demo_list = dis_batch_demo.tolist()
    
    ret_batch_demo_list = [[0 for _ in range(len(dis_batch_demo_list[0]))] for _ in range(len(dis_batch_demo_list))]
    for i in range(len(dis_batch_demo_list[0])): 
        ques_dis_for_one_demo = [row[i] for row in dis_batch_demo_list]
        if i % 10000==0:
            print('index:'+str(i))
        sorted_indices = sorted(range(len(ques_dis_for_one_demo)), key=lambda k: ques_dis_for_one_demo[k])
        if dataset_name in dataset_for_code or dataset_name in dataset_for_jd or dataset_name in dataset_for_qa: 
            for j in range(len(sorted_indices)):
                if dis_batch_demo_list[sorted_indices[j]][i] <= k1:
                    ret_batch_demo_list[sorted_indices[j]][i]=1
        else:
            for j in range(upper_ques):
                if j<len(sorted_indices) and dis_batch_demo_list[sorted_indices[j]][i] <= k1:
                    ret_batch_demo_list[sorted_indices[j]][i]=1
    
    for i in range(len(ret_batch_demo_list)):
        have_demo = False
        for one_demo in ret_batch_demo_list[i]:
            if one_demo==True:
                have_demo = True
                break
        if not have_demo:
            print('assign_grpDemAff error: question '+str(i)+' do not have similar demonstrations    in assign_grpDemAff')
            exit(-1)
    return ret_batch_demo_list


def assign_grpDemAff_for_code(dis_batch_demo,upper_ques, k1,dataset_name):
    dis_batch_demo_list = dis_batch_demo.tolist()
    
    ret_batch_demo_list = [[0 for _ in range(len(dis_batch_demo_list[0]))] for _ in range(len(dis_batch_demo_list))]
    for i in range(len(dis_batch_demo_list[0])):
        ques_dis_for_one_demo = [row[i] for row in dis_batch_demo_list]
        if i % 10000==0:
            print('index:'+str(i))
        sorted_indices = sorted(range(len(ques_dis_for_one_demo)), key=lambda k: ques_dis_for_one_demo[k]) 
        for j in range(upper_ques):
            if j<len(sorted_indices) and dis_batch_demo_list[sorted_indices[j]][i] <= k1:
                ret_batch_demo_list[sorted_indices[j]][i]=1
    
    for i in range(len(ret_batch_demo_list)):
        have_demo = False
        for one_demo in ret_batch_demo_list[i]:
            if one_demo==True:
                have_demo = True
                break
        if not have_demo:
            print('assign_grpDemAff error: question '+str(i)+' do not have similar demonstrations  in assign_grpDemAff_for_code')
            exit(-1)
    return ret_batch_demo_list

Item_type = {
    'Beer': 'Beer',
    'iTunesAmazon': 'Song',
    'FodorsZagats': 'Restaurant',
    'WalmartAmazon': 'Product',
    'DblpScholar': 'Paper',
    'AbtBuy': 'Product',
    'DblpAcm': 'Paper',
    'AmazonGoogle': 'Product'
}

def get_prompt(dataset_name, demos, pairs):
    if dataset_name in dataset_for_er:
        item_type = "Product"
        task_description = f'When determining whether two Restraunts are the same, you should only focus on critical properties and overlook noisy factors.'
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
        prompt += '\n For the following questions, please give the answer and the corresponding reasons:\n'
        for j in range(len(pairs)):
            prompt += f'Question{j+1}:\n'
            prompt += pairs[j]['question']
            prompt += '\n'
        return prompt 
    elif dataset_name in dataset_for_jd:
        task_description = '''You need to transform the data into another format based on the giving examples. In each example, the question is the data format before transformation, and the answer if the data format after transformation. There are several transformation examples, return answers in JSON like this:
        {
            "Question1":...,
            "Question2":...,
            ...
        }
        Examples:
        '''
        prompt = task_description
        for i in range(len(demos)):
            prompt += "Question:\n"
            prompt += demos[i]['question']
            prompt += '\n'
            prompt += "After transformation:\n"
            prompt += demos[i]['answer']
            prompt += '\n\n'
        prompt += 'Please answer the following questions:\n'
        for j in range(len(pairs)):
            prompt += f'Question{j+1}:\n'
            prompt += pairs[j]['question']
            prompt += '\n'
        return prompt
    else:
        print('dataset error in get_prompt.  dataset:'+dataset_name)
        exit(-1)  
        

def askLLM(prompt, dataset_name, model='gpt-3.5-turbo-0125'):
    if 'gpt' in model: 
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if dataset_name in dataset_for_er:
            try:
                completion = openai.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
            except Exception as e:
                print(e)
        elif dataset_name in dataset_for_code or dataset_name in dataset_for_qa or dataset_name in dataset_for_jd:
            if model == 'gpt-3.5-turbo-0301':
                completion = openai.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
            else:
                completion = openai.ChatCompletion.create(
                    model=model,
                    response_format={ "type": "json_object" }, # json格式返回
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
        ans = completion.choices[0].message['content']
    elif 'claude' in model: 
        client_claude = anthropic.Anthropic(
            api_key="xxxx",
        )
        message = client_claude.messages.create(
            model=model,
            max_tokens=4096,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        ans = message.content[0].text
    elif 'qwen'in model: 
        messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': prompt}]
        dashscope.api_key = 'xxx'
        response = dashscope.Generation.call(model="qwen-long",
                               messages=messages,
                               seed=random.randint(1, 10000),
                               result_format='message')
        ans = ''
        if response.status_code == HTTPStatus.OK:
            ans = response.output.choices[0].message.content
        else:
            print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                response.request_id, response.status_code,
                response.code, response.message
            ))

    return ans


def askLLMandEval(dataset_name, prompts, model):
    max_iters = 30
    if dataset_name in dataset_for_er:
        tp, fp, tn, fn = 0, 0, 0, 0
        for i in range(len(prompts)):
            print('index:'+str(i)+' total:'+str(len(prompts)))
            llm_ans = []
            ans = prompts[i][1]
            cur_iter = 0
            while cur_iter < max_iters:
                cur_iter+=1
                try:
                    res = askLLM(prompts[i][0], dataset_name, model)
                except:
                    print('test except')
                    sleep(5)
                    continue
                print('output')
                print(res)
                res = res.split('\n')
                for s in res:
                    s=s.replace('No', 'no')
                    s=s.replace('Yes', 'yes')
                    if 'yes' in s:
                        llm_ans.append(1)
                    elif 'no' in s:
                        llm_ans.append(0)
                if len(llm_ans) != len(ans): 
                    print('len of llm_ans:'+str(len(llm_ans))+'   len of ans:'+str(len(ans))+' they are not same length')
                    llm_ans = []
                else:
                    break
            for i in range(len(llm_ans)):
                if llm_ans[i] == 1: 
                    if i>=len(ans):
                        break
                    if int(ans[i]) == llm_ans[i]:
                        tp += 1
                    else:
                        fp += 1
                elif llm_ans[i] == 0: 
                    if i>=len(ans):
                        break
                    if int(ans[i]) == llm_ans[i]:
                        tn += 1
                    else:
                        fn += 1
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
        if (tp + fn) != 0 and (tp + fp) != 0:
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
            print('ans:')
            print(ans)
            cur_iter = 0
            while cur_iter < max_iters:
                print('test cur_iters:'+str(cur_iter))
                cur_iter+=1
                tmp_t = 0 
                try:
                    res = askLLM(prompts[i][0], dataset_name, model)
                    print('llm_ans:')
                    print(res)
                    llm_ans = json.loads(res)
                    for j in range(len(ans)):
                        ques = 'Question'+str(j+1)
                        if ques not in llm_ans.keys():
                            continue
                
                        llm_ans_filter=''
                        llm_ans_filter = llm_ans[f'Question{j+1}'].rstrip(';').replace(' ', '')
                        ans_filter = ans[j].replace(' ', '')
                        if llm_ans_filter == ans_filter:
                            t += 1
                            tmp_t += 1
                    num += len(ans)
                    print('number of questions:'+str(len(ans)))
                    if len(ans)!=0:
                        print('local acc:', str(float(tmp_t) / len(ans)))
                    if num!=0:
                        print('acc: ', str(float(t) / num))
                    
                except Exception as e:
                    if tmp_t !=0:
                        t -= tmp_t
                        tmp_t = 0
                    print('test except'+str(e))
                    sleep(10)
                    continue
                break            
            
        if num!=0:
            print('total acc: ', str(float(t) / num))
    elif dataset_name in dataset_for_qa:
        t, num = 0, 0
        for i in range(len(prompts)):
            ans = prompts[i][1]
            print('ans:')
            print(ans)
            ans = ['yes' if i == True else 'no' for i in ans]
            cur_iter = 0
            while cur_iter < max_iters:
                print('test cur_iters:'+str(cur_iter))
                cur_iter+=1
                tmp_t = 0
                try:
                    res = askLLM(prompts[i][0], dataset_name, model)
                    print('llm_ans:')
                    print(res)
                    llm_ans = json.loads(res) 
                    for j in range(len(ans)):
                        ques = 'Question'+str(j+1)
                        if ques not in llm_ans.keys():
                            continue
                
                        llm_ans_filter=''
                        llm_ans_filter = llm_ans[f'Question{j+1}'].rstrip(';').replace(' ', '')
                        ans_filter = ans[j].replace(' ', '')
                        if llm_ans_filter == ans_filter:
                            t += 1
                            tmp_t += 1
                    num += len(ans)
                    print('number of questions:'+str(len(ans)))
                    if len(ans)!=0:
                        print('local acc:', str(float(tmp_t) / len(ans)))
                    if num!=0:
                        print('acc: ', str(float(t) / num))
                except Exception as e:
                    if tmp_t !=0:
                        t -= tmp_t
                        tmp_t = 0
                    print('test except'+str(e))
                    sleep(10)
                    continue
                break
        print('total acc: ', str(float(t) / num))
    elif dataset_name in dataset_for_jd:
        total_t, total_num = 0, 0
        for table, one_table_prompts in prompts.items():
            t, num = 0, 0
            print('table:'+table)
            for i in range(len(one_table_prompts)):
                ans = one_table_prompts[i][1]
                print('ans:')
                print(ans)
                cur_iter = 0
                while cur_iter < max_iters:
                    print('test cur_iters:'+str(cur_iter))
                    cur_iter+=1
                    tmp_t = 0 
                    try:
                        res = askLLM(one_table_prompts[i][0], dataset_name, model)
                        print('llm_ans:')
                        print(res)
                        llm_ans = json.loads(res)
                        for j in range(len(ans)):
                            ques = 'Question'+str(j+1)
                            if ques not in llm_ans.keys():
                                continue
                
                            llm_ans_filter=''
                            if type(llm_ans[f'Question{j+1}']) == int:
                                llm_ans_filter = str(llm_ans[f'Question{j+1}'])
                            else:
                                llm_ans_filter = llm_ans[f'Question{j+1}'].rstrip(';').replace(' ', '')
                            ans_filter = ans[j].replace(' ', '')
                            if llm_ans_filter == ans_filter:
                                t += 1
                                total_t += 1
                                tmp_t += 1
                        num += len(ans)
                        total_num += len(ans)
                        print('number of questions:'+str(len(ans)))
                        if len(ans)!=0:
                            print('local acc:', str(float(tmp_t) / len(ans)))
                        if num!=0:
                            print('cumulative acc: ', str(float(t) / num))
                    except Exception as e:
                        if tmp_t !=0:
                            t -= tmp_t
                            total_t -= tmp_t
                            tmp_t = 0
                        print('test except'+str(e))
                        sleep(10)
                        continue
                    break            
            if num!=0:
                print('table:'+table+'    total acc: ', str(float(t) / num))
        if total_num != 0:
            print('all tables  total acc:'+str(float(total_t) / total_num))
    else:
        print('dataset error in askLLMandEval.  dataset:'+dataset_name)
        exit(-1)  