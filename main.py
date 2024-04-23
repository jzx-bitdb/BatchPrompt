import os
import random
import argparse
import json
import jsonlines
from batch_process.ILP.ILP_jzx import ILP
from batch_process.Approx.group import approx_grouping
from batch_process.Approx.approx import approx_all
from batch_process.Approx.approx_for_testpara import approx_all_for_testpara
from batch_process.Approx.batcher import batcher_for_code, compute_efficient
from util import *
import tiktoken
import openai
random.seed(2024)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='AmazonGoogle', choices=[
                        'AmazonGoogle', 'FodorsZagats','DblpScholar', 'DblpAcm', 'WalmartAmazon', 'AbtBuy', 'iTunesAmazon', 'Beer', 
                        'Atlas_assertEquals','Atlas_assertNotNull','Atlas_assertThat','Atlas_assertTrue']) #数据集的名称
    parser.add_argument('--batch_dir', type=str, default='data/batch_data') #question所在的路径
    parser.add_argument('--demo_dir', type=str, default='data/demostrations') #demonstration所在的路径
    parser.add_argument('--batch_method', type=str, default='ApproxGroup',
                        choices=['ILP','ApproxGroup','ApproxAll', 'Triangle','Batcher']) 
    parser.add_argument('--upper_ques', type=int,
                        default=3, help='the upper number of questions for each demo can cover')
    parser.add_argument('--k0', type=float, default='0.77',help='Group Aff')
    parser.add_argument('--k1', type=float, default='0.6',help='Demo Aff')
    parser.add_argument('--k2', type=int, default='500',help='Cost Budget per group(Token number)')
    parser.add_argument('--batch_type', type=str, default='diverse',  #question affinity type
                        choices=['random', 'similar', 'diverse'])
    parser.add_argument('--sim_type', type=str, default='structure',
                        choices=['structure', 'semantic'])
    parser.add_argument('--output_dir', type=str,  #结果输出路径
                        default='output/group_assignment', help='directory name of the output')
    parser.add_argument('--is_dominated', type=bool,
                        default=False, help='filter dominated demonstrations')
    parser.add_argument('--for_testpara', type=bool,
                        default=False, help='For test parameter')
    parser.add_argument('--filter_path', type=str,
                        default='xxx', help='dir of xxx_group.json')
    parser.add_argument('--model', type=str,
                        default='gpt-3.5-turbo-0125', help='LLMs model')
    return parser.parse_args()

if __name__ == '__main__':
    
    
    args = arg_parser()
    batch_method = args.batch_method
    dataset_name = args.dataset_name
    batch_dir = args.batch_dir
    demo_dir = args.demo_dir
    batch_type = args.batch_type
    sim_type = args.sim_type
    upper_ques = args.upper_ques
    output_dir = args.output_dir
    llm_model = args.model
    batch_data = []
    demonstrations_data = []
    #问题集合
    with jsonlines.open(batch_dir + '/' + dataset_name + '/' + dataset_name + '.jsonl', 'r') as f_batch:
        for i in f_batch:
            batch_data.append(i)
    with jsonlines.open(demo_dir + '/' + dataset_name + '/' + dataset_name + '.jsonl', 'r') as f_demo:
        for i in f_demo:
            demonstrations_data.append(i)
    tCost = 0
    encoder = tiktoken.encoding_for_model(llm_model)
    if dataset_name in dataset_for_er:
        taskdes_for_batcher = 'When determining whether two Restraunts are the same, you should only focus on critical properties and overlook noisy factors.\n\n \n\nUse domain knowledge of Products to help understand the text and answer the above 4 questions in the format: For Question i, Yes, Product A and Product B are the same product./No, Product A and Product B are different products. For Question i+1, (repeat the above procedures)", [0, 0, 0, 0]'        
    elif dataset_name in dataset_for_code:
        taskdes_for_batcher = '''You are a professional Java expert at generating meaningful assert statement for test method. 
Given a test method and a focal method , give a assert statement to assess the correctness of the focal method.
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
    tCost = get_cost_for_des(taskdes_for_batcher, encoder)
    print('tCost:'+str(tCost))
    if batch_method == 'ILP':
        print('question num:'+str(len(batch_data)))
        k0 = args.k0
        k1 = args.k1
        k2 = args.k2
        prompts, tokens = ILP(dataset_name, tCost, upper_ques, k0, k1, k2, batch_type,sim_type, batch_data, demonstrations_data)
        print("tokens: ", str(tokens))
        askLLMandEval(dataset_name, prompts)
    elif batch_method == 'ApproxGroup':
        print('question num:'+str(len(batch_data)))
        k0 = args.k0
        k1 = args.k1
        k2 = args.k2
        prompts, tokens = approx_grouping(dataset_name, tCost,upper_ques, k0, k1, k2, output_dir, batch_type,sim_type, batch_data, demonstrations_data)
        print("tokens: ", str(tokens))
        askLLMandEval(dataset_name, prompts)
    if batch_method == 'ApproxAll':
        print('question num:'+str(len(batch_data)))
        k0 = args.k0
        k1 = args.k1
        k2 = args.k2
        for_testpara = args.for_testpara
        filter_path = args.filter_path
        time_computation = 0
        if for_testpara:
            if dataset_name in dataset_for_code: 
                prompts, tokens = approx_all_for_testpara(dataset_name, filter_path, tCost, upper_ques, k0, k1, k2, output_dir, batch_type, sim_type, batch_data, demonstrations_data, model=llm_model)
            else:
                prompts, tokens = approx_all_for_testpara(dataset_name, filter_path, tCost, upper_ques, k0, k1, k2, output_dir, batch_type, sim_type, batch_data, demonstrations_data,model=llm_model)
        else:
            if dataset_name in dataset_for_code:
                prompts, tokens = approx_all(dataset_name, tCost, upper_ques, k0, k1, k2, output_dir, batch_type, sim_type, batch_data, demonstrations_data,model=llm_model)
            else:
                prompts, tokens = approx_all(dataset_name, tCost, upper_ques, k0, k1, k2, output_dir, batch_type, sim_type, batch_data, demonstrations_data,model=llm_model)
        print("total tokens: ", str(tokens))
        print('group_num: ' + str(len(prompts)))
        start = time.time()
        if len(prompts) > 0:
            askLLMandEval(dataset_name, prompts,llm_model)
        end = time.time()
        time_ask_llm = end - start
        print('total time ask llm:'+str(time_ask_llm))
    elif batch_method == 'Batcher':
        #only for code dataset
        filter_path = args.filter_path
        demo_method = args.demo_method
        batch_size = args.batch_size
        demo_percentile = args.demo_percentile
        prompts, tokens = batcher_for_code(dataset_name, demo_method, filter_path, output_dir, batch_type=batch_type,sim_type='semantic', batch_size=batch_size, demo_percentile=demo_percentile, real_batch_data=batch_data, demo_data=demonstrations_data, model=llm_model)
        print("total tokens: ", str(tokens))
        start = time.time()
        if len(prompts) > 0:
            askLLMandEval(dataset_name, prompts,llm_model)
        end = time.time()
        time_ask_llm = end - start
        print('total time ask llm:'+str(time_ask_llm))
    elif batch_method == 'Triangle':
        k0 = args.k0
        k1 = args.k1
        compute_efficient(dataset_name,k0,k1,batch_data,demonstrations_data)
    else:
        print('batch_method:'+batch_method+' is wrong')
        exit(-1)
    
    