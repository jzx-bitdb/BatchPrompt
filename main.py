import os
import random
import argparse
import jsonlines
import time
from batch_process.Approx.approx import approx_all
from util import *
import tiktoken
random.seed(2024)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'



def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='AmazonGoogle', choices=[
                        'StrategyQA', 'CommonsenseQA', 
                        #for entity resolution
                        'AmazonGoogle', 'FodorsZagats','DblpScholar', 'DblpAcm', 'WalmartAmazon', 'AbtBuy', 'iTunesAmazon', 'Beer', 
                        #for code assertion generation
                        'Atlas_assertEquals','Atlas_assertNotNull','Atlas_assertThat','Atlas_assertTrue',
                        # Joinable detection
                        'DXF', 'FF', 'AJ'
                        ]) 
    parser.add_argument('--batch_dir', type=str, default='./data/batch_data')
    parser.add_argument('--demo_dir', type=str, default='./data/demostrations')
    parser.add_argument('--batch_method', type=str, default='ApproxAll',
                        choices=['ApproxAll', 'askLLM', 'Batcher'])
    parser.add_argument('--upper_ques', type=int,  #paper中的demonstration coverage
                        default=3, help='the upper number of questions for each demo can cover')
    parser.add_argument('--k0', type=float, default='0.77',help='Group Aff')
    parser.add_argument('--k1', type=float, default='0.6',help='Demo Aff')
    parser.add_argument('--k2', type=int, default='500',help='Cost Budget per group(Token number)')
    parser.add_argument('--batch_type', type=str, default='diverse',  #question affinity type
                        choices=['random', 'similar', 'diverse'])
    parser.add_argument('--sim_type', type=str, default='structure', 
                        choices=['structure', 'semantic','bm_25'])
    parser.add_argument('--output_dir', type=str, 
                        default='output/group_assignment_for_vldb', help='directory name of the output')
    parser.add_argument('--is_dominated', type=bool,
                        default=False, help='filter dominated demonstrations')
    parser.add_argument('--for_testpara', type=bool,
                        default=False, help='For test parameter')
    parser.add_argument('--random_demos', type=bool,
                        default=False, help='select demonstrations randomly')
    parser.add_argument('--prompt_path', type=str,
                        default='xxx', help='dir of xxx_prompts.json')
    parser.add_argument('--filter_path', type=str,  
                        default='xxx', help='dir of xxx_group.json')
    parser.add_argument('--set_cover_num', type=int, 
                        default='2', help='select first filter_num questions for Atlas due to large dataset')
    parser.add_argument('--model', type=str,  
                        default='gpt-3.5-turbo-0125', help='LLMs model')
    parser.add_argument('--balance_method', type=int,  
                        default=1, help='LLMs model')
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
    elif dataset_name in dataset_for_qa:
        taskdes_for_batcher = '''Answer common sense questions based on your knowledge, only yes or no.There are several questions below Examples,return answers in JSON like this:
        {
            "Question1":yes/no,
            "Question2":yes/no,
            ...
        }
        Examples:
        '''
    elif dataset_name in dataset_for_jd:
        taskdes_for_batcher = '''You need to transform the data into another format based on the giving examples. There are several transformation examples: '''
    else:
        print('dataset error:'+dataset_name)
        exit(-1)
        
    tCost = get_cost_for_des(taskdes_for_batcher, encoder)
    print('tCost:'+str(tCost))

    
    if batch_method == 'ApproxAll':
        print('question num:'+str(len(batch_data)))
        k0 = args.k0
        k1 = args.k1
        k2 = args.k2
        set_cover_num = args.set_cover_num
        filter_path = args.filter_path
        time_computation = 0
        balance_method = args.balance_method
        prompts, tokens, filter_time, compute_time, getfeature_time, cluster_time, assign_time, cltsplt_time, time_preprocess = approx_all(dataset_name, tCost, upper_ques, k0, k1, k2, output_dir, batch_type, sim_type, batch_data, demonstrations_data,model=llm_model)
        print("total tokens: ", str(tokens))
        print('total compute_time:'+str(compute_time))
        print('total filter_time:'+str(filter_time))
        print('total getfeature_time:'+str(getfeature_time))
        print('cluster_time:'+str(cluster_time))
        print('assign_time:'+str(assign_time))
        print('cltsplt_time:'+str(cltsplt_time))
        print('group_num: ' + str(len(prompts)))
        start = time.time()
        if len(prompts) > 0:
            askLLMandEval(dataset_name, prompts,llm_model)
        else:
            print('null prompts')
        end = time.time()
        time_ask_llm = end - start
        print('total time ask llm:'+str(time_ask_llm))
    elif batch_method == 'Batcher':
        pass
    elif batch_method == 'askLLM':
        pass
    else:
        print('batch_method:'+batch_method+' is wrong')
        exit(-1)
    
    