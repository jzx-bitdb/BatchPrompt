import jsonlines
import tiktoken
import time
from util import *
from sklearn.cluster import DBSCAN
import copy
import binpacking
from collections import Counter
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

def overlap_items_func(dp_keys, cur_i, sets_ori):
    if cur_i == 0: 
        total_items = {}
        for one_set in sets_ori:
            for one_item in one_set:
                if one_item not in total_items.keys():
                    total_items[one_item] = 0
                total_items[one_item] += 1
        ret_removes_for_first = []
        for one_item in sets_ori[cur_i]: 
            if total_items[one_item] > 1:
                ret_removes_for_first.append(one_item)
        return {'':ret_removes_for_first}
    
    total_items ={}
    for i in range(0, cur_i):
        assert(i!=cur_i)
        for one_item in sets_ori[i]:
            if one_item not in total_items.keys(): 
                total_items[one_item] = 0
            total_items[one_item] += 1
    overlap_items=[]
    for cur_item in sets_ori[cur_i]:
        if cur_item in total_items.keys():
            overlap_items.append(cur_item)
    
    ret_removes = {}
    for one_del in dp_keys:
        tmp_total_items = copy.deepcopy(total_items)
        if '_' in one_del:
            remove_items = one_del.split('_') 
            for one_item in remove_items:
                int_item = int(one_item)
                tmp_total_items[int_item]-=1 
        del_items = []
        for one_overlap in overlap_items:
            if tmp_total_items[one_overlap] >=1: 
                del_items.append(one_overlap)
        ret_removes[one_del]=del_items
    return ret_removes 

def balance_sets_dp(sets_ori):
    set_size = len(sets_ori)
    dp = [{} for _ in range(set_size+1)]
    
    dp[0]['']=[0,[]]
    print('total set size:'+str(set_size+1))
    for i in range(1, set_size+1):
        print('processing index:'+str(i))
        ret_removes = overlap_items_func(dp[i-1].keys(),i-1, sets_ori)
        for ori_remove, one_remove in ret_removes.items():
            if len(one_remove)==0:
                dp[i][ori_remove] = copy.deepcopy(dp[i-1][ori_remove])
                dp[i][ori_remove][1].append(sets_ori[i-1])
            
            for j in range(1,len(one_remove)+1):
                all_combs = list(itertools.combinations(one_remove,j))
                if ori_remove == '':
                    ori_remove_list = []
                else:
                    ori_remove_list = ori_remove.split('_')
                for one_comb in all_combs:
                    final_key = []
                    for d in ori_remove_list:
                        final_key.append(str(d))
                    cur_i_set_after_remove = copy.deepcopy(sets_ori[i-1])
                    for dd in one_comb:
                        final_key.append(str(dd))
                        cur_i_set_after_remove.remove(dd)
                    final_key_str = '_'.join(final_key)
                    if final_key_str not in dp[i].keys():
                        dp[i][final_key_str] = [0,[]]
                        dp[i][final_key_str][0] = max(dp[i-1][ori_remove][0],len(sets_ori[i-1])-j)
                        dp[i][final_key_str][1] = copy.deepcopy(dp[i-1][ori_remove][1])
                        dp[i][final_key_str][1].append(cur_i_set_after_remove)
                    else:
                        dp[i][final_key_str][0] = min(dp[i][final_key_str][0], max(dp[i-1][ori_remove][0],len(sets_ori[i-1])-j))
                        dp[i][final_key_str][1][-1] = copy.deepcopy(cur_i_set_after_remove)
    total_items = {}
    for one_set in sets_ori:
        for one_item in one_set:
            if one_item not in total_items.keys():
                total_items[one_item] = 0
            total_items[one_item] += 1
    total_overlaps = []
    for k, v in total_items.items():
        while v>1:
            v-=1
            total_overlaps.append(str(k))
    total_overlaps = sorted(total_overlaps)
    total_overlaps_str = '_'.join(total_overlaps)
    min_size = 1000000000
    ret_list = []
    print('tt:'+total_overlaps_str)
    print('len of dp[set_size]:'+str(len(dp[set_size].keys())))
    for total_remove_keys, values in dp[set_size].items():
        total_keys = total_remove_keys.split('_')
        total_keys = sorted(total_keys)
        total_keys_str = '_'.join(total_keys)
        if total_keys_str == total_overlaps_str:
            print('test1')
            if values[0] < min_size:
                print('test2')
                min_size = values[0]
                ret_list = copy.deepcopy(values[1])
    if min_size != 1000000000:
        print('ret_lsit:'+str(ret_list))
        return ret_list
    else:
        print('dp error')
        exit(-1)


def random_balance_sets(sets):
    random.shuffle(sets)
    unique_all_sets = set()
    ret_all_sets = []
    for one_set in sets:
        ret_one_set = set()
        for one_data in one_set:
            if one_data in unique_all_sets:
                pass
            else:
                ret_one_set.add(one_data)
                unique_all_sets.add(one_data)
        ret_all_sets.append(ret_one_set)
    return ret_all_sets


def balance_sets_greedy(sets_ori):
    sets = copy.deepcopy(sets_ori)
    total_item = set()
    for one_set in sets:
        for one_item in one_set:
            total_item.add(one_item)
    total_item_num = len(total_item) 
    avg_items_per_set = int(total_item_num/len(sets)) 
    overlap_items = Counter(item for s in sets for item in s)
    
    while True:
        max_overlap_item, max_overlap_count = overlap_items.most_common(1)[0] 
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
    while True: 
        max_overlap_item, max_overlap_count = overlap_items.most_common(1)[0] 
        if max_overlap_count == 1:
            break
        for s in sets:
            if overlap_items[max_overlap_item] == 1:
                break
            if max_overlap_item in s:
                s.remove(max_overlap_item)
                overlap_items[max_overlap_item]-=1  
    return sets


def approx_all(dataset_name, tCost, upper_ques, k0, k1, k2, output_dir, batch_type='diverse',sim_type='structure', batch_data=None, demo_data=None, is_dominated=False, set_cover_num=2, model='gpt-3.5-turbo-0125'):
    print('approx_all')
    encoder = tiktoken.encoding_for_model(model)
    batch_index = []
    if dataset_name in dataset_for_code:
        batch_feature = get_feature_semantic([d['total'] for d in batch_data], dataset_name)
        min_samples_ = 1
        dbscanF_ = DBSCAN(eps=k0, min_samples=min_samples_)
        preds_ = dbscanF_.fit_predict(batch_feature.cpu())  
        pair2cluster, cluster2pair = {},  {}  
        for pid, cluster in enumerate(preds_):
            pair2cluster[pid] = cluster
            if cluster not in cluster2pair:
                cluster2pair[cluster] = [pid]
            else:
                cluster2pair[cluster].append(pid)
        print('original cluster num for '+dataset_name+': '+str(len(cluster2pair)))
        for k, one_cluster in cluster2pair.items():
            if len(one_cluster)>2:
                for one_que in one_cluster:
                    batch_index.append(one_que)
    print('final data after cluster:'+str(len(batch_index)))
    print('original batch_data num:'+str(len(batch_data))+'  demo_data num:'+str(len(demo_data)))
    start_filter = time.time()
    batch_data_index, demo_data_index = filter_ques_demos(dataset_name, batch_data, demo_data, k1, upper_ques, sim_type, encoder, batch_index=batch_index)
    print('after filter')
    print('batch_data num:'+str(len(batch_data_index))+'  demo_data num:'+str(len(demo_data_index)))
    end_filter = time.time()
    
    
    start_compute  = time.time()
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
    
    demo_dis = get_dis(batch_feature, demo_feature, dataset_name, sim_type)
    start_assign = time.time()
    grpDemAff = assign_grpDemAff(demo_dis, upper_ques , k1, dataset_name) 
    end_assign = time.time()
    print('grpDemAff:'+str(len(grpDemAff)))
    print('grpDemAff[0]:'+str(len(grpDemAff[0])))
    end_getfeature = time.time()
    time_getfeature = end_getfeature-start_getfeature 
    
    # Use DBSCAN for correlation clustering
    pair2cluster, cluster2pair = {},  {}
    if batch_type != 'random':
        min_samples_ = 1 
        
        if batch_type == 'diverse':
            strat_preprocess = time.time()
            start_cluster = time.time()
            end_preprocess = time.time()
            dbscan = DBSCAN(eps=k0, min_samples=min_samples_, metric=normalized_euclidean) 
            preds = dbscan.fit_predict(batch_feature)
        elif batch_type == 'similar' and sim_type == 'semantic':
            dbscan = DBSCAN(eps=k0, min_samples=min_samples_, metric='cosine') 
            preds = dbscan.fit_predict(batch_feature.cpu())
        end_cluster = time.time()
        for pid, cluster in enumerate(preds):
            pair2cluster[pid] = cluster
            if cluster not in cluster2pair:
                cluster2pair[cluster] = [pid]
            else:
                cluster2pair[cluster].append(pid)
    
    else: 
        cluster2pair[0] = []
        for i in range(len(batch_feature)):
            cluster2pair[0].append(i)
            pair2cluster[i]=0

    
    if -1 in cluster2pair.keys():
        exit(-1)
    print('final cluster num:'+str(len(cluster2pair)))
    total_ques = 0
    for one_cluster in cluster2pair.values():
        total_ques+=len(one_cluster)
    print('total ques:'+str(total_ques))
    new_batch_data_cost = [-1 for _ in range(len(new_batch_data))]
    new_demo_data_cost = [-1 for _ in range(len(new_demo_data))]
    
    time_cluster = end_cluster - start_cluster
    
    all_groups = []
    prompts = []
    start_cltsplt = time.time()
    for each_cluster in cluster2pair.keys():
        ques = cluster2pair[each_cluster] 
        demo_ques_list_for_each_cluster = {}
        demo_for_each_cluster = set()
        for d in ques:
            for i in range(len(new_demo_data)):
                if grpDemAff[d][i] == 1:
                    demo_for_each_cluster.add(i)
        print('demo_for_each_cluster:'+str(len(demo_for_each_cluster)))
        print('ques:'+str(ques)+'   len:'+str(len(ques)))
        for each_demo in demo_for_each_cluster:
            for d in ques:
                if grpDemAff[d][each_demo] == 1:
                    if each_demo not in demo_ques_list_for_each_cluster.keys():
                        demo_ques_list_for_each_cluster[each_demo] = set()
                    demo_ques_list_for_each_cluster[each_demo].add(d)
        if  is_dominated:
            print('before min_demo:'+str(len(demo_for_each_cluster)))
            print('len of demo_ques_list_for_each_cluster:'+str(len(demo_ques_list_for_each_cluster)))
            demo_for_each_cluster = filter_dominated(demo_ques_list_for_each_cluster, new_demo_data, encoder, dataset_name)
            print('after min_demo:'+str(len(demo_for_each_cluster)))
        
        mini_demo_data = []
        for i in demo_for_each_cluster:
            mini_demo_data.append(new_demo_data[i])
        
        queCost = []
        for question in ques:
            if dataset_name in dataset_for_code:
                queCost.append(get_cost(new_batch_data[question]['total'], encoder, dataset_name))
            else:
                queCost.append(get_cost(new_batch_data[question]['question'], encoder, dataset_name))
        
        demCost = []
        for question in mini_demo_data:
            if dataset_name in dataset_for_code:
                demCost.append(get_cost(question['total'], encoder, dataset_name))
            else:
                demCost.append(get_cost(question['question'], encoder, dataset_name))
    
        set_cost = []
        subsets = []
        demo_index = 0
        list_demo_for_each_cluster =[]
        for i in demo_for_each_cluster:
            one_set_cost = demCost[demo_index]
            demo_index+=1
            one_set = set()
            for j in ques:
                ques_index = 0
                if grpDemAff[j][i] == 1:
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
        cover_num = 0
        for one_cover in covers:
            cover_num += len(one_cover)
        print('cover_num:'+str(cover_num))
        each_set_cost_after_reassign = balance_sets_greedy(covers)
        ques_num = 0
        #bin packing
        items = {}
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
            for each_ques in each_set_cost_after_reassign[i]:
                if new_batch_data_cost[each_ques]==-1:
                    if dataset_name in dataset_for_code:
                        one_queCost = get_cost(new_batch_data[each_ques]['total'],encoder, dataset_name)
                    else:
                        one_queCost = get_cost(new_batch_data[each_ques]['question'],encoder, dataset_name)
                    que_cost += one_queCost
                    new_batch_data_cost[each_ques] = one_queCost
                else:
                    que_cost += new_batch_data_cost[each_ques]
            items[i]=(demo_cost+que_cost)
            print('items['+str(i)+']:'+str(items[i]))
        
        
        bins = binpacking.to_constant_volume(items, k2-tCost)
        print(k2-tCost)
        
        for one_bin in bins:
            print('one_bin:'+str(one_bin))
            ori_ques_list = []
            ori_demo_list = set()
            for each_demo in one_bin.keys():
                ori_demo_list.add(list_demo_for_each_cluster[indexes[each_demo]])
                for each_ques in each_set_cost_after_reassign[each_demo]:
                    ori_ques_list.append(each_ques) 
            if dataset_name in dataset_for_code or dataset_name in dataset_for_qa:
                universe_per_group = set()
                for one_ques in ori_ques_list:
                    universe_per_group.add(one_ques)
                for _ in range(set_cover_num):
                    subsets_per_group = []
                    demo_index = 0
                    if len(ori_demo_list) == len(demo_for_each_cluster):
                        break
                    for i in demo_for_each_cluster:
                        if i in ori_demo_list:
                            empty_set = set()
                            subsets_per_group.append(empty_set)
                            continue
                        demo_index+=1
                        one_set = set()
                        for j in ori_ques_list:
                            ques_index = 0
                            if grpDemAff[j][i] == 1:
                                one_set.add(j)
                            ques_index+=1
                        subsets_per_group.append(one_set)
                    _, indexes_per_group = set_cover(universe_per_group, subsets_per_group, set_cost)
                    
                    have_bound = False
                    for one_demo in indexes_per_group:
                        ori_demo_list.add(list_demo_for_each_cluster[one_demo])
                        if len(ori_demo_list) >= len(ori_ques_list):
                            have_bound=True
                            break
                    if have_bound:
                        break
            final_demo_list = []
            for one_demo in ori_demo_list:
                final_demo_list.append(new_demo_data_to_old[one_demo])
            final_batch_list =[]
            for one_batch in ori_ques_list:
                final_batch_list.append(new_batch_data_to_old[one_batch])
            one_group = {}
            ques_num += len(final_batch_list)
            if len(final_batch_list) != 0 and len(final_demo_list)!=0:
                one_group['ques'] = final_batch_list 
                one_group['demo'] = final_demo_list
                all_groups.append(one_group)
        print('after depulication ques num:'+str(ques_num))
    end_CltSplt = time.time()
    with jsonlines.open(output_dir + '/' + dataset_name +'/'+batch_type+'_'+sim_type+'_'+str(k0)+'_'+str(k1)+'_'+str(k2)+'_'+str(upper_ques)+'_'+str(set_cover_num)+'_'+model+'_groups.json', 'w') as f_group:
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
            if dataset_name in dataset_for_code or dataset_name in dataset_for_qa:
                ans.append(batch_data[j]['answer'])
            else:
                ans.append(int(batch_data[j]['answer']))
        prompt = get_prompt(dataset_name, demos_data, batches_data)
        tokens += len(encoder.encode(prompt))
        prompts.append([prompt, ans])
    with jsonlines.open(output_dir + '/' + dataset_name +'/'+batch_type+'_'+sim_type+'_'+str(k0)+'_'+str(k1)+'_'+str(k2)+'_'+str(upper_ques)+'_'+str(set_cover_num)+'_'+model+'_prompts.json', 'w') as f_prompt:
        f_prompt.write_all(prompts)
    end_compute = time.time()
    filter_time = end_filter - start_filter
    compute_time = end_compute - start_compute
    assign_time = end_assign - start_assign
    cltsplt_time = end_CltSplt - start_cltsplt
    time_preprocess = end_preprocess - strat_preprocess
    return  prompts, tokens, filter_time, compute_time, time_getfeature, time_cluster, assign_time, cltsplt_time,time_preprocess
        
    