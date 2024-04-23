# ILP
# Fodors-Zagats
python main.py --k0 0.15 --k1 0.03 --k2 1000 --dataset_name FodorsZagats --group_num 8 > ./Results/ILP/Fodors-Zagats.log
# Beer
python main.py --k0 0.01 --k1 0.005 --k2 1000 --dataset_name Beer --group_num 8 > ./Results/ILP/Beer.log
# iTunes-Amazon
python main.py --k0 0.01 --k1 0.005 --k2 1000 --dataset_name iTunesAmazon --group_num 8 > ./Results/ILP/iTunes-Amazon.log 
# Dblp-Acm
python main.py --k0 0.01 --k1 0.005 --k2 1000 --dataset_name DblpAcm --group_num 8 > ./Results/ILP/Dblp-Acm.log
# Walmart-Amazon
python main.py --k0 0.08 --k1 0.005 --k2 1000 --dataset_name WalmartAmazon --group_num 8 > ./Results/ILP/Walmart-Amazon.log
# Abt-Buy
python main.py --k0 0.015 --k1 0.005 --k2 1000 --dataset_name AbtBuy --group_num 8 > ./Results/ILP/Abt-Buy.log
# Amazon-Google
python main.py --k0 0.06 --k1 0.003 --k2 1000 --dataset_name AmazonGoogle --group_num 8 > ./Results/ILP/Amazon-Google.log
# Dblp-Scholar
python main.py --k0 0.1 --k1 0.005 --k2 1000 --dataset_name DblpScholar --group_num 8 > ./Results/ILP/Dblp-Scholar.log
# AssertTrue
python main.py --k0 0.5 --k1 0.1 --k2 5000 --dataset_name Atlas_assertTrue_ILP --batch_type similar --sim_type semantic --group_num 16 --cover_cnt 1 > ./Results/ILP/AssertTrue.log
# AssertNotNull
python main.py --k0 0.5 --k1 0.1 --k2 5000 --dataset_name Atlas_assertNotNull_ILP --batch_type similar --sim_type semantic --group_num 16 --cover_cnt 1 > ./Results/ILP/AssertNotNull.log
# AssertEquals
python main.py --k0 0.5 --k1 0.1 --k2 5000 --dataset_name Atlas_assertEquals_ILP --batch_type similar --sim_type semantic --group_num 16 --cover_cnt 1 > ./Results/ILP/AssertEquals.log
# AssertThat
python main.py --k0 0.45 --k1 0.1 --k2 5000 --dataset_name Atlas_assertThat_ILP --batch_type similar --sim_type semantic --group_num 16 --cover_cnt 1 > ./Results/ILP/AssertThat.log

# OBP
# Fodors-Zagats
python main.py --dataset_name FodorsZagats --batch_method ApproxAll --k0 0.92 --k1 0.31 --k2 700 --upper_ques 5 --model gpt-3.5-turbo-0125 > ./Results/OBP/Fodors-Zagats_0125.log
python main.py --dataset_name FodorsZagats --batch_method ApproxAll --k0 0.92 --k1 0.31 --k2 700 --upper_ques 5 --model gpt-3.5-turbo-0301 > ./Results/OBP/Fodors-Zagats_0301.log
# Beer
python main.py --dataset_name Beer --batch_method ApproxAll --k0 1.9 --k1 0.13 --k2 300 --upper_ques 5 --model gpt-3.5-turbo-0125 > ./Results/OBP/Beer_0125.log
python main.py --dataset_name Beer --batch_method ApproxAll --k0 1.9 --k1 0.13 --k2 300 --upper_ques 5 --model gpt-3.5-turbo-0301 > ./Results/OBP/Beer_0301.log
# iTunes-Amazon
python main.py --dataset_name iTunesAmazon --batch_method ApproxAll --k0 2.28 --k1 0.02 --k2 300 --upper_ques 20 --model gpt-3.5-turbo-0125 > ./Results/OBP/iTunes-Amazon_0125.log
python main.py --dataset_name iTunesAmazon --batch_method ApproxAll --k0 2.28 --k1 0.02 --k2 300 --upper_ques 20 --model gpt-3.5-turbo-0301 > ./Results/OBP/iTunes-Amazon_0301.log 
# Dblp-Acm
python main.py --dataset_name DblpAcm --batch_method ApproxAll --k0 0.88 --k1 0.24 --k2 800 --upper_ques 5 --model gpt-3.5-turbo-0125 --for_testpara 1 --filter_path DblpAcm/diverse_structure_0.88_0.24_800_5_2_gpt-3.5-turbo-0125_groups.json > ./Results/OBP/DblpAcm_0125.log
python main.py --dataset_name DblpAcm --batch_method ApproxAll --k0 1 --k1 0.24 --k2 600 --upper_ques 5 --model gpt-3.5-turbo-0301 --for_testpara 1 --filter_path DblpAcm/diverse_structure_0.88_0.24_800_5_2_gpt-3.5-turbo-0125_groups.json > ./Results/OBP/DblpAcm_0301.log
# Walmart-Amazon
python main.py --dataset_name WalmartAmazon --batch_method ApproxAll --k0 1.42 --k1 0.19 --k2 800 --upper_ques 5 --model gpt-3.5-turbo-0125 > ./Results/OBP/Walmart-Amazon_0125.log 
python main.py --dataset_name WalmartAmazon --batch_method ApproxAll --k0 1.42 --k1 0.19 --k2 800 --upper_ques 5 --model gpt-3.5-turbo-0301 > ./Results/OBP/Walmart-Amazon_0301.log 
# Abt-Buy
python main.py --dataset_name AbtBuy --batch_method ApproxAll --k0 10 --k1 0.02 --k2 400 --upper_ques 5 --model gpt-3.5-turbo-0125 --for_testpara 1 --filter_path AbtBuy/diverse_structure_4.3_0.02_300_5_2_gpt-3.5-turbo-0125_groups.json > ./Results/OBP/Abt-Buy_0125.log
python main.py --dataset_name AbtBuy --batch_method ApproxAll --k0 10 --k1 0.02 --k2 400 --upper_ques 5 --model gpt-3.5-turbo-0301 --for_testpara 1 --filter_path AbtBuy/diverse_structure_4.3_0.02_300_5_2_gpt-3.5-turbo-0125_groups.json > ./Results/OBP/Abt-Buy_0301.log
# Amazon-Google
python main.py --dataset_name AmazonGoogle --batch_method ApproxAll --k0 10000 --k1 0.08 --k2 400 --upper_ques 2 --model gpt-3.5-turbo-0125 --for_testpara 1 --filter_path AmazonGoogle/diverse_structure_0.98_0.08_500_2_2_gpt-3.5-turbo-0125_groups.json > ./Results/OBP/Amazon-Google_0125.log
python main.py --dataset_name AmazonGoogle --batch_method ApproxAll --k0 10000 --k1 0.08 --k2 400 --upper_ques 2 --model gpt-3.5-turbo-0301 --for_testpara 1 --filter_path AmazonGoogle/diverse_structure_0.98_0.08_500_2_2_gpt-3.5-turbo-0125_groups.json > ./Results/OBP/Amazon-Google_0301.log
# Dblp-Scholar
python main.py --dataset_name DblpScholar --batch_method ApproxAll --k0 10 --k1 0.01 --k2 600 --upper_ques 5 --model gpt-3.5-turbo-0125 > ./Results/OBP/Dblp-Scholar_0125.log
python main.py --dataset_name DblpScholar --batch_method ApproxAll --k0 10 --k1 0.01 --k2 600 --upper_ques 5 --model gpt-3.5-turbo-0301 > ./Results/OBP/Dblp-Scholar_0301.log
# AssertTrue
python main.py --dataset_name Atlas_assertTrue --batch_method ApproxAll --batch_type similar --sim_type semantic --k0 0.2 --k1 0.03 --k2 3000 --upper_ques 2 --set_cover_num 0 --output_dir output_0413/group_assignment --model gpt-3.5-turbo-0125 > ./Results/OBP/AssertTrue_0125.log
python main.py --dataset_name Atlas_assertTrue --batch_method ApproxAll --batch_type similar --sim_type semantic --k0 0.2 --k1 0.03 --k2 3000 --upper_ques 2 --set_cover_num 0 --output_dir output_0413/group_assignment --model gpt-3.5-turbo-0301 > ./Results/OBP/AssertTrue_0301.log
# AssertNotNull
python main.py --dataset_name Atlas_assertNotNull --batch_method ApproxAll --batch_type similar --sim_type semantic --k0 0.6 --k1 0.15 --k2 4000 --set_cover_num 0 --upper_ques 3 --model gpt-3.5-turbo-0125 --for_testpara 1 --filter_path output_0413/group_assignment/Atlas_assertNotNull/similar_semantic_0.38_0.1_2000_1_0_gpt-3.5-turbo-0125_groups.json > ./Results/OBP/AssertNotNull_0125.log
python main.py --dataset_name Atlas_assertNotNull --batch_method ApproxAll --batch_type similar --sim_type semantic --k0 0.6 --k1 0.15 --k2 4000 --set_cover_num 0 --upper_ques 3 --model gpt-3.5-turbo-0301 --for_testpara 1 --filter_path output_0413/group_assignment/Atlas_assertNotNull/similar_semantic_0.38_0.1_2000_1_0_gpt-3.5-turbo-0125_groups.json > ./Results/OBP/AssertNotNull_0301.log
# AssertEquals
python main.py --dataset_name Atlas_assertEquals --batch_method ApproxAll --batch_type similar --sim_type semantic --k0 0.3 --k1 0.05 --k2 3000 --upper_ques 2 --set_cover_num 0 --model gpt-3.5-turbo-0125 --for_testpara 1 --filter_path output_0413/group_assignment/Atlas_assertEquals/similar_semantic_0.09_0.015_2000_3_0_gpt-3.5-turbo-0125_groups.json > ./Results/OBP/AssertEquals_0125.log
python main.py --dataset_name Atlas_assertEquals --batch_method ApproxAll --batch_type similar --sim_type semantic --k0 0.35 --k1 0.05 --k2 3000 --upper_ques 3 --set_cover_num 0 --model gpt-3.5-turbo-0301 --for_testpara 1 --filter_path output_0413/group_assignment/Atlas_assertEquals/similar_semantic_0.09_0.015_2000_3_0_gpt-3.5-turbo-0125_groups.json > ./Results/OBP/AssertEquals_0301.log
# AssertThat
python main.py --dataset_name Atlas_assertThat --batch_method ApproxAll --batch_type similar --sim_type semantic --k0 0.11 --k1 0.26 --k2 4000 --set_cover_num 0 --upper_ques 2 --output_dir output_0413/group_assignment --model gpt-3.5-turbo-0125 > ./Results/OBP/AssertThat_0125.log
python main.py --dataset_name Atlas_assertThat --batch_method ApproxAll --batch_type similar --sim_type semantic --k0 0.11 --k1 0.26 --k2 4000 --set_cover_num 0 --upper_ques 2 --output_dir output_0413/group_assignment --model gpt-3.5-turbo-0301 > ./Results/OBP/AssertThat_0301.log