# OBP for ER tasks : Some examples

# # Beer
python main.py --dataset_name Beer --batch_method ApproxAll --k0 1.9 --k1 0.13 --k2 300 --upper_ques 5 --model gpt-3.5-turbo-0125 --output_dir output/group_assignment > ./Results/Beer_gpt35-0125_1.9_0.13_300_5.log

python main.py --dataset_name Beer --batch_method ApproxAll --k0 1.9 --k1 0.13 --k2 300 --upper_ques 4 --model gpt-3.5-turbo-0125 --output_dir output/group_assignment > ./Results/Beer_gpt35-0125_1.9_0.13_300_4.log

# Walmart-Amazon
python main.py --dataset_name WalmartAmazon --batch_method ApproxAll --k0 1.42 --k1 0.19 --k2 800 --upper_ques 5 --model gpt-3.5-turbo --output_dir output/group_assignment > ./Results/Walmart-Amazon_gpt35_1.42_0.19_800_5.log 

python main.py --dataset_name WalmartAmazon --batch_method ApproxAll --k0 1.42 --k1 0.19 --k2 800 --upper_ques 4 --model gpt-3.5-turbo --output_dir output/group_assignment > ./Results/Walmart-Amazon_gpt35_1.42_0.19_800_4.log 
