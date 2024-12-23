
# OBP 
# Beer
python main.py --dataset_name Beer --batch_method ApproxAll --k0 1.9 --k1 0.13 --k2 300 --upper_ques 5 --model gpt-3.5-turbo-0125 --output_dir output/group_assignment > ./Results/Beer_gpt35-0125.log

# # iTunes-Amazon
python main.py --dataset_name iTunesAmazon --batch_method ApproxAll --k0 2.28 --k1 0.02 --k2 300 --upper_ques 20 --model gpt-3.5-turbo-0125 --output_dir output/group_assignment > ./Results/iTunes-Amazon_gpt35-0125.log

# Fodors-Zagats
python main.py --dataset_name FodorsZagats --batch_method ApproxAll --k0 0.98 --k1 0.31 --k2 400 --upper_ques 5 --model gpt-3.5-turbo-0125 --output_dir output/group_assignment > ./Results/Fodors-Zagats_gpt35-0125.log

