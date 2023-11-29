#python main.py --reduce_bayes_steps --base composite --bayes_opt --data_splitter MaterialSplitter --split_ratio 0.3 0.1 0.6
#python main.py --reduce_bayes_steps --base composite --bayes_opt --data_splitter MaterialSplitter

#python main.py --reduce_bayes_steps --base complex_metallic_alloys --bayes_opt --data_splitter MaterialSplitter --split_ratio 0.3 0.1 0.6
#python main.py --reduce_bayes_steps --base complex_metallic_alloys --bayes_opt --data_splitter MaterialSplitter

#python main.py --reduce_bayes_steps --base additively_manufactured_alloys --bayes_opt --data_splitter MaterialSplitter --split_ratio 0.3 0.1 0.6
#python main.py --reduce_bayes_steps --base additively_manufactured_alloys --bayes_opt --data_splitter MaterialSplitter

python main.py --base composite --bayes_opt --data_splitter StressCycleSplitter --split_ratio 0.3 0.1 0.6
python main.py --base composite --bayes_opt --data_splitter StressCycleSplitter

python main.py --base composite --bayes_opt --data_splitter RandomSplitter --split_ratio 0.3 0.1 0.6
python main.py --base composite --bayes_opt --data_splitter RandomSplitter


