python main.py --base composite --bayes_opt --data_splitter MaterialSplitter --split_ratio 0.3 0.1 0.6
python main.py --base composite --bayes_opt --data_splitter MaterialSplitter

python main.py --base complex_metallic_alloys --bayes_opt --data_splitter MaterialSplitter --split_ratio 0.3 0.1 0.6
python main.py --base complex_metallic_alloys --bayes_opt --data_splitter MaterialSplitter

python main.py --base additively_manufactured_alloys --bayes_opt --data_splitter MaterialSplitter --split_ratio 0.3 0.1 0.6
python main.py --base additively_manufactured_alloys --bayes_opt --data_splitter MaterialSplitter

python main.py --base composite_5mat --bayes_opt --data_splitter StrictCycleSplitter --split_ratio 0.3 0.1 0.6
python main.py --base composite_5mat --bayes_opt --data_splitter StrictCycleSplitter

python main.py --base composite_5mat --bayes_opt --data_splitter RandomSplitter --split_ratio 0.3 0.1 0.6
python main.py --base composite_5mat --bayes_opt --data_splitter RandomSplitter


