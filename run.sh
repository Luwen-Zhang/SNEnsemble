python main.py --base composite --bayes_opt --data_splitter StressCycleSplitter --split_ratio 0.6 0.2 0.2 --limit_batch_size 128
python main.py --base composite --bayes_opt --data_splitter StressCycleSplitter --split_ratio 0.3 0.1 0.6 --limit_batch_size 128

python main.py --base composite --bayes_opt --data_splitter RandomSplitter --split_ratio 0.6 0.2 0.2
python main.py --base composite --bayes_opt --data_splitter RandomSplitter --split_ratio 0.3 0.1 0.6

python main.py --base composite --bayes_opt --data_splitter ExtremeCycleSplitter --split_ratio 0.6 0.2 0.2
python main.py --base composite --bayes_opt --data_splitter ExtremeCycleSplitter --split_ratio 0.3 0.1 0.6

python main_clustering.py --base composite --bayes_opt --data_splitter RandomSplitter
python main_clustering.py --base composite --bayes_opt --data_splitter StressCycleSplitter --limit_batch_size 128
python main_clustering.py --base composite --bayes_opt --data_splitter ExtremeCycleSplitter

python main_for_analysis.py --base composite --bayes_opt --data_splitter RandomSplitter
python main_for_analysis.py --base composite --bayes_opt --data_splitter ExtremeCycleSplitter
python main_for_analysis.py --base composite --bayes_opt --data_splitter StressCycleSplitter --limit_batch_size 256

python main_no_relative_stress.py --base composite_no_relative_stress --bayes_opt --data_splitter RandomSplitter
python main_no_relative_stress.py --base composite_no_relative_stress --bayes_opt --data_splitter StressCycleSplitter --limit_batch_size 128
python main_no_relative_stress.py --base composite_no_relative_stress --bayes_opt --data_splitter ExtremeCycleSplitter

python main_layup.py --base modulus --bayes_opt --data_splitter RandomSplitter

