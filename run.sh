# Scenario A
python main.py --base composite --bayes_opt --data_splitter RandomSplitter --split_ratio 0.6 0.2 0.2
python main.py --base composite --bayes_opt --data_splitter RandomSplitter --split_ratio 0.3 0.1 0.6
# Scenario B
python main.py --base composite --bayes_opt --data_splitter StressCycleSplitter --split_ratio 0.6 0.2 0.2 --limit_batch_size 128
python main.py --base composite --bayes_opt --data_splitter StressCycleSplitter --split_ratio 0.3 0.1 0.6 --limit_batch_size 128
# Scenario C
python main.py --base composite --bayes_opt --data_splitter ExtremeCycleSplitter --split_ratio 0.6 0.2 0.2
python main.py --base composite --bayes_opt --data_splitter ExtremeCycleSplitter --split_ratio 0.3 0.1 0.6
# Different clustering schemes in Empiricism
python main_clustering.py --base composite --bayes_opt --data_splitter RandomSplitter
python main_clustering.py --base composite --bayes_opt --data_splitter StressCycleSplitter --limit_batch_size 128
python main_clustering.py --base composite --bayes_opt --data_splitter ExtremeCycleSplitter
# 8:1:1, train once
python main_for_analysis.py --base composite --bayes_opt --data_splitter RandomSplitter
python main_for_analysis.py --base composite --bayes_opt --data_splitter ExtremeCycleSplitter
python main_for_analysis.py --base composite --bayes_opt --data_splitter StressCycleSplitter --limit_batch_size 256
# Train 25 models without derived stress-related features
python main_no_relative_stress.py --base composite_no_relative_stress --bayes_opt --data_splitter RandomSplitter
python main_no_relative_stress.py --base composite_no_relative_stress --bayes_opt --data_splitter StressCycleSplitter --limit_batch_size 128
python main_no_relative_stress.py --base composite_no_relative_stress --bayes_opt --data_splitter ExtremeCycleSplitter
# Without using latent representations
python main.py --base composite --bayes_opt --data_splitter RandomSplitter --split_ratio 0.6 0.2 0.2 --nowrap
python main.py --base composite --bayes_opt --data_splitter StressCycleSplitter --split_ratio 0.6 0.2 0.2 --limit_batch_size 128 --nowrap
python main.py --base composite --bayes_opt --data_splitter ExtremeCycleSplitter --split_ratio 0.6 0.2 0.2 --nowrap
# Without using intermediate outcomes including latent representations
python main.py --base composite --bayes_opt --data_splitter RandomSplitter --split_ratio 0.6 0.2 0.2 --use_raw
python main.py --base composite --bayes_opt --data_splitter StressCycleSplitter --split_ratio 0.6 0.2 0.2 --limit_batch_size 128 --use_raw
python main.py --base composite --bayes_opt --data_splitter ExtremeCycleSplitter --split_ratio 0.6 0.2 0.2 --use_raw
# Utilize lay-up sequence
python main_layup.py --base modulus --bayes_opt --data_splitter RandomSplitter

