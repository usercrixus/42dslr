describe:
	python srcs/plotter/describe.py datasets/dataset_train.csv

plot:
	python srcs/plotter/histogram.py datasets/dataset_train.csv
	python srcs/plotter/scatter_plot.py datasets/dataset_train.csv
	python srcs/plotter/pair_plot.py datasets/dataset_train.csv

train:
	python3 srcs/model/mainTrain.py datasets/dataset_train.csv 

infer:
	python3 srcs/model/mainInfer.py datasets/dataset_test.csv 
