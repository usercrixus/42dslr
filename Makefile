describe:
	python srcs/plotter/describe.py datasets/dataset_train.csv

plot:
	python srcs/plotter/histogram.py datasets/dataset_train.csv
	python srcs/plotter/scatter_plot.py datasets/dataset_train.csv
	python srcs/plotter/pair_plot.py datasets/dataset_train.csv

train:
	python3 srcs/model/logreg_train.py datasets/dataset_train.csv 

predict:
	python3 srcs/model/logreg_predict.py datasets/dataset_test.csv 

clean:
	rm -fR *.npy
