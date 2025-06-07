init:
	python3 -m venv venv; source venv/bin/activate; pip install pandas; pip install matplotlib

plot:
	python srcs/plotter/histogram.py datasets/dataset_train.csv
	python srcs/plotter/scatter_plot.py datasets/dataset_train.csv
	python srcs/plotter/pair_plot.py datasets/dataset_train.csv

clean:
	rm -fR histograms pair_plots scatter