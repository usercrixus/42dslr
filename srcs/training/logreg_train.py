from utils.normalize import normalize
from utils.createCsv import createCsv
from utils.train import training

def main():
	df_normalized = normalize("../../datasets/dataset_train.csv")
	createCsv()
	training(df_normalized)

if __name__=='__main__':
    main()