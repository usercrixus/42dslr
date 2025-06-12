import sys
import numpy as np
import pandas as pd

def softmax(z):
	exp_z = np.exp(z - np.max(z))
	return exp_z / exp_z.sum()

def main(dataset, data_test):
	df_dataset = pd.read_csv(dataset)
	df_dataset = df_dataset.drop(columns=['Id'], errors='ignore')
	df_datatest = pd.read_csv(data_test)
	df_datatest = df_datatest.drop(columns=[
		'Index',
		'First Name',
		'Last Name',
		'Best Hand',
		'Hogwarts House',
		'Birthday'
	], errors='ignore')
	df_datatest['Biais'] = 1
	cols = ['Biais'] + [col for col in df_datatest.columns if col != 'Biais']
	df_datatest = df_datatest[cols]
	for j in range(len(df_datatest)):
		z = [0, 0, 0, 0]
		line = df_datatest.iloc[j]
		for i in range(13): # 12 features + 1 biais
			feature = line.iloc[i]
			for k in range(4):
				weight = df_dataset.iat[k, i]
				z[k] += weight * feature
		p = softmax(z)
		predicted_index = np.argmax(p)
		houses = ["Ravenclaw", "Gryffindor", "Slytherin", "Hufflepuff"]
		predicted_house = houses[predicted_index]
		print(f"Ligne {j} prédite : {predicted_house}")
	
if __name__=='__main__':
    if len(sys.argv) != 3:
        print("Usage : python script.py arg1 arg2")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])