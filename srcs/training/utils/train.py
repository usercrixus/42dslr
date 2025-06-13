import pandas as pd
import numpy as np

learning_rate = 0.1

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def softmax(z):
	exp_z = np.exp(z - np.max(z))  # pour stabilité numérique
	return exp_z / np.sum(exp_z)

def getResult(j):
	df = pd.read_csv('../../datasets/dataset_train.csv')
	result = df.at[j, "Hogwarts House"]
	return result

def training(df_normalized):
	df_normalized.fillna(0, inplace=True)
	df = pd.read_csv('dataset.csv')
	df = df.astype(float)
	df = df.drop(columns=['Id'], errors='ignore')
	for epoch in range(5):
		accuracy = 0
		for j in range(len(df_normalized)):
			z = [0, 0, 0, 0]
			line = df_normalized.iloc[j]
			for i in range(13): # 12 features + 1 biais
				feature = line.iloc[i]
				for k in range(4):
					weight = df.iat[k, i]
					z[k] += weight * feature

			p = softmax(z)
			y = [0, 0, 0, 0]
			good_result = getResult(j)
			if good_result == "Ravenclaw":
				y[0] = 1
			elif good_result == "Slytherin":
				y[1] = 1
			elif good_result == "Gryffindor":
				y[2] = 1
			elif good_result == "Hufflepuff":
				y[3] = 1
			else:
				continue

			for i in range(13):
				for k in range(4):
					error = p[k] - y[k]
					df.iat[k, i] -= learning_rate * error * line.iloc[i]
			id = p.argmax()
			if id == 0 and good_result == "Ravenclaw":
				accuracy += 1
			if id == 1 and good_result == "Slytherin":
				accuracy += 1
			if id == 2 and good_result == "Gryffindor":
				accuracy += 1
			if id == 3 and good_result == "Hufflepuff":
				accuracy += 1
		accuracy = (accuracy / len(df_normalized)) * 100
		print(f"Accuracy : {accuracy}")
	df['Id'] = [1, 2, 3, 4]
	cols = ['Id'] + [col for col in df.columns if col not in ['Id']]
	df = df[cols]
	df.to_csv('dataset.csv', index=False)
