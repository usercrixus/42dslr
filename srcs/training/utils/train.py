import pandas as pd
import numpy as np

learning_rate = 0.1

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def getResult(j):
	df = pd.read_csv('../../datasets/dataset_train.csv')
	result = df.at[j, "Hogwarts House"]
	return result

def training(df_normalized):
	df = pd.read_csv('dataset.csv')
	df = df.drop(columns=['Id'], errors='ignore')
	df = df.drop(columns=['Biais'], errors='ignore')
	print(df.head())

	for j in range(len(df_normalized)):
		z = [0, 0, 0, 0]
		for i in range(12):
			for k in range(4):
				z[k] += df.iat[k, i] * df_normalized.iat[j, i]

		p1 = sigmoid(z[0])
		p2 = sigmoid(z[1])
		p3 = sigmoid(z[2])
		p4 = sigmoid(z[3])
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

		print(p1)
		# for i in range(12):
		# 	for k in range(4):
		# 		error = p[k] - y[k]
		# 		df.iat[k, i] -= learning_rate * error * df_normalized.iat[j, i]

