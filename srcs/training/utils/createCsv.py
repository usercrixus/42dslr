import pandas as pd
import os

def createCsv():
	if os.path.exists('dataset.csv'):
		return
	data = {
    'Id': [1, 2, 3, 4],
    'Biais': [0, 0, 0, 0],
    'Arithmancy': [0, 0, 0, 0],
    'Herbology': [0, 0, 0, 0],
    'Defense Against the Dark Arts': [0, 0, 0, 0],
	'Divination': [0, 0, 0, 0],
	'Muggle Studies': [0, 0, 0, 0],
	'Ancient Runes': [0, 0, 0, 0],
	'History of Magic': [0, 0, 0, 0],
	'Transfiguration': [0, 0, 0, 0],
	'Potions': [0, 0, 0, 0],
	'Care of Magical Creatures': [0, 0, 0, 0],
	'Charms': [0, 0, 0, 0],
	'Flying': [0, 0, 0, 0]
	}
	df = pd.DataFrame(data)
	df.to_csv('dataset.csv', index=False)