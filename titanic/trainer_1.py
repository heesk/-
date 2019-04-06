import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ctx = r'C:/Users/ezen/PycharmProjects/2019.03.30_1/titanic/data/'
train = pd.read_csv(ctx + 'train.csv')
test = pd.read_csv(ctx + 'test.csv')
# df=pd.DataFrame(train)
# print(df.columns)
train.head()
train=train.drop([''])