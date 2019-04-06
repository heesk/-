import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ctx =r'C:/Users/ezen/PycharmProjects/2019.03.30_1/titanic/data/'
train=pd.read_csv(ctx+'train.csv')
test=pd.read_csv(ctx+'test.csv')
df=pd.DataFrame(train)
print(df.columns)
"""
['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
"""
"""
PassengerId 고객아이디
Survived 생존여부, Survival 0=No,1=Yes
Pclass 승선권 클래스 Ticket class 1=1st, 2=2nd, 3=3rd
Name 이름
Sex 성별 Sex
Age 나이 Age in years
SibSp 동반한 형제자매, 배우자 수 # of siblins
Parch
Ticket
Fare
Cabin
Embarked
"""
f, ax=plt.subplots(1,2,figsize=(18,8))
train['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct="%1.1f%%", ax=ax[0], shadow=True)
ax[0].set_title('Survived')
ax[0].set_ylabel('')
sns.countplot('Survived', data=train, ax=ax[1])
ax[1].set_title('Survived')
plt.show()
#************
#성별 : 고쳐야함
#***************
"""
f, ax=plt.subplots(1,2,figsize=(18,8))
train['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct="%1.1f%%", ax=ax[0], shadow=True)
ax[0].set_title('Survived')
ax[0].set_ylabel('')
sns.countplot('Survived', data=train, ax=ax[1])
ax[1].set_title('Survived')
plt.show()
"""
#************
#승선권 Pclass
#***************
df_1=[train['Sex'],train['Survived']]
df_2=train['Pclass']
df=pd.crosstab(df_1, df_2, margins=True)
print(df.head())


#결측치 제거
train.info()
train.isnull().sum()
print(train.isnull().sum())

def bar_chart(feature):
    survived=train[train['Survived']] ==1][feature].value_counts()
    dead=train[train['Survived']]==0][feature].value_counts()
    df=pd.DataFrame([survived,dead])
    df.index=['survived','dead']
    df.plot(kind='bar', stacked=True, figsize=(10,5))
    plt.show()  # 예제에는 빠졌어도 넣어야 함
bar_chart('Sex')
bar_chart('Pclass')  # 승선권 클래스
 # 사망한 사람은 3등석, 생존한 사람은 1등석이 많음
bar_chart('SibSp')  # 동반한 형제자매, 배우자 수
bar_chart('Parch')  # 동반한 부모, 자식 수
bar_chart('Embarked')  # 승선한 항구명

train = train.drop(['Cabin'], axis=1)
test = test.drop(['Cabin'], axis=1)
train = train.drop(['Ticket'], axis=1)
test = test.drop(['Ticket'], axis=1)
train.head()
test.head()
#Embarked값 가공
s_city=train[train["Embarked"]=='S'].shape[0]#스칼라
c_city = train[train["Embarked"] == 'C'].shape[0]
q_city = train[train["Embarked"] == 'Q'].shape[0]
print("S={}, C={}, Q={}", format(s_city,c_city,q_city))

