import os
import pandas as pd
df=pd.read_csv("321.csv",encoding="gb2312")
print("123")
print(df.head(1))
print(df["age"].mean()) #平均值
print(df["age"].median()) #中位数
print(df["age"].max())  #最大值
print(df["age"].quantile(q=0.25)) #分位数
print(df["Hobby"].mode())   #众数
print(df["Hobby"].std())  #标准差
print(df["Hobby"].var())  #方差

# data = pd.read_csv(r'321.csv')   #打开一个csv，得到data对象
# print(data.columns)#获取列索引值
# data1 = 'China','China','US','US'#获取name列的数据
# data['Nation'] = data1 #将数据插入新列new
# data.to_csv(r"1.csv",mode = 'a',index =False)
# #保存到csv,  mode=a，以追加模式写入,header表示列名，默认为true,index表示行名，默认为true，再次写入不需要行名
# print(data)
# os.remove('321.csv')
