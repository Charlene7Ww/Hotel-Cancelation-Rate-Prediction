#!/usr/bin/env python
# coding: utf-8

# In[1]:



import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 32)


import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import plotly.express as px
get_ipython().run_line_magic('matplotlib', 'inline')


import missingno as msno

import folium
from folium.plugins import HeatMap


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, precision_score, f1_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier


import warnings
warnings.filterwarnings('ignore')


# In[2]:


df=pd.read_csv('hotel_bookings.csv')
df.head()


# In[3]:



df.columns


# 字段的具体中文含义：
# 
# hotel 酒店
# 
# is_canceled 是否取消
# 
# lead_time 预订时间
# 
# arrival_date_year 入住年份
# 
# arrival_date_month 入住月份
# 
# arrival_date_week_number 入住周次
# 
# arrival_date_day_of_month 入住天号
# 
# stays_in_weekend_nights 周末夜晚数
# 
# stays_in_week_nights 工作日夜晚数
# adults 成人数量
# 
# children 儿童数量
# 
# babies 幼儿数量
# 
# meal 餐食
# 
# country 国家
# 
# market_segment 细分市场
# 
# distribution_channel 分销渠道
# 
# is_repeated_guest 是否是回头客
# 
# previous_cancellations 先前取消数
# 
# previous_bookings_not_canceled 先前未取消数
# 
# reserved_room_type 预订房间类型
# 
# assigned_room_type 实际房间类型
# 
# booking_changes 预订更改数
# 
# deposit_type 押金方式
# 
# agent 代理
# 
# company 公司
# 
# days_in_waiting_list 排队天数
# 
# customer_type 客户类型
# 
# adr 每日房间均价 （Average Daily Rate）
# 
# required_car_parking_spaces 停车位数量
# 
# total_of_special_requests 特殊需求数(例如高层或双床)
# 
# reservation_status 订单状态
# 
# reservation_status_date 订单状态确定日期

# In[4]:



len(df.columns)


# In[5]:



df.dtypes


# In[6]:



df.dtypes.value_counts()


# In[7]:



df.shape


# In[8]:



df.describe()


# In[9]:



df.info()


# ### 缺失值信息
# 
# #### 统计每个字段缺失值信息
# 
# 统计每个字段的缺失值数量及比例

# In[10]:


null_df = pd.DataFrame({"Null Values": df.isnull().sum(),
                         "Percentage Null Values": (df.isnull().sum()) / (df.shape[0]) * 100
                         })

null_df


# ### 缺失值可视化
# #### 将缺失值信息进行可视化展示：

# In[11]:


msno.bar(df, color="blue")

plt.show()


# 缺失值处理
# 
# 1、字段 children和字段country 缺失值比例都不到1%，比例很小；直接把缺失值的部分删除

# In[12]:




df = df[df["country"].isnull() == False]
df = df[df["children"].isnull() == False]

df.head()


# 2、字段company缺失值比例高达94.3%，考虑直接删除该字段：

# In[13]:


df.drop("company", axis=1, inplace=True)


# 3、字段agent(代理商费用)的缺失值为13.68%，处理为：

# In[14]:



df["agent"].value_counts()


# 我们可以考虑使用的值来进行填充，比如：
# 
# 0：无法确定缺失值的具体数据
# 
# 9：众数
# 
# 均值：字段现有值的均值
# 
# 在这里我们考虑使用0来进行填充：

# In[15]:


df["agent"].fillna(0,inplace=True)


# ### 特殊处理
# 
# 处理1：入住人数不能为0
# 
# 考虑到一个房间中adults、children和babies的数量不能同时为0：|

# In[16]:


special = (df["children"] == 0) & (df.adults == 0) & (df.babies == 0)
special.head()


# In[17]:



df = df[~special
       3
       ]


# ### 处理2：adr（日均价）
# 
# 取值不能为负数
# 
# 最大值为5400，可以判断属于异常值

# In[18]:


df["adr"].value_counts().sort_index()


# In[19]:



px.violin(y=df["adr"])   # 处理前


# In[20]:


px.box(df,y="adr")


# In[21]:


# 删除大于1000的信息  df = df.drop(df[df.adr >1000].index)

df = df[(df["adr"] >= 0) & (df["adr"] < 5400)]  # 排除异常值


# In[22]:


px.violin(y=df["adr"])  # 删除后


# In[23]:


px.box(df,y="adr",color="hotel")   # 删除后


# ## 数据EDA-Exploratory Data Analysis
# ### 取消和未取消的顾客数对比

# In[24]:


df["is_canceled"].value_counts()


# In[25]:


# 取消和未取消人数对比  0-未取消 1-取消
sns.countplot(df["is_canceled"])

plt.show()


# In[26]:



data = df[df.is_canceled == 0]  # 未取消的数据


# In[27]:


number_no_canceled = data["country"].value_counts().reset_index()
number_no_canceled.columns = ["country", "number_of_no_canceled"]

number_no_canceled


# In[28]:


# 地图可视化

basemap = folium.Map()
guests_map = px.choropleth(number_no_canceled, # 传入数据
                           locations = number_no_canceled['country'],  # 地理位置
                           color = number_no_canceled['number_of_no_canceled'],  # 颜色取值
                           hover_name = number_no_canceled['country'])  # 悬停信息
guests_map.show()


# ### 结论1：预订的顾客主要是来自Portugal，大部分是欧洲的国家

# In[29]:


#房间的每日均价是多少？

px.box(data,  # 数据
       x="reserved_room_type",  # x
       y="adr", # y
       color="hotel",  # 颜色
       template="plotly_dark",  # 主题
       category_orders={"reserved_room_type":["A","B","C","D","E","F","G","H","L"]} # 指定排列顺序
      )


# ### 结论2：每个房间的均价还是取决于它的类型和标准差

# 全年每晚的价格是多少？
# 
# 两种不同类型酒店的全年均价变化

# In[30]:


data_resort = data[data["hotel"] == "Resort Hotel"]
data_city = data[data["hotel"] == "City Hotel"]


# In[31]:


resort_hotel = data_resort.groupby(['arrival_date_month'])['adr'].mean().reset_index()
city_hotel = data_city.groupby(['arrival_date_month'])['adr'].mean().reset_index()
city_hotel


# In[32]:




total_hotel = pd.merge(resort_hotel, city_hotel,
                        on="arrival_date_month"
                        )
total_hotel.columns = ["month","price_resort","price_city"]
total_hotel


# 

# In[33]:


#!pip install sort-dataframeby-monthorweek
#!pip install sorted-months-weekdays


# In[34]:


import sort_dataframeby_monthorweek as sd

#  自定义排序函数
def sort_month(df, column):
    result = sd.Sort_Dataframeby_Month(df,column)
    return result


# In[35]:


new_total_hotel = sort_month(total_hotel, "month")
new_total_hotel


# In[36]:


fig = px.line(new_total_hotel,
        x = "month",
        y = ["price_resort", "price_city"],
        title = "Price of per night over the Months",
        template = "plotly_dark"      
       )

fig.show()


# ### 结论：
# 
# Resort Hotel在夏季的价格明显比 City Hotel的价格高
# 
# City Hotel的价格变化相对更小。但是City Hotel的价格从4月开始就已经很高，一直持续到9月份
# 
# ### KDE图
# 
# KDE(Kernel Density Estimation，核密度图)，可以认为是对直方图的加窗平滑。通过KDE分布图场内看数据在不同情形下的分布

# In[37]:


plt.figure(figsize=(6,3), dpi=150)

ax = sns.kdeplot(new_total_hotel["price_resort"], 
                 color="green", 
                 shade=True)

ax = sns.kdeplot(new_total_hotel["price_city"], 
                 color="blue", 
                 shade=True)

ax.set_xlabel("month")
ax.set_ylabel("Price per night over the month")
ax = ax.legend(["Resort","City"])


# ### 最为繁忙的季节-the most busy months

# In[38]:


resort_guests = data_resort['arrival_date_month'].value_counts().reset_index()
resort_guests.columns=['Month','No_Resort_Guests']

city_guests = data_city['arrival_date_month'].value_counts().reset_index()
city_guests.columns=['Month','No_City_Guests']


final_guests = pd.merge(resort_guests, city_guests)


# In[39]:


#同样的将月份进行排序处理
new_final_guests = sort_month(final_guests, "Month")
new_final_guests


# In[40]:


fig = px.line(new_final_guests,
        x = "Month",
        y = ["No_Resort_Guests", "No_City_Guests"],
        title = "No of per Month",
        template = "plotly_dark"      
       )

fig.show()


# ### 结论：
# 
# 很明显：City Hotel的人数是高于Resort Hotel，更受欢迎
# 
# City Hotel在7-8月份的时候，尽管价格高（上图），但人数也达到了峰值
# 
# 两个Hotel在冬季的顾客都是很少的
# 
# ### 顾客停留多久？

# In[41]:


data["total_nights"] = data['stays_in_weekend_nights'] + data['stays_in_week_nights']
data.head()


# In[42]:


#两个不同酒店在不同停留时间下的统计：

stay_groupby = (data.groupby(['total_nights', 'hotel'])["is_canceled"]
                .agg("count")
                .reset_index()
                .rename(columns={"is_canceled":"Number of stays"}))

stay_groupby.head()


# In[43]:


stay_groupby = (data.groupby(['total_nights', 'hotel'])["is_canceled"]
                .agg("count")
                .reset_index()
                .rename(columns={"is_canceled":"Number of stays"}))

stay_groupby.head()


# In[44]:


fig = px.bar(stay_groupby,
       x = "total_nights",
       y = "Number of stays",
       color = "hotel",
       barmode = "group"
      )

fig.show()


# ### 数据预处理-Data Pre Processing
# ### 相关性判断

# In[45]:


plt.figure (figsize=(24,12))

corr = df.corr()
sns.heatmap(corr, annot = True, linewidths = 1)
plt.show()


# In[46]:


#查看每个特征和目标变量is_canceled的相关系数的绝对值，并降序排列：
corr_with_iscanceled = df.corr()["is_canceled"].abs().sort_values(ascending=False)

corr_with_iscanceled


# In[47]:


#删除无效字段
no_use_col = ['arrival_date_year', 'assigned_room_type',
             'booking_changes','reservation_status', 
             'country', 'days_in_waiting_list']


# In[48]:


df.drop(no_use_col, axis=1, inplace=True)


# ### 特征工程
# 
# 离散型变量处理

# In[49]:


df["hotel"].dtype # Series型数据的字段类型


# In[50]:


cat_cols = [col for col in df.columns if df[col].dtype == "O"]
cat_cols


# In[51]:


cat_df = df[cat_cols]


# In[52]:


cat_df.dtypes


# In[53]:


# 1、转成时间类型数据

cat_df['reservation_status_date'] = pd.to_datetime(cat_df['reservation_status_date'])


# In[54]:


#  2、提取年月日

cat_df["year"] = cat_df['reservation_status_date'].dt.year
cat_df['month'] = cat_df['reservation_status_date'].dt.month
cat_df['day'] = cat_df['reservation_status_date'].dt.day


# In[55]:


# 3、删除无效字段
cat_df.drop(['reservation_status_date','arrival_date_month'], axis=1, inplace=True)


# In[56]:


# 4、每个字段的唯一值

for col in cat_df.columns:
    print(f"{col}: \n{cat_df[col].unique()}\n")


# ### 特征编码

# In[57]:


# 酒店
cat_df['hotel'] = cat_df['hotel'].map({'Resort Hotel' : 0, 
                                       'City Hotel' : 1})
# 餐食
cat_df['meal'] = cat_df['meal'].map({'BB' : 0, 
                                     'FB': 1, 
                                     'HB': 2, 
                                     'SC': 3, 
                                     'Undefined': 4})
# 细分市场
cat_df['market_segment'] = (cat_df['market_segment']
                            .map({'Direct': 0,
                                 'Corporate':1,
                                 'Online TA':2, 
                                 'Offline TA/TO': 3,
                                 'Complementary': 4,
                                 'Groups': 5,
                                 'Undefined': 6,
                                 'Aviation': 7}))
# 分销渠道
cat_df['distribution_channel'] = (cat_df['distribution_channel']
                                  .map({'Direct': 0,
                                        'Corporate': 1,
                                        'TA/TO': 2, 
                                        'Undefined': 3,
                                        'GDS': 4}))
# 预订房间类型
cat_df['reserved_room_type'] = (cat_df['reserved_room_type']
                                .map({'C': 0, 
                                      'A': 1, 
                                      'D': 2, 
                                      'E': 3, 
                                      'G': 4, 
                                      'F': 5, 
                                      'H': 6,
                                      'L': 7, 
                                      'B': 8}))
# 押金方式
cat_df['deposit_type'] = (cat_df['deposit_type']
                          .map({'No Deposit': 0, 
                                'Refundable': 1, 
                                'Non Refund': 3}))
# 顾客类型
cat_df['customer_type'] = (cat_df['customer_type']
                           .map({'Transient': 0, 
                                 'Contract': 1, 
                                 'Transient-Party': 2, 
                                 'Group': 3})
                          )
# 年份
cat_df['year'] = cat_df['year'].map({2015: 0, 2014: 1, 2016: 2, 2017: 3})


# ### 连续型变量处理

# In[58]:


num_df = df.drop(columns=cat_cols,axis=1)

num_df.drop("is_canceled",axis=1,inplace=True)


# In[59]:


# 方差偏大的字段进行对数化处理
log_col = ["lead_time","arrival_date_week_number","arrival_date_day_of_month","agent","adr"]

for col in log_col:
    num_df[col] = np.log(num_df[col] + 1)
    
num_df.head()


# ### 建模
# ### 合并两份df

# In[60]:


X = pd.concat([cat_df, num_df], axis=1)
y = df["is_canceled"]


# In[61]:


print(X.shape)
print(y.shape)


# In[62]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state=412)


# ### 建模1：逻辑回归

# In[63]:


# 模型实例化
lr = LogisticRegression()
lr.fit(X_train, y_train)

# 预测值
y_pred_lr = lr.predict(X_test)

# 分类问题不同评价指标
acc_lr = accuracy_score(y_test, y_pred_lr)
conf = confusion_matrix(y_test, y_pred_lr)
clf_report = classification_report(y_test, y_pred_lr)

print(f"Accuracy Score of Logistic Regression is : {acc_lr}")
print(f"Confusion Matrix : \n{conf}")
print(f"Classification Report : \n{clf_report}")


# In[64]:


# 混淆矩阵可视化

classes = ["0","1"]

disp = ConfusionMatrixDisplay(confusion_matrix=conf, display_labels=classes)
disp.plot(
    include_values=True,            # 混淆矩阵每个单元格上显示具体数值
    cmap="GnBu",                 # matplotlib识别的颜色图
    ax=None,
    xticks_rotation="horizontal",
    values_format="d"
)

plt.show()


# In[65]:


#KNN 模型
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

y_pred= knn.predict(X_test)

acc_knn = accuracy_score(y_test, y_pred)
conf = confusion_matrix(y_test, y_pred)
clf_report = classification_report(y_test, y_pred)


# In[66]:


#决策树模型
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

y_pred_dtc = dtc.predict(X_test)

acc_dtc = accuracy_score(y_test, y_pred_dtc)
conf = confusion_matrix(y_test, y_pred_dtc)
clf_report = classification_report(y_test, y_pred_dtc)


# In[67]:


#随机森林
rd_clf = RandomForestClassifier()
rd_clf.fit(X_train, y_train)

y_pred_rd_clf = rd_clf.predict(X_test)

acc_rd_clf = accuracy_score(y_test, y_pred_rd_clf)
conf = confusion_matrix(y_test, y_pred_rd_clf)
clf_report = classification_report(y_test, y_pred_rd_clf)


# In[68]:


#Adaboost模型
ada = AdaBoostClassifier(base_estimator = dtc)
ada.fit(X_train, y_train)

y_pred_ada = ada.predict(X_test)

acc_ada = accuracy_score(y_test, y_pred_ada)
conf = confusion_matrix(y_test, y_pred_ada)
clf_report = classification_report(y_test, y_pred_ada)


# In[69]:


#梯度提升树-Gradient Boosting Classifier
gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)

y_pred_gb = gb.predict(X_test)

acc_gb = accuracy_score(y_test, y_pred_gb)
conf = confusion_matrix(y_test, y_pred_gb)
clf_report = classification_report(y_test, y_pred_gb)


# In[70]:


#XGBoost模型
xgb = XGBClassifier(booster='gbtree', 
                    learning_rate=0.1, 
                    max_depth=5, 
                    n_estimators=180)

xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)

acc_xgb = accuracy_score(y_test, y_pred_xgb)
conf = confusion_matrix(y_test, y_pred_xgb)
clf_report = classification_report(y_test, y_pred_xgb)


# In[71]:


#CatBoost 模型
cat = CatBoostClassifier(iterations=100)
cat.fit(X_train, y_train)

y_pred_cat = cat.predict(X_test)

acc_cat = accuracy_score(y_test, y_pred_cat)
conf = confusion_matrix(y_test, y_pred_cat)
clf_report = classification_report(y_test, y_pred_cat)


# In[72]:


#极端树-Extra Trees Classifier
etc = ExtraTreesClassifier()
etc.fit(X_train, y_train)

y_pred_etc = etc.predict(X_test)

acc_etc = accuracy_score(y_test, y_pred_etc)
conf = confusion_matrix(y_test, y_pred_etc)
clf_report = classification_report(y_test, y_pred_etc)


# In[73]:


#LGBM
lgbm = LGBMClassifier(learning_rate = 1)
lgbm.fit(X_train, y_train)

y_pred_lgbm = lgbm.predict(X_test)

acc_lgbm = accuracy_score(y_test, y_pred_lgbm)
conf = confusion_matrix(y_test, y_pred_lgbm)
clf_report = classification_report(y_test, y_pred_lgbm)


# In[74]:


#模型11：投票分类器-Voting Classifier
#这个是重点建模：多分类器的投票表决

classifiers = [('Gradient Boosting Classifier', gb), 
               ('Cat Boost Classifier', cat), 
               ('XGboost', xgb),  
               ('Decision Tree', dtc),
               ('Extra Tree', etc), 
               ('Light Gradient', lgbm), 
               ('Random Forest', rd_clf), 
               ('Ada Boost', ada), 
               ('Logistic', lr),
               ('Knn', knn)]

vc = VotingClassifier(estimators = classifiers)
vc.fit(X_train, y_train)

y_pred_vc = vc.predict(X_test)

acc_vtc = accuracy_score(y_test, y_pred_vc)
conf = confusion_matrix(y_test, y_pred_vc)
clf_report = classification_report(y_test, y_pred_vc)


# ### 基于深度学习keras建模
# ### 数据预处理和切割

# In[75]:


from tensorflow.keras.utils import to_categorical

X = pd.concat([cat_df, num_df], axis = 1)
# 转成分类型变量数据
y = to_categorical(df['is_canceled'])


# In[76]:


# 切割数据

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[77]:


import tensorflow as tf
import keras
from keras.layers import Dense
from keras.models import Sequential


# In[78]:


X.shape[1]


# In[79]:


model = Sequential()

model.add(Dense(100, activation="relu",input_shape=(X.shape[1], )))
model.add(Dense(100, activation="relu"))
model.add(Dense(2, activation="sigmoid"))

model.compile(optimizer="adam", 
              loss="binary_crossentropy",
              metrics=["accuracy"])

model_history = model.fit(X_train, 
                          y_train, 
                          validation_data = (X_test, y_test),
                          epochs = 50)


# In[80]:


#指标可视化-loss
train_loss = model_history.history["loss"]
val_loss = model_history.history["val_loss"]

epoch = range(1,51)

loss = pd.DataFrame({"train_loss": train_loss,
                     "val_loss":val_loss
                    })
loss.head()


# In[81]:


train_loss = model_history.history["loss"]
val_loss = model_history.history["val_loss"]

epoch = range(1,51)

loss = pd.DataFrame({"train_loss": train_loss,
                     "val_loss":val_loss
                    })
loss.head()


# In[82]:


fig = px.line(loss, 
        x=epoch, 
        y=['val_loss','train_loss'], 
        title='Train and Val Loss')

fig.show()


# In[83]:


#指标可视化-acc
train_acc = model_history.history["accuracy"]
val_acc = model_history.history["val_accuracy"]

epoch = range(1,51)

acc = pd.DataFrame({"train_acc": train_acc,
                     "val_acc":val_acc
                    })

px.line(acc, 
        x=epoch, 
        y=['val_acc','train_acc'], 
        title = 'Train and Val Accuracy',
        template = 'plotly_dark')


# In[84]:


#最终预测值

acc_ann = model.evaluate(X_test, y_test)[1]
acc_ann


# ### 模型对比
# #### 不同模型的结果对比

# In[85]:


models = pd.DataFrame({
    'Model' : ['Logistic Regression', 'KNN', 
               'Decision Tree Classifier', 
               'Random Forest Classifier',
               'Ada Boost Classifier',
               'Gradient Boosting Classifier', 
               'XgBoost', 'Cat Boost', 
               'Extra Trees Classifier', 
               'LGBM', 'Voting Classifier','ANN'],
    'Score' : [acc_lr, acc_knn, acc_dtc, 
               acc_rd_clf, acc_ada, acc_gb, 
               acc_xgb, acc_cat, acc_etc, 
               acc_lgbm, acc_vtc, acc_ann]
})


models = models.sort_values(by = 'Score', ascending = True, ignore_index=True)

models["Score"] = models["Score"].apply(lambda x: round(x,4))
models


# In[86]:


fig = px.bar(models,
       x="Score",
       y="Model",
       text="Score",
       color="Score",
       template="plotly_dark",
       title="Models Comparision"
      )

fig.show()


# ### Cat Boost分类达到了99.61%！

# In[ ]:




