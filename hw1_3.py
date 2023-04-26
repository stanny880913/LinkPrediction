# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 17:15:03 2023

@author: jrwan
"""
import networkx as nx
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import csv
from xgboost import XGBClassifier



# 讀取CSV檔
df = pd.read_csv('new_train_data.csv')

# 建立有向圖
G = nx.Graph()
G1 = nx.DiGraph()

# 加入節點
nodes = set(df['node1']).union(set(df['node2']))
G.add_nodes_from(nodes)
G1.add_nodes_from(nodes)

# 加入邊
for index, row in df.iterrows():
    if row['label'] == 1:
        G.add_edge(row['node1'], row['node2'])
        G1.add_edge(row['node1'], row['node2'])

# 分割訓練集和測試集
train_edges, test_edges, train_labels, test_labels = train_test_split(
    df[['node1', 'node2']].values, df['label'].values, test_size=0.001)

# 提取特徵向量
train_features = []
for edge in train_edges:
    node1 = edge[0]
    node2 = edge[1]
    common_neighbors = len(list(nx.common_neighbors(G, node1, node2)))

    # try:
    #     # 可能引發異常的程式碼
    #     shortest_path = len(nx.shortest_path(G1, node1, node2))
    # except:
    #     # 異常處理的程式碼
    #     shortest_path = 1000000
    # indegree = G1.in_degree(int(node1)) + G1.in_degree(int(node2)) + \
    #     G1.out_degree(int(node1)) + G1.out_degree(int(node2))
    jaccard_coefficient = len(
        list(nx.jaccard_coefficient(G, [(node1, node2)])))
    train_features.append([common_neighbors, jaccard_coefficient])
    #train_features.append([common_neighbors, jaccard_coefficient, indegree, shortest_path])

test_features = []
for edge in test_edges:
    node1 = edge[0]
    node2 = edge[1]
    common_neighbors = len(list(nx.common_neighbors(G, node1, node2)))
    # try:
    #     # 可能引發異常的程式碼
    #     shortest_path = len(nx.shortest_path(G1, node1, node2))
    # except:
    #     # 異常處理的程式碼
    #     shortest_path = 1000000

    # indegree = G1.in_degree(int(node1)) + G1.in_degree(int(node2)) + \
    #     G1.out_degree(int(node1)) + G1.out_degree(int(node2))
    jaccard_coefficient = len(
        list(nx.jaccard_coefficient(G, [(node1, node2)])))
    test_features.append([common_neighbors, jaccard_coefficient])
    #test_features.append([common_neighbors, jaccard_coefficient, indegree, shortest_path])

# 訓練模型
model = LogisticRegression()
model.fit(train_features, train_labels)

# 預測測試集
test_preds = model.predict(test_features)



# 計算準確率
accuracy = accuracy_score(test_labels, test_preds)
print('Accuracy:', accuracy)


df1 = pd.read_csv('new_test_data.csv')
test_edges1 = df1[['node1', 'node2']].values


nodes1 = set(df1['node1']).union(set(df1['node2']))
G.add_nodes_from(nodes1)
G1.add_nodes_from(nodes1)

test_features1 = []
for node in test_edges1:
    node1 = node[0]
    node2 = node[1]
    common_neighbors = len(list(nx.common_neighbors(G, node1, node2)))
    # try:
    #     # 可能引發異常的程式碼
    #     shortest_path = len(nx.shortest_path(G1, node1, node2))
    # except:
    #     # 異常處理的程式碼
    #     shortest_path = 1000000
    # indegree = G1.in_degree(int(node1)) + G1.in_degree(int(node2)) + \
    #     G1.out_degree(int(node1)) + G1.out_degree(int(node2))
    # print(common_neighbors)
    jaccard_coefficient = len(
        list(nx.jaccard_coefficient(G, [(node1, node2)])))
    test_features1.append([common_neighbors, jaccard_coefficient])
    #test_features1.append([common_neighbors, jaccard_coefficient, indegree, shortest_path])

test_preds1 = model.predict(test_features1)
print(test_preds1)
print(sum(test_preds1))

output = "ans.csv"

with open(output, 'a') as f:
    index = 0
    for i in test_preds1:
        csv_write = csv.writer(f)
        csv_write.writerow([index, i])
        index += 1
        # csv_write.writecol(test_preds1[i])
    print("finished")
