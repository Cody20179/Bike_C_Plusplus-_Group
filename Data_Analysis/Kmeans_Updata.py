import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 資料讀取
df = pd.read_csv('All_caculate.csv')
df_data = df.iloc[:, 1:]

# 2. 資料清洗
# 去除none值和'[]'字符串
df_data.replace('[]', np.nan, inplace=True)
df_data.dropna(inplace=True)

# 檢查是否所有的列都是數值型，並轉換
for col in df_data.columns:
    df_data[col] = pd.to_numeric(df_data[col], errors='coerce')

# 再次去除none值（因為可能會因為轉換失敗出現NaN值）
df_data.dropna(inplace=True)

# 去除極值 - 使用IQR方法
Q1 = df_data.quantile(0.25)
Q3 = df_data.quantile(0.75)
IQR = Q3 - Q1

df_data_no_outliers = df_data[~((df_data < (Q1 - 1.5 * IQR)) | (df_data > (Q3 + 1.5 * IQR))).any(axis=1)]

# 3. 正規化或標準化
scaler = StandardScaler()
scaled_features_no_outliers = scaler.fit_transform(df_data_no_outliers)

# 4. 選擇使用的特徵
# 根據相關性和特徵重要性選擇特徵，這裡假設選擇了所有特徵
selected_features_no_outliers = df_data_no_outliers.loc[:, ['總騎乘長度','最高海拔','平均海拔','爬升坡率平均(趨勢)','最大爬升坡率(趨勢)','爬升坡率平均(Per)','最大爬升坡率(Per)','下降坡率平均(趨勢)','最大下降坡率(趨勢)','下降坡率平均(Per)','最大下降坡率(Per)','總爬升海拔','最大爬升海拔','平均爬升海拔','總下降海拔','最大下降海拔','平均下降海拔','爬升路段比例','下降路段比例','平均路徑變化率']]

# 正規化或標準化選擇的特徵
scaled_selected_features_no_outliers = scaler.fit_transform(selected_features_no_outliers)

# 5. 選擇要分幾類
k = 9  # 假設選擇分3類

# 6. 進行聚類並添加聚類標籤到 DataFrame
kmeans = KMeans(n_clusters=k)
clusters_no_outliers = kmeans.fit_predict(scaled_selected_features_no_outliers)
df_data_no_outliers['Cluster'] = clusters_no_outliers

# 7. 將結果做PCA降為可視化
pca = PCA(n_components=2)
principal_components_no_outliers = pca.fit_transform(scaled_selected_features_no_outliers)
df_pca_no_outliers = pd.DataFrame(data=principal_components_no_outliers, columns=['PCA1', 'PCA2'])
df_pca_no_outliers['Cluster'] = clusters_no_outliers

# 可視化
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df_pca_no_outliers, palette='Set1')
plt.title('K-means Clustering with PCA (No Outliers)')
plt.show()

# 8. 將三個群集的內容，分別存到三個list中
cluster_0_no_outliers = df_data_no_outliers[df_data_no_outliers['Cluster'] == 0]
cluster_1_no_outliers = df_data_no_outliers[df_data_no_outliers['Cluster'] == 1]
cluster_2_no_outliers = df_data_no_outliers[df_data_no_outliers['Cluster'] == 2]

# 可以進一步將這些群集轉換為列表
list_cluster_0_no_outliers = cluster_0_no_outliers.values.tolist()
list_cluster_1_no_outliers = cluster_1_no_outliers.values.tolist()
list_cluster_2_no_outliers = cluster_2_no_outliers.values.tolist()

# 可以打印結果或保存到文件中
print("Cluster 0:", list_cluster_0_no_outliers)
print("Cluster 1:", list_cluster_1_no_outliers)
print("Cluster 2:", list_cluster_2_no_outliers)
