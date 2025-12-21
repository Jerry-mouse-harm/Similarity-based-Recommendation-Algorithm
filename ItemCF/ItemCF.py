import numpy as np
import pandas as pd
import sqlite3
from typing import Dict, List, Tuple, Set, Any
from sklearn.metrics import jaccard_score

# 可以使用sklearn中的jaccard_score直接结算，我们在这里选择手动实现
"""物品协同过滤(ItemCF)算法"""

class ItemCF(object):
    """基于相似性的推荐算法"""

    def __init__(self, lamda = 50):
        # 添加惩罚因子lamba
        self.lamda = lamda
        self.interactions_df = None
        self.item_user = {}
        """不同文档不同用户的访问次数"""
        self.user_item = {}
        """用户对不同的文档的评分"""
        self.similarity_matrix = {}
        """任意两个事物的相似性"""
        self.user = set()
        self.item = set()

    def loading_database(self, db_path: str, query: str = None) -> pd.DataFrame:
        """从数据库中加载数据"""
        这里没有实现，等待数据库的建立完成

    def loading_frame(self, df: pd.DataFrame):
         """
         从DataFrame加载数据
         表格数据的col为user_id、docx_id、rating

         Args:
             df: 包含 user_id, document_id, rating 列的DataFrame
         """

         self.interactions_df = df.copy()
         print(f"加载 {len(self.interactions_df)} 条交互记录")

    def build_interact_dict(self):
        """
        构建字典
        - item_users: 每个物品被哪些用户交互过
        - user_items: 每个用户交互过哪些物品及评分
        """
        if self.interactions_df is None:
            print("数据未加载！")
            raise Exception("访问未加载的空数据！")



        for _, row in self.interactions_df.iterrows():
            """加载表格数据"""
            user_id = row['user_id']
            item_id = row['document_id']
            rating = row['rating']

            if user_id not in self.user_item:
                # 在遇到新用户时，初始化{}容器
                # 因为对文档的评分需要二维数据来表示
                self.user_item[user_id] = {}
            self.user_item[user_id][item_id] = rating

            if item_id not in self.item_user:
                # 遇到新的文档，初始化set容器，不断记录用户的访问次数
                self.item_user[item_id] = set()
            self.item_user[item_id].add(user_id)

        self.item = set(self.item_user.keys())
        self.user = set(self.user_item.keys())

        print("数据加载完毕")
        print(f"  物品总数: {len(self.item)}")
        print(f"  用户总数: {len(self.user)}")

    def get_similarity(self, item1, item2) -> float:
        """计算两个事物的相似性"""

        item1_visit_users = self.user_item[item1]
        item2_visit_users = self.user_item[item2]
        """计算交、并集合"""

        jiao_ji = item1_visit_users & item2_visit_users
        bing_ji = item1_visit_users | item2_visit_users

        if bing_ji:
            jaccard_score = len(jiao_ji) / len(bing_ji)
            punishment = len(jiao_ji) / (len(jiao_ji) + self.lamda)

            return jaccard_score * punishment
        else:
            return 0.0

    def get_similarity_matrix(self):
        """创建相似性矩阵"""
        print(f"使用策略: Jaccard相似度 + 惩罚因子(λ={self.lamda})")
        mylist = list(self.item)
        num = len(mylist)
        similarity_dict = {}
        # 两两遍历所有事物，得到相似性矩阵
        # 我们用字典存储，键值为（两个不同事物的item_id）,值为相似性

        for i, item_i in enumerate(mylist):
            for j, item_j in enumerate(mylist[i+1:]):
                sim = self.get_similarity(item_i, item_j)

                if sim > 0:
                    if (item_i, item_j) not in similarity_dict:
                        similarity_dict[item_i, item_j] = sim
                    if (item_j, item_i) not in similarity_dict:
                        similarity_dict[item_j, item_i] = sim

        for item in mylist:
            if len(similarity_dict[item]) > 10:
                top_items = sorted(
                    similarity_dict[item].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
                similarity_dict[item] = dict(top_items)

        self.similarity_matrix = similarity_dict
        print("相似矩阵构建完毕！！！")

    def recommend_for_user(self, user: int, k: int) -> list[tuple[int, float]]:
        """
        为用户推荐未浏览的文档
        K：根据前k个相似文件来进行分数的预测

        最后生成Top10推荐文档，从高到低排序，如果不够10个取前最大数个
        """
        if user not in self.user_item:
            print("用户不在数据库中！")
            return []

        used_items = set(self.user_item[user].keys())
        candidate_items = self.item - used_items

        if not candidate_items:
            print("用户游览了所有文档")
            return []

        predict_scores = []
        for item_id in candidate_items:
            score = self.predict_rating(user, item_id, k = k)
            if score > 0:
                predict_scores.append((item_id, score))

        predict_scores.sort(key=lambda x: x[1], reverse=True)

        if len(predict_scores) > 10:
            return predict_scores[:10]
        else:
            return predict_scores

    def predict_rating(self, user_id: int, document_id: int, k=10) -> float:
        """
        预测用户对未浏览文档的兴趣评分

        预测公式:
        r̂(u,i) = Σ[sim(i,j) × r(u,j)] / Σ|sim(i,j)|

        Args:
            user_id: 用户ID
            document_id: 文档ID
            k: 使用前k个最相似的物品进行预测

        Returns:
            预测评分
        """
        if user_id not in self.user_item:
            print("该用户不在数据库之中")
            return 0.0

        if document_id not in self.item:
            return 0.0

        used_items = self.user_item[user_id]
        similar_target_items = self.similarity_matrix[document_id]

        # 筛选出用户已经交互且与目标文档相似的物品
        relevant_items = []

        for item, score in similar_target_items.items():
            if item in used_items:
                relevant_items.append((item, score))

        if not relevant_items:
            return 0.0

        relevant_items.sort(key=lambda x: x[1], reverse=True)
        top_k_item = relevant_items[:k]

        final_score = 0.0;
        for item, score in top_k_item:
            final_score += self.similarity_matrix[document_id][item]
        denominator = 0.0;
        for item, score in top_k_item:
            denominator += self.similarity_matrix[document_id][item]

        if denominator == 0:
            return 0.0
        return final_score / denominator

def create_sample_database(db_path='sample_interactions.db'):
    """创建示例数据库"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 创建表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            document_id INTEGER NOT NULL,
            rating FLOAT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 插入示例数据
    sample_data = [
        (1, 101, 5.0),
        (1, 102, 4.0),
        (1, 103, 3.0),
        (2, 101, 4.0),
        (2, 104, 5.0),
        (2, 105, 4.0),
        (3, 102, 5.0),
        (3, 103, 4.0),
        (3, 105, 3.0),
        (4, 101, 3.0),
        (4, 103, 4.0),
        (4, 106, 5.0),
        (5, 102, 4.0),
        (5, 104, 5.0),
        (5, 105, 4.0),
        (5, 106, 3.0),
        (6, 101, 5.0),
        (6, 107, 4.0),
        (7, 102, 3.0),
        (7, 106, 5.0),
        (8, 103, 4.0),
        (8, 107, 5.0),
    ]

    cursor.executemany(
        'INSERT INTO user_interactions (user_id, document_id, rating) VALUES (?, ?, ?)',
        sample_data
    )

    conn.commit()
    conn.close()
    print(f"✓ 示例数据库已创建: {db_path}")


if __name__ == "__main__":
    print("="*60)
    print("ItemCF推荐系统 - Jaccard相似度+惩罚因子")
    print("="*60)

    # 创建示例数据库
    db_path = 'sample_interactions.db'
    create_sample_database(db_path)

    # 创建推荐器实例
    recommender = ItemCF()  # 小数据集用较小的λ

    # 从数据库加载数据
    print("\n1. 从数据库加载数据")
    df = recommender.loading_database(db_path)
    print("\n数据预览:")
    print(df.head(10))

    # 构建交互数据结构
    print("\n2. 构建交互数据结构")
    recommender.build_interact_dict()

    # 计算物品相似度
    print("\n3. 计算物品相似度矩阵")
    recommender.get_similarity_matrix()
    """
    # 查看物品相似度示例
    print("\n文档101的相似文档:")
    similar_items = recommender.get_similar_items(101, n=5)
    for item_id, sim in similar_items:
        print(f"  文档 {item_id}: 相似度 {sim:.4f}")
    """
    # 预测单个评分
    print("\n5. 预测评分示例")
    user_id = 1
    doc_id = 104
    predicted = recommender.predict_rating(user_id, doc_id, k=3)
    print(f"用户 {user_id} 对文档 {doc_id} 的预测评分: {predicted:.4f}")

    # 为用户生成推荐
    print("\n6. 生成推荐列表")
    recommendations = recommender.recommend_for_user(user=1, k=3)
    print(f"\n为用户 {user_id} 推荐的文档:")
    for doc_id, score in recommendations:
        print(f"  文档 {doc_id}: 预测评分 {score:.4f}")


    print("\n")
    print("演示完成！")
