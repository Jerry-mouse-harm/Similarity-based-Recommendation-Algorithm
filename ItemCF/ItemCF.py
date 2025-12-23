import pandas as pd
from typing import Dict, Any
import psycopg2  # openGauss is PG-compatible

db_config = {
     "host": "localhost",
     "port": "",
     "user": "",
     "password": "",
     "dbname": ""
 }

"""物品协同过滤(ItemCF)算法"""

class ItemCF(object):
    """基于相似性的推荐算法"""

    def __init__(self, lamda=50):
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

    def loading_database(self, db_config: Dict[str, Any], query: str = None) -> pd.DataFrame:
        """
        从OpenGauss数据库中加载数据
        Args:
            db_config: 包含 host, port, user, password, dbname 的字典
        """
        if query is None:
            query = """
            SELECT user_id, document_id, rating
            FROM user_interactions
            ORDER BY user_id, document_id;
            """

        try:
            with psycopg2.connect(**db_config) as conn:
                df = pd.read_sql_query(query, conn)
            self.interactions_df = df.copy()
            print(f"加载 {len(self.interactions_df)} 条交互记录")
            return self.interactions_df
        except Exception as e:
            print(f"数据库连接失败: {e}")
            raise

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
            user_id = row["user_id"]
            item_id = row["document_id"]
            rating = row["rating"]

            if user_id not in self.user_item:
                self.user_item[user_id] = {}
            self.user_item[user_id][item_id] = rating

            if item_id not in self.item_user:
                self.item_user[item_id] = set()
            self.item_user[item_id].add(user_id)

        self.item = set(self.item_user.keys())
        self.user = set(self.user_item.keys())

        print("数据加载完毕")
        print(f"  物品总数: {len(self.item)}")
        print(f"  用户总数: {len(self.user)}")

    def get_similarity(self, item1, item2) -> float:
        """计算两个事物的相似性"""
        item1_visit_users = self.item_user[item1]
        item2_visit_users = self.item_user[item2]

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
        similarity_dict: Dict[Any, Dict[Any, float]] = {item: {} for item in mylist}

        for i, item_i in enumerate(mylist):
            for item_j in mylist[i + 1 :]:
                sim = self.get_similarity(item_i, item_j)
                if sim > 0:
                    similarity_dict[item_i][item_j] = sim
                    similarity_dict[item_j][item_i] = sim

        for item in mylist:
            if len(similarity_dict[item]) > 10:
                top_items = sorted(
                    similarity_dict[item].items(), key=lambda x: x[1], reverse=True
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
            score = self.predict_rating(user, item_id, k=k)
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
        similar_target_items = self.similarity_matrix.get(document_id, {})

        relevant_items = []
        for item, score in similar_target_items.items():
            if item in used_items:
                relevant_items.append((item, score))

        if not relevant_items:
            return 0.0

        relevant_items.sort(key=lambda x: x[1], reverse=True)
        top_k_item = relevant_items[:k]

        numerator = 0.0
        denominator = 0.0
        for item, _ in top_k_item:
            sim = self.similarity_matrix[document_id][item]
            numerator += sim * used_items[item]
            denominator += abs(sim)

        if denominator == 0:
            return 0.0
        return numerator / denominator


if __name__ == "__main__":
    print("=" * 60)
    print("ItemCF推荐系统 - Jaccard相似度+惩罚因子")
    print("=" * 60)

    # 示例：连接OpenGauss数据库


    recommender = ItemCF()
    recommender.loading_database(db_config)
    recommender.build_interact_dict()
    recommender.get_similarity_matrix()
    print(recommender.item_user)
    print(recommender.recommend_for_user(1,6))

