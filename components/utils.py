from pymongo import MongoClient
import numpy as np
import streamlit as st
import logging
from typing import List, Dict, Any


def vector_search(
    query: str,
    top_k: int,
    milvus_collection,
    mongo_col,
    vector_processor,
    filter_mode: str = "similarity",    # "similarity" 或 "distance"
    filter_threshold: float = 0.0,
    output_fields: List[str] = ["text", "metadata"]
) -> List[Dict[str, Any]]:
    """
    优化后的向量+Mongo 搜索功能，支持过滤/统计和异常处理
    """
    try:
        # 向量化查询
        query_vector = vector_processor.encode([query])[0]

        # Milvus 搜索
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }
        results = milvus_collection.search(
            data=[query_vector.tolist()],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            output_fields=output_fields,
        )

        # 整理搜索结果：批量Mongo查元数据
        ids = [hit.id for hit in results[0]]
        scores = [float(hit.distance) for hit in results[0]]

        docs = list(mongo_col.find({"_id": {"$in": ids}}, {field:1 for field in output_fields}))
        id2doc = {str(doc["_id"]): doc for doc in docs}

        combined = []
        for id_, score in zip(ids, scores):
            doc = id2doc.get(str(id_), {})
            combined.append({
                "id": id_,
                "score": score,
                "text": doc.get("text", ""),
                "metadata": doc.get("metadata", {}),
            })

        # 相似度/距离过滤
        if filter_mode == "similarity":
            filtered = [r for r in combined if r["score"] >= filter_threshold]
        else:
            filtered = [r for r in combined if r["score"] <= filter_threshold]

        return filtered

    except Exception as e:
        logging.error(f"向量检索失败: {e}")
        return []


# 自动MongoDB连接
def auto_connect_mongodb(mongodb_config):
    """
    初始化 MongoDB 连接，返回三元组：(连接成功, 错误消息, client对象)
    外部调用无需直接写入 session_state，由主入口统一赋值可以防止覆盖。
    """
    

    if not mongodb_config or not mongodb_config.get("host"):
        return False, "缺少 MongoDB 配置", None

    try:
        username = mongodb_config.get("username", "")
        password = mongodb_config.get("password", "")
        host = mongodb_config.get("host", "localhost")
        port = mongodb_config.get("port", 27017)
        db_name = mongodb_config.get("db_name", "textdb")
        col_name = mongodb_config.get("col_name", "metadata")

        if username and password:
            uri = f"mongodb://{username}:{password}@{host}:{port}/{db_name}?authSource=admin"
        else:
            uri = f"mongodb://{host}:{port}/"

        client = MongoClient(uri, serverSelectionTimeoutMS=3000)
        db = client[db_name]
        col = db[col_name]
        _ = col.estimated_document_count()
        return True, None, client
    except Exception as e:
        return False, str(e), None

# mongoDB状态
def get_mongodb_stats(mongodb_client, mongodb_config):
    """
    统计MongoDB主业务集合的状态与数据量。
    - 输入: mongodb_client (pymongo.MongoClient对象), mongodb_config（dict配置）
    - 返回: dict {connected: bool, error: str or None, count: int, sample_texts: list, vector_info: str, vector_size: float}
    """
    stats = {
        "connected": False,
        "error": None,
        "count": 0,
        "sample_texts": [],
        "vector_info": "N/A",
        "vector_size": 0.0
    }

    if not mongodb_client or not mongodb_config:
        stats["error"] = "缺少连接对象或配置"
        return stats

    try:
        db = mongodb_client[mongodb_config.get("db_name", "textdb")]
        col = db[mongodb_config.get("col_name", "metadata")]
        stats["count"] = col.count_documents({})
        stats["connected"] = True

        sample_docs = list(col.find({}, {"text": 1}).limit(10))
        stats["sample_texts"] = [doc.get("text", "") for doc in sample_docs]

        # 检查向量字段
        vector_sample = col.find_one({"vector": {"$exists": True}}, {"vector": 1})
        if vector_sample and vector_sample.get("vector") is not None:
            import numpy as np
            sample_vector = np.array(vector_sample["vector"])
            stats["vector_info"] = sample_vector.shape[0] if sample_vector.ndim > 0 else "N/A"
            stats["vector_size"] = sample_vector.nbytes / 1024 / 1024
    except Exception as e:
        stats["error"] = str(e)

    return stats


def get_mongodb_data(mongodb_config):
    """从MongoDB读取文本与向量数据，并统计"""
    host = mongodb_config.get('host', 'localhost')
    port = mongodb_config.get('port', 27017)
    db_name = mongodb_config.get('db_name', '')
    col_name = mongodb_config.get('col_name', 'texts')
    user = mongodb_config.get('username')
    pwd = mongodb_config.get('password')
    auth = f"{user}:{pwd}@" if user and pwd else ""
    mongo_uri = f"mongodb://{auth}{host}:{port}"
    results = {
        "connected": False,
        "count": 0,
        "texts": [],
        "vectors": None,
        "error": None
    }
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=3000)
        db = client[db_name]
        col = db[col_name]
        docs = list(col.find({}, {"_id": 0}))
        texts = [doc.get("text", "") for doc in docs]
        vectors = np.array([doc["vector"] for doc in docs if "vector" in doc])
        doc_count = col.estimated_document_count()
        data_loaded = len(texts) > 0
        results.update({
            "connected": True,
            "count": doc_count,
            "texts": texts,
            "vectors": vectors if data_loaded else None,
        })
    except Exception as e:
        results["error"] = str(e)
    return results

