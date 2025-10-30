from pymongo import MongoClient
import numpy as np
import streamlit as st


# 自动MongoDB连接
def auto_connect_mongodb(mongodb_config):
    """
    初始化 MongoDB 连接，返回三元组：(连接成功, 错误消息, client对象)
    外部调用无需直接写入 session_state，由主入口统一赋值可以防止覆盖。
    """
    from pymongo import MongoClient

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

