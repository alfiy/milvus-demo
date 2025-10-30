from pymongo import MongoClient
import numpy as np
import streamlit as st


# 自动MongoDB连接
def auto_connect_mongodb():
    """
    初始化 MongoDB 连接，仅设置 session_state，不做 st.xxx 调用。
    调用后可通过 st.session_state['mongodb_connected'] 和 st.session_state['mongodb_connect_error'] 判断连接状态。
    """
 
    # 直接从已初始化的 session_state 读取配置
    mongodb_config = st.session_state.get('mongodb_config', {})
    if not mongodb_config or not mongodb_config.get("host"):
        st.session_state['mongodb_connected'] = False
        st.session_state['mongodb_connect_error'] = "缺少 MongoDB 配置"
        st.session_state['mongodb_client'] = None
        return False

    # 已有连接直接复用
    if st.session_state.get('mongodb_connected') and st.session_state.get('mongodb_client') is not None:
        return True

    try:
        # 构建MongoDB URI
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

        # 写入连接状态到 session
        st.session_state['mongodb_connected'] = True
        st.session_state['mongodb_connect_error'] = None
        st.session_state['mongodb_client'] = client
        return True

    except Exception as e:
        st.session_state['mongodb_connected'] = False
        st.session_state['mongodb_connect_error'] = str(e)
        st.session_state['mongodb_client'] = None
        return False

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

