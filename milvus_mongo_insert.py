# -*- coding: utf-8 -*-
"""
Milvus + MongoDB 批量插入脚本
与现有 Streamlit 数据上传与处理逻辑对接，
完成向量数据插入 Milvus，元数据插入 MongoDB。
"""

from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType,utility
from pymongo import MongoClient
import numpy as np


# ----- Milvus 初始化 -----

def get_milvus_collection(collection_name="text_embeddings", dim=384):
    """
    获取或创建 Milvus collection
    不重复 connect，避免连接冲突
    """
    # 判断是否已连接
    if not connections.has_connection("default"):
        connections.connect(host='localhost', port='19530')
    # 检查 collection 是否已存在
    if not utility.has_collection(collection_name):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        ]
        schema = CollectionSchema(fields, "文本向量集合")
        collection = Collection(collection_name, schema)
    else:
        collection = Collection(collection_name)
    return collection

# ----- MongoDB 初始化 -----
def get_mongo_collection():
    """
    推荐：从全局 session_state 自动复用连接对象和配置信息，获取集合对象。
    """
    import streamlit as st
    config = st.session_state.get("mongodb_config", None)
    client = st.session_state.get("mongodb_client", None)
    if config and client:
        db = client[config["db_name"]]
        col = db[config["col_name"]]
        return col
    else:
        raise Exception("MongoDB未配置或未连接，请先在配置页面连接MongoDB。")

# ----- 批量插入逻辑 -----

def insert_batch(texts, vectors, metadata_list, collection, mongo_col, batch_size=1000):
    """
    批量插入 texts, vectors, metadata 到 Milvus + MongoDB
    texts: List[str]
    vectors: np.ndarray (N, dim)
    metadata_list: List[dict]
    """
    assert len(texts) == vectors.shape[0] == len(metadata_list), "数量不一致"
    N = len(texts)
    all_milvus_ids = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        mini_vectors = vectors[start:end].tolist()
        # Milvus 插入
        res = collection.insert([mini_vectors])
        batch_ids = res.primary_keys
        all_milvus_ids.extend(batch_ids)
        # MongoDB 插入
        docs = [
            {
                "_id": milvus_id,
                "text": text,
                "metadata": meta
            }
            for milvus_id, text, meta in zip(batch_ids, texts[start:end], metadata_list[start:end])
        ]
        mongo_col.insert_many(docs)
    return all_milvus_ids

# ----- 用于 Streamlit 集成的主入口 -----

def milvus_mongo_upload(texts, vectors, metadata_list, milvus_dim=384, collection_name="text_embeddings"):
    """
    集成上传入口，Streamlit 内直接调用
    texts, vectors, metadata_list 来自你 session_state
    """
    collection = get_milvus_collection(collection_name=collection_name, dim=milvus_dim)
    mongo_col = get_mongo_collection()
    milvus_ids = insert_batch(texts, vectors, metadata_list, collection, mongo_col)
    return milvus_ids

# ----- 示例如何在 Streamlit 业务代码中调用 -----
# texts = st.session_state.texts
# vectors = st.session_state.vectors
# metadata = st.session_state.metadata
# embedding_dim = vectors.shape[1]
# milvus_mongo_upload(texts, vectors, metadata, milvus_dim=embedding_dim)