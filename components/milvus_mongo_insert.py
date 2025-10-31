# -*- coding: utf-8 -*-
"""
Milvus + MongoDB 批量插入脚本 
与现有 Streamlit 数据上传与处理逻辑对接，
完成向量数据插入 Milvus，元数据插入 MongoDB。
"""

from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import numpy as np
import streamlit as st
import json
import time


# ----- Milvus 初始化 -----

def get_milvus_collection(collection_name="text_vectors", dim=384):
    """
    获取或创建 Milvus collection
    使用现有的 MilvusManager 连接，避免连接冲突
    """
    try:
        # 优先使用 MilvusManager 的连接
        milvus_manager = st.session_state.components.get('milvus_manager')
        if milvus_manager and milvus_manager.is_connected:
            # 如果已有集合且名称匹配，直接返回
            if milvus_manager.collection and milvus_manager.collection_name == collection_name:
                return _ensure_collection_ready(milvus_manager.collection)
            
            # 创建新集合
            return _create_or_get_collection(collection_name, dim)
        
        # 备用方案：直接连接
        if not connections.has_connection("default"):
            connections.connect(host='localhost', port='19530')
        
        return _create_or_get_collection(collection_name, dim)
        
    except Exception as e:
        st.error(f"❌ Milvus集合初始化失败: {e}")
        raise


def _create_or_get_collection(collection_name, dim):
    """创建或获取集合的内部函数"""
    try:
        if not utility.has_collection(collection_name):
            # 创建新集合
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
                FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535)
            ]
            schema = CollectionSchema(fields, "文本向量集合")
            collection = Collection(collection_name, schema)
            
            # 等待集合创建完成
            time.sleep(0.5)
            
            # 创建索引
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            collection.create_index("vector", index_params)
            
            # 加载集合
            collection.load()
            st.success(f"✅ 成功创建Milvus集合: {collection_name}")
            
        else:
            # 获取现有集合
            collection = Collection(collection_name)
            
            # 检查并创建索引（如果不存在）
            _ensure_index_exists(collection)
            
            # 确保集合已加载
            _ensure_collection_loaded(collection)
            
        return collection
        
    except Exception as e:
        st.error(f"❌ 创建或获取集合失败: {e}")
        raise


def _ensure_index_exists(collection):
    """确保索引存在"""
    try:
        indexes = collection.indexes
        vector_index_exists = any(idx.field_name == "vector" for idx in indexes)
        
        if not vector_index_exists:
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            collection.create_index("vector", index_params)
            st.info("📊 已创建向量索引")
            
    except Exception as e:
        # 如果索引已存在或其他非关键错误，记录警告但不中断流程
        st.warning(f"⚠️ 索引操作警告: {e}")


def _ensure_collection_loaded(collection):
    """确保集合已加载"""
    try:
        # 检查集合是否已加载
        if hasattr(collection, '_is_loaded') and not collection._is_loaded:
            collection.load()
        else:
            # 尝试加载，如果已加载会抛异常但不影响使用
            try:
                collection.load()
            except Exception:
                pass  # 忽略已加载的异常
                
    except Exception as e:
        st.warning(f"⚠️ 集合加载警告: {e}")


def _ensure_collection_ready(collection):
    """确保集合准备就绪"""
    try:
        _ensure_index_exists(collection)
        _ensure_collection_loaded(collection)
        return collection
    except Exception as e:
        st.error(f"❌ 集合准备失败: {e}")
        raise

def get_mongo_collection():
    """
    从全局 session_state 自动复用连接对象和配置信息，获取集合对象。

    Returns:
        Collection: MongoDB 集合对象

    Raises:
        Exception: 当 MongoDB 未连接或连接失效时抛出异常
    """
    # 🔧 第一步：检查连接状态
    if not st.session_state.get("mongodb_connected", False):
        raise Exception("MongoDB未配置或未连接，请先在 '🍃 MongoDB配置管理' 页面连接MongoDB。")

    # 🔧 第二步：获取配置和客户端（添加空值检查）
    config = st.session_state.get("mongodb_config")
    client = st.session_state.get("mongodb_client")

    if not config:
        st.session_state['mongodb_connected'] = False
        raise Exception("MongoDB配置信息缺失")

    if not client:
        st.session_state['mongodb_connected'] = False
        raise Exception("MongoDB客户端未初始化")

    # 🔧 第三步：获取集合并测试连接
    try:
        db_name = config.get("db_name", "textdb")
        col_name = config.get("col_name", "metadata")

        db = client[db_name]
        col = db[col_name]

        # 测试连接是否仍然有效
        _ = col.estimated_document_count()

        return col

    except Exception as e:
        # 连接失效，更新状态
        st.session_state['mongodb_connected'] = False
        st.session_state['mongodb_connect_error'] = str(e)
        raise Exception(f"MongoDB连接已断开: {e}")

# ----- 批量插入逻辑 -----
def insert_batch(texts, vectors, metadata_list, collection, mongo_col, batch_size=500):
    """
    批量插入 texts, vectors, metadata 到 Milvus + MongoDB
    自动处理向量shape与类型，适配所有标准Milvus schema。
    """
    assert len(texts) == vectors.shape[0] == len(metadata_list), (
        f"数量不一致: texts={len(texts)}, vectors={vectors.shape[0]}, metadata={len(metadata_list)}"
    )
    N = len(texts)
    all_milvus_ids = []
    st.info(f"🚀 开始批量插入 {N:,} 条数据到 Milvus 和 MongoDB")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_texts = texts[start:end]
            batch_vectors = np.asarray(vectors[start:end])
            batch_metadata = metadata_list[start:end]
            
            # 保险：保证向量shape为二维(batch, dim)
            if batch_vectors.ndim != 2:
                st.error(f"❌ 向量shape不合法，必须二维(batch, dim)，实际: {batch_vectors.shape}")
                raise ValueError(f"向量shape不合法: {batch_vectors.shape}")
            
            # 转为浮点类型列表
            vectors_clean = batch_vectors.astype(np.float32).tolist()
            
            # 🔧 关键修复：构建每条记录的字典列表
            milvus_data = [
                {
                    "vector": vec,
                    "text": text,
                    "metadata": json.dumps(meta, ensure_ascii=False)
                }
                for vec, text, meta in zip(vectors_clean, batch_texts, batch_metadata)
            ]
            
            status_text.text(f"📝 插入批次 {start//batch_size + 1}/{(N-1)//batch_size + 1} 到 Milvus...")
            
            try:
                res = collection.insert(milvus_data)
                batch_ids = res.primary_keys if hasattr(res, "primary_keys") else res
                all_milvus_ids.extend(batch_ids)
                collection.flush()
            except Exception as e:
                st.error(f"❌ Milvus插入失败: {e}")
                st.error(f"vector shape={batch_vectors.shape}, text count={len(batch_texts)}, metadata count={len(batch_metadata)}")
                # print("❌ DEBUG - milvus_data[0]:", milvus_data[0] if milvus_data else None)
                # print("❌ DEBUG - vector[0] type:", type(milvus_data[0]['vector']) if milvus_data else None)
                # print("❌ DEBUG - vector[0][0] type:", type(milvus_data[0]['vector'][0]) if milvus_data and milvus_data[0]['vector'] else None)
                raise
            
            # 插入 MongoDB
            status_text.text(f"💾 插入批次 {start//batch_size + 1}/{(N-1)//batch_size + 1} 到 MongoDB...")
            try:
                docs = [
                    {
                        "_id": int(milvus_id),
                        "text": text,
                        "metadata": meta
                    }
                    for milvus_id, text, meta in zip(batch_ids, batch_texts, batch_metadata)
                ]
                mongo_col.insert_many(docs)
            except Exception as e:
                st.error(f"❌ MongoDB插入失败: {e}")
                st.warning("⚠️ 建议检查 MongoDB 连接或重复主键")
                raise
            
            progress = end / N
            progress_bar.progress(progress)
            status_text.text(f"✅ 已完成 {end:,}/{N:,} 条记录 ({progress*100:.1f}%)")
        
        _ensure_collection_loaded(collection)
        progress_bar.progress(1.0)
        status_text.text(f"🎉 批量插入完成，共 {len(all_milvus_ids):,} 条记录")
        return all_milvus_ids
        
    except Exception as e:
        st.error(f"❌ 批量插入过程中出错: {e}")
        raise
    finally:
        progress_bar.empty()
        status_text.empty()

# def insert_batch(texts, vectors, metadata_list, collection, mongo_col, batch_size=500):
#     """
#     批量插入 texts, vectors, metadata 到 Milvus + MongoDB
#     """
#     assert len(texts) == vectors.shape[0] == len(metadata_list), (
#         f"数量不一致: texts={len(texts)}, vectors={vectors.shape[0]}, metadata={len(metadata_list)}"
#     )

#     N = len(texts)
#     all_milvus_ids = []

#     st.info(f"🚀 开始批量插入 {N:,} 条数据到 Milvus 和 MongoDB")
#     progress_bar = st.progress(0)
#     status_text = st.empty()

#     try:
#         for start in range(0, N, batch_size):
#             end = min(start + batch_size, N)
#             batch_texts = texts[start:end]
#             batch_vectors = vectors[start:end].astype(np.float32)
#             batch_metadata = metadata_list[start:end]

#             # np.ndarray保证shape=(N, dim)
#             vectors_list = batch_vectors.astype(np.float32).tolist()
#             vectors_clean = [
#                 [float(x) for x in vec] if isinstance(vec, (list, np.ndarray)) else []
#                 for vec in vectors_list
#             ]

#             # 🔧 修复数据格式：Milvus 要求 list 格式，顺序与 schema 对齐
#             milvus_data = {
#                 "vector": vectors_clean,  # list of vectors
#                 "text": batch_texts,               # list of string
#                 "metadata": [json.dumps(meta, ensure_ascii=False) for meta in batch_metadata]  # list of JSON string
#             }

#             status_text.text(f"📥 插入批次 {start//batch_size + 1}/{(N-1)//batch_size + 1} 到 Milvus...")

#             try:
#                 res = collection.insert(milvus_data)
#                 batch_ids = res.primary_keys
#                 all_milvus_ids.extend(batch_ids)
#                 collection.flush()
#             except Exception as e:
#                 st.error(f"❌ Milvus插入失败: {e}")
#                 st.error(f"vector shape={batch_vectors.shape}, text count={len(batch_texts)}, metadata count={len(batch_metadata)}")
#                 raise

#             # 插入 MongoDB
#             status_text.text(f"💾 插入批次 {start//batch_size + 1}/{(N-1)//batch_size + 1} 到 MongoDB...")

#             try:
#                 docs = [
#                     {
#                         "_id": int(milvus_id),
#                         "text": text,
#                         "metadata": meta
#                     }
#                     for milvus_id, text, meta in zip(batch_ids, batch_texts, batch_metadata)
#                 ]
#                 mongo_col.insert_many(docs)
#             except Exception as e:
#                 st.error(f"❌ MongoDB插入失败: {e}")
#                 st.warning("⚠️ 建议检查 MongoDB 连接或重复主键")
#                 raise

#             progress = end / N
#             progress_bar.progress(progress)
#             status_text.text(f"✅ 已完成 {end:,}/{N:,} 条记录 ({progress*100:.1f}%)")

#         _ensure_collection_loaded(collection)
#         progress_bar.progress(1.0)
#         status_text.text(f"🎉 批量插入完成，共 {len(all_milvus_ids):,} 条记录")

#         return all_milvus_ids

#     except Exception as e:
#         st.error(f"❌ 批量插入过程中出错: {e}")
#         raise
#     finally:
#         progress_bar.empty()
#         status_text.empty()

# ----- 用于 Streamlit 集成的主入口 -----
def milvus_mongo_upload(texts, vectors, metadata_list, milvus_dim=384, collection_name="text_vectors"):
    """
    集成上传入口，Streamlit 内直接调用
    texts: List[str] - 文本列表
    vectors: np.ndarray - 向量数组 (N, dim)
    metadata_list: List[dict] - 元数据列表
    milvus_dim: int - 向量维度
    collection_name: str - 集合名称
    """
    try:
        # ===== 强力清洗文本字段 =====
        texts_flat = []
        for t in texts:
            # 多层嵌套list彻底剥平
            if isinstance(t, list):
                for sub_t in t:
                    if isinstance(sub_t, str):
                        texts_flat.append(sub_t)
            elif isinstance(t, str):
                texts_flat.append(t)
        texts = texts_flat

        # 其它输入验证不变
        if not texts or len(texts) == 0:
            raise ValueError("文本列表不能为空")
        if vectors is None or vectors.size == 0:
            raise ValueError("向量数组不能为空")
        if not metadata_list or len(metadata_list) == 0:
            raise ValueError("元数据列表不能为空")
        if len(texts) != vectors.shape[0] or len(texts) != len(metadata_list):
            raise ValueError(f"数据长度不一致: texts={len(texts)}, vectors={vectors.shape[0]}, metadata={len(metadata_list)}")
        if vectors.shape[1] != milvus_dim:
            raise ValueError(f"向量维度不匹配: 期望 {milvus_dim}, 实际 {vectors.shape[1]}")
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors, dtype=np.float32)
        else:
            vectors = vectors.astype(np.float32)

        st.info(f" 准备上传数据: {len(texts):,} 条文本，向量维度: {milvus_dim}")

        collection = get_milvus_collection(collection_name=collection_name, dim=milvus_dim)
        mongo_col = get_mongo_collection()

        print("DEBUG insert_batch输入类型:", type(texts), type(vectors), type(metadata_list))
        print("DEBUG texts示例:", texts[:5])
        print("DEBUG vectors shape:", vectors.shape)


        milvus_ids = insert_batch(texts, vectors, metadata_list, collection, mongo_col)

        st.success(f" 数据上传成功！")
        st.success(f" Milvus: 插入 {len(milvus_ids):,} 条向量记录")
        st.success(f" MongoDB: 插入 {len(milvus_ids):,} 条元数据记录")

        return milvus_ids

    except Exception as e:
        st.error(f"❌ 数据上传失败: {e}")
        st.info(" 请检查以下项目:")
        st.info("1. Milvus 数据库是否正常连接")
        st.info("2. MongoDB 数据库是否正常连接")  
        st.info("3. 数据格式是否正确")
        st.info("4. 向量维度是否匹配")
        raise


# ----- 数据验证和统计函数 -----

def verify_upload_success(collection_name="text_vectors"):
    """
    验证上传是否成功，返回统计信息
    """
    try:
        # 验证 Milvus
        if utility.has_collection(collection_name):
            collection = Collection(collection_name)
            _ensure_collection_loaded(collection)
            milvus_count = collection.num_entities
        else:
            milvus_count = 0
        
        # 验证 MongoDB
        try:
            mongo_col = get_mongo_collection()
            mongodb_count = mongo_col.estimated_document_count()
        except:
            mongodb_count = 0
        
        return {
            "milvus_count": milvus_count,
            "mongodb_count": mongodb_count,
            "success": milvus_count > 0 and mongodb_count > 0,
            "synchronized": milvus_count == mongodb_count
        }
        
    except Exception as e:
        st.error(f"❌ 验证上传状态失败: {e}")
        return {
            "milvus_count": 0,
            "mongodb_count": 0,
            "success": False,
            "synchronized": False
        }


# ----- 调试和诊断函数 -----

def debug_collection_info(collection_name="text_vectors"):
    """
    调试集合信息，用于排查问题
    """
    try:
        st.subheader("🔍 集合调试信息")
        
        if utility.has_collection(collection_name):
            collection = Collection(collection_name)
            
            # 基本信息
            st.write(f"**集合名称:** {collection_name}")
            st.write(f"**记录数量:** {collection.num_entities}")
            
            # Schema信息
            st.write("**Schema字段:**")
            for field in collection.schema.fields:
                st.write(f"- {field.name}: {field.dtype} (主键: {field.is_primary})")
            
            # 索引信息
            st.write("**索引信息:**")
            try:
                indexes = collection.indexes
                if indexes:
                    for idx in indexes:
                        st.write(f"- 字段: {idx.field_name}, 类型: {idx.params}")
                else:
                    st.write("- 无索引")
            except Exception as e:
                st.write(f"- 索引查询失败: {e}")
            
        else:
            st.write(f"❌ 集合 {collection_name} 不存在")
            
    except Exception as e:
        st.error(f"❌ 调试信息获取失败: {e}")
