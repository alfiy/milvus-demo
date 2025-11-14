from pymongo import MongoClient
import numpy as np
import streamlit as st
import logging
from typing import List, Dict, Any, Optional, Tuple
import time
from dataclasses import dataclass
from sklearn.preprocessing import normalize

# ============================================================
# 数据类和异常类
# ============================================================

@dataclass
class SearchResult:
    """搜索结果数据类"""
    id: str
    score: float
    text: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "score": self.score,
            "text": self.text,
            "metadata": self.metadata
        }


class VectorSearchError(Exception):
    """向量搜索异常基类"""
    pass


class VectorEncodingError(VectorSearchError):
    """向量编码异常"""
    pass


class MilvusSearchError(VectorSearchError):
    """Milvus搜索异常"""
    pass


class MongoQueryError(VectorSearchError):
    """MongoDB查询异常"""
    pass


# ============================================================
# 主搜索函数
# ============================================================

def vector_search(
    query: str,
    top_k: int,
    milvus_collection,
    mongo_col,
    vector_processor,
    filter_mode: str = "similarity",
    filter_threshold: float = 0.0,
    output_fields: List[str] = None,
    nprobe: int = 10,
    timeout: float = 30.0,
    enable_stats: bool = False,
    metric_type: str = "COSINE"  # 🎯 新增：支持配置指标类型
) -> List[Dict[str, Any]]:
    """
    优化后的向量+Mongo混合搜索功能
    
    Args:
        query: 查询文本
        top_k: 返回top-k结果
        milvus_collection: Milvus集合对象
        mongo_col: MongoDB集合对象
        vector_processor: 向量化处理器
        filter_mode: 过滤模式，"similarity"(相似度，越大越好) 或 "distance"(距离，越小越好)
        filter_threshold: 过滤阈值
        output_fields: 从MongoDB获取的字段列表
        nprobe: Milvus搜索的nprobe参数
        timeout: 搜索超时时间(秒)
        enable_stats: 是否返回统计信息（会添加到第一个结果的metadata中）
        metric_type: 距离度量类型，默认"COSINE"
        
    Returns:
        结果列表，每个结果包含: id, score, text, metadata
        如果enable_stats=True，会在第一个结果的metadata中添加'_search_stats'字段
        
    Raises:
        VectorSearchError: 搜索过程中的各类异常
    """
    stats = {"start_time": time.time()} if enable_stats else None
    
    # 参数验证
    _validate_params(query, top_k, filter_mode, filter_threshold, output_fields)
    
    if output_fields is None:
        output_fields = ["text", "metadata"]
    
    try:
        # 1. 向量化查询
        query_vector = _encode_query(query, vector_processor, stats)
        
        # 2. Milvus向量搜索
        milvus_results = _search_milvus(
            query_vector=query_vector,
            milvus_collection=milvus_collection,
            top_k=top_k,
            output_fields=output_fields,
            nprobe=nprobe,
            timeout=timeout,
            stats=stats,
            metric_type=metric_type  # 🎯 传递metric_type
        )
        
        if not milvus_results:
            logging.warning("Milvus返回空结果")
            return []
        
        # 3. 批量查询MongoDB元数据
        enriched_results = _enrich_with_mongo(
            milvus_results=milvus_results,
            mongo_col=mongo_col,
            output_fields=output_fields,
            timeout=timeout,
            stats=stats
        )
        
        # 4. 应用过滤
        filtered_results = _apply_filter(
            results=enriched_results,
            filter_mode=filter_mode,
            filter_threshold=filter_threshold,
            stats=stats
        )
        
        if enable_stats and filtered_results:
            stats["total_time"] = time.time() - stats["start_time"]
            stats["final_count"] = len(filtered_results)
            # 将统计信息添加到第一个结果的metadata中
            filtered_results[0]["_search_stats"] = stats
        
        return filtered_results
        
    except VectorSearchError:
        raise
    except Exception as e:
        logging.error(f"向量检索未知错误: {e}", exc_info=True)
        raise VectorSearchError(f"搜索失败: {str(e)}") from e


# ============================================================
# 辅助函数
# ============================================================

def _validate_params(
    query: str,
    top_k: int,
    filter_mode: str,
    filter_threshold: float,
    output_fields: Optional[List[str]]
) -> None:
    """验证输入参数"""
    if not query or not query.strip():
        raise ValueError("查询文本不能为空")
    
    if top_k <= 0:
        raise ValueError(f"top_k必须大于0，当前值: {top_k}")
    
    if top_k > 1000:
        logging.warning(f"top_k={top_k}过大，可能影响性能")
    
    if filter_mode not in ["similarity", "distance"]:
        raise ValueError(f"filter_mode必须是'similarity'或'distance'，当前值: {filter_mode}")
    
    if not 0 <= filter_threshold <= 1:
        logging.warning(f"filter_threshold={filter_threshold}可能超出常规范围[0, 1]")
    
    if output_fields and not isinstance(output_fields, list):
        raise ValueError("output_fields必须是列表类型")


def _encode_query(query: str, vector_processor, stats: Optional[Dict]) -> List[float]:
    """向量化查询文本"""
    try:
        if stats is not None:
            encode_start = time.time()
        
        query_vector = vector_processor.encode([query])[0]
        
        # 向量归一化（可选，根据实际需求决定是否使用）
        # 注意：如果训练数据没有归一化，这里也不应该归一化
        query_vector = normalize([query_vector], axis=1)[0]
        
        if stats is not None:
            stats["encode_time"] = time.time() - encode_start
            stats["vector_dim"] = len(query_vector)
        
        return query_vector.tolist() if hasattr(query_vector, 'tolist') else list(query_vector)
        
    except Exception as e:
        logging.error(f"向量编码失败: {e}")
        raise VectorEncodingError(f"无法编码查询文本: {str(e)}") from e


def _search_milvus(
    query_vector: List[float],
    milvus_collection,
    top_k: int,
    output_fields: List[str],
    nprobe: int,
    timeout: float,
    stats: Optional[Dict],
    metric_type: str = "COSINE"  # 🎯 新增参数
) -> List[Tuple[str, float]]:
    """
    执行Milvus向量搜索
    
    Returns:
        List[Tuple[str, float]]: (文档ID, 分数) 列表
        - COSINE: 返回相似度 [0, 1]，1表示完全相同
        - L2: 返回距离，越小越相似
        - IP: 返回内积，越大越相似
    """
    try:
        if stats is not None:
            milvus_start = time.time()
        
        # 🎯 优化1：输入验证
        if not query_vector:
            raise ValueError("查询向量不能为空")
        
        if top_k <= 0:
            raise ValueError(f"top_k必须大于0，当前值: {top_k}")
        
        # 🎯 优化2：使用传入的metric_type
        search_params = {
            "metric_type": metric_type,
            "params": {"nprobe": nprobe}
        }
        
        # 🎯 优化3：记录搜索参数（便于调试）
        if stats is not None:
            stats["search_params"] = {
                "metric_type": metric_type,
                "nprobe": nprobe,
                "top_k": top_k,
                "vector_dim": len(query_vector)
            }
        
        results = milvus_collection.search(
            data=[query_vector],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            output_fields=[],  # 只获取ID和分数，元数据从MongoDB获取
            timeout=timeout
        )
        
        # 🎯 优化4：更详细的空结果检查
        if not results:
            logging.warning("Milvus返回空结果对象")
            return []
        
        if not results[0]:
            logging.warning("Milvus返回空搜索结果列表")
            return []
        
        # 🎯 优化5：根据指标类型智能转换分数
        milvus_results = _convert_scores(
            hits=results[0],
            metric_type=metric_type,
            stats=stats
        )
        
        # 🎯 优化6：记录更详细的统计信息
        if stats is not None:
            stats["milvus_time"] = time.time() - milvus_start
            stats["milvus_count"] = len(milvus_results)
            stats["milvus_metric"] = metric_type
            
            # 记录分数统计
            if milvus_results:
                scores = [score for _, score in milvus_results]
                stats["score_stats"] = {
                    "min": min(scores),
                    "max": max(scores),
                    "avg": sum(scores) / len(scores)
                }
        
        # 🎯 优化7：记录搜索质量警告
        if milvus_results:
            _log_search_quality_warnings(milvus_results, metric_type, top_k)
        
        return milvus_results
        
    except ValueError as e:
        # 参数错误
        logging.error(f"Milvus搜索参数错误: {e}")
        raise MilvusSearchError(f"搜索参数无效: {str(e)}") from e
        
    except Exception as e:
        # 其他错误
        logging.error(f"Milvus搜索失败: {e}", exc_info=True)
        raise MilvusSearchError(f"向量搜索失败: {str(e)}") from e


def _convert_scores(
    hits,
    metric_type: str,
    stats: Optional[Dict] = None
) -> List[Tuple[str, float]]:
    """
    根据度量类型转换搜索结果的分数
    
    Args:
        hits: Milvus搜索结果的hit对象列表
        metric_type: 度量类型
        stats: 统计信息字典（可选）
        
    Returns:
        (ID, 转换后的分数) 元组列表
        
    Note:
        不同度量类型的转换规则：
        - COSINE: 转换为相似度 (1 - distance)，范围 [0, 1]，越大越相似
        - L2: 保持距离值，范围 [0, ∞)，越小越相似
        - IP: 保持内积值，范围 (-∞, ∞)，越大越相似
    """
    results = []
    raw_distances = []
    
    for hit in hits:
        hit_id = str(hit.id)  # 统一转为字符串ID
        raw_distance = float(hit.distance)
        raw_distances.append(raw_distance)
        
        # 根据指标类型转换分数
        if metric_type == "COSINE":
            # COSINE: 将距离转为相似度
            score = 1.0 - raw_distance
            # 处理浮点数精度问题，确保在 [0, 1] 范围内
            score = max(0.0, min(1.0, score))
        elif metric_type == "L2":
            # L2: 保持距离值，越小越相似
            score = raw_distance
        elif metric_type == "IP":
            # IP (内积): 保持原值，越大越相似
            score = raw_distance
        else:
            # 其他未知指标：保持原值并记录警告
            logging.warning(f"未知的度量类型: {metric_type}，使用原始距离值")
            score = raw_distance
        
        results.append((hit_id, score))
    
    # 记录原始距离统计（用于调试）
    if stats is not None and raw_distances:
        stats["raw_distance_stats"] = {
            "min": min(raw_distances),
            "max": max(raw_distances),
            "avg": sum(raw_distances) / len(raw_distances)
        }
    
    return results


def _log_search_quality_warnings(
    results: List[Tuple[str, float]],
    metric_type: str,
    top_k: int
) -> None:
    """
    记录搜索质量相关的警告信息
    
    Args:
        results: 搜索结果列表
        metric_type: 度量类型
        top_k: 请求的结果数量
    """
    if not results:
        return
    
    scores = [score for _, score in results]
    
    # 警告1：返回结果少于请求数量
    if len(results) < top_k:
        logging.warning(
            f"返回结果数({len(results)})少于请求数({top_k})，"
            f"可能是集合中数据不足"
        )
    
    # 警告2：COSINE相似度都很低
    if metric_type == "COSINE":
        max_score = max(scores)
        avg_score = sum(scores) / len(scores)
        
        if max_score < 0.3:
            logging.warning(
                f"最高相似度仅为{max_score:.3f}，建议检查："
                "1) 查询文本是否合适 "
                "2) 向量模型是否匹配 "
                "3) 数据库中是否有相关内容"
            )
        elif avg_score < 0.2:
            logging.warning(
                f"平均相似度仅为{avg_score:.3f}，大部分结果可能不相关"
            )
    
    # 警告3：L2距离都很大
    elif metric_type == "L2":
        min_distance = min(scores)
        
        if min_distance > 100:  # 阈值可调整
            logging.warning(
                f"最小L2距离为{min_distance:.2f}，所有结果可能都不相关"
            )
    
    # 警告4：分数分布异常
    score_range = max(scores) - min(scores)
    if metric_type == "COSINE" and score_range < 0.01:
        logging.warning(
            f"相似度分数范围很小({score_range:.4f})，"
            "可能表示查询向量质量问题或数据同质化严重"
        )


def _enrich_with_mongo(
    milvus_results: List[Tuple[str, float]],
    mongo_col,
    output_fields: List[str],
    timeout: float,
    stats: Optional[Dict]
) -> List[SearchResult]:
    """用MongoDB数据丰富搜索结果"""
    try:
        if stats is not None:
            mongo_start = time.time()
        
        # 提取所有ID
        ids = [id_ for id_, _ in milvus_results]
        
        if not ids:
            return []
        
        # 批量查询MongoDB - 注意ID类型转换
        projection = {field: 1 for field in output_fields}
        projection["_id"] = 1
        
        docs = []
        id_mappings = {}  # 记录ID映射关系：MongoDB _id -> Milvus id
        
        # 策略1: 直接用字符串ID查询
        logging.debug(f"尝试字符串ID查询: {ids[:3]}...")
        docs = list(mongo_col.find(
            {"_id": {"$in": ids}},
            projection
        ).max_time_ms(int(timeout * 1000)))
        
        if docs:
            logging.info(f"字符串ID查询成功，找到 {len(docs)} 条记录")
            for doc in docs:
                id_mappings[str(doc["_id"])] = str(doc["_id"])
        
        # 策略2: 尝试转换为ObjectId
        if len(docs) < len(ids):
            try:
                from bson import ObjectId
                logging.debug("尝试ObjectId转换...")
                
                # 收集未找到的ID
                found_ids = {str(doc["_id"]) for doc in docs}
                missing_ids = [id_ for id_ in ids if id_ not in found_ids]
                
                # 尝试转换为ObjectId
                object_ids = []
                oid_to_str = {}  # ObjectId -> 原始字符串映射
                
                for id_str in missing_ids:
                    if ObjectId.is_valid(id_str):
                        oid = ObjectId(id_str)
                        object_ids.append(oid)
                        oid_to_str[oid] = id_str
                
                if object_ids:
                    oid_docs = list(mongo_col.find(
                        {"_id": {"$in": object_ids}},
                        projection
                    ).max_time_ms(int(timeout * 1000)))
                    
                    if oid_docs:
                        logging.info(f"ObjectId查询成功，找到 {len(oid_docs)} 条记录")
                        for doc in oid_docs:
                            # 建立映射：原始Milvus字符串ID -> MongoDB文档
                            original_str = oid_to_str.get(doc["_id"])
                            if original_str:
                                id_mappings[original_str] = str(doc["_id"])
                        docs.extend(oid_docs)
                        
            except Exception as oid_error:
                logging.warning(f"ObjectId转换失败: {oid_error}")
        
        # 策略3: 尝试转换为整数ID
        if len(docs) < len(ids):
            try:
                logging.debug("尝试整数ID转换...")
                found_ids = set(id_mappings.keys())
                missing_ids = [id_ for id_ in ids if id_ not in found_ids]
                
                int_ids = []
                int_to_str = {}
                
                for id_str in missing_ids:
                    try:
                        int_id = int(id_str)
                        int_ids.append(int_id)
                        int_to_str[int_id] = id_str
                    except ValueError:
                        continue
                
                if int_ids:
                    int_docs = list(mongo_col.find(
                        {"_id": {"$in": int_ids}},
                        projection
                    ).max_time_ms(int(timeout * 1000)))
                    
                    if int_docs:
                        logging.info(f"整数ID查询成功，找到 {len(int_docs)} 条记录")
                        for doc in int_docs:
                            original_str = int_to_str.get(doc["_id"])
                            if original_str:
                                id_mappings[original_str] = str(doc["_id"])
                        docs.extend(int_docs)
                        
            except Exception as int_error:
                logging.warning(f"整数ID转换失败: {int_error}")
        
        # 构建ID到文档的映射
        id2doc = {}
        for doc in docs:
            doc_id_str = str(doc["_id"])
            id2doc[doc_id_str] = doc
            # 如果有映射关系，也添加原始ID的映射
            for milvus_id, mongo_id in id_mappings.items():
                if mongo_id == doc_id_str:
                    id2doc[milvus_id] = doc
        
        # 按Milvus返回顺序构建结果
        enriched = []
        missing_count = 0
        missing_ids_list = []
        
        for id_, score in milvus_results:
            doc = id2doc.get(id_)
            
            if doc:
                enriched.append(SearchResult(
                    id=id_,
                    score=score,
                    text=doc.get("text", ""),
                    metadata=doc.get("metadata", {})
                ))
            else:
                missing_count += 1
                missing_ids_list.append(id_)
                logging.warning(f"MongoDB中未找到ID: {id_} (类型: {type(id_).__name__})")
                # 保留该结果但标记为缺失
                enriched.append(SearchResult(
                    id=id_,
                    score=score,
                    text="[数据缺失]",
                    metadata={
                        "_missing": True,
                        "_milvus_id": id_,
                        "_note": "该记录存在于Milvus但在MongoDB中找不到"
                    }
                ))
        
        if stats is not None:
            stats["mongo_time"] = time.time() - mongo_start
            stats["mongo_found"] = len(docs)
            stats["mongo_missing"] = missing_count
            stats["missing_ids"] = missing_ids_list[:5]  # 只记录前5个
        
        if missing_count > 0:
            logging.error(f"⚠️ 有 {missing_count}/{len(milvus_results)} 条记录在MongoDB中缺失")
            logging.error(f"示例缺失ID: {missing_ids_list[:3]}")
        
        return enriched
        
    except Exception as e:
        logging.error(f"MongoDB查询失败: {e}", exc_info=True)
        raise MongoQueryError(f"元数据查询失败: {str(e)}") from e


def _apply_filter(
    results: List[SearchResult],
    filter_mode: str,
    filter_threshold: float,
    stats: Optional[Dict]
) -> List[Dict[str, Any]]:
    """应用相似度/距离过滤"""
    if filter_mode == "similarity":
        # COSINE相似度: 值越大越相似，保留 >= threshold
        filtered = [r for r in results if r.score >= filter_threshold]
    else:  # distance
        # 距离: 值越小越相似，保留 <= threshold
        filtered = [r for r in results if r.score <= filter_threshold]
    
    if stats is not None:
        stats["before_filter"] = len(results)
        stats["after_filter"] = len(filtered)
        stats["filtered_out"] = len(results) - len(filtered)
    
    return [r.to_dict() for r in filtered]


# ============================================================
# MongoDB 辅助函数（保留原有功能）
# ============================================================

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