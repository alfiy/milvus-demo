import numpy as np
from typing import List, Dict, Any
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class SearchEngine:
    def __init__(self):
        """
        初始化搜索引擎
        """
        self.vectors = None
        self.texts = None
        self.metadata = None
        self.vector_processor = None
        self.milvus_manager = None
    
    def load_data(self, vectors: np.ndarray, texts: List[str], metadata: List[Dict]):
        """
        加载本地数据用于搜索
        """
        self.vectors = vectors
        self.texts = texts
        self.metadata = metadata
    
    def set_vector_processor(self, processor):
        """
        设置向量处理器
        """
        self.vector_processor = processor
    
    def set_milvus_manager(self, manager):
        """
        设置Milvus管理器
        """
        self.milvus_manager = manager
    
    def search_local(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        在本地数据中搜索
        """
        if not self.vector_processor or self.vectors is None:
            st.error("搜索引擎未正确初始化")
            return []
        
        try:
            # 对查询文本进行向量化
            query_vector = self.vector_processor.encode_single_text(query)
            if query_vector.size == 0:
                return []
            
            # 计算余弦相似度
            similarities = cosine_similarity([query_vector], self.vectors)[0]
            
            # 获取top_k结果
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                results.append({
                    'index': int(idx),
                    'text': self.texts[idx],
                    'metadata': self.metadata[idx],
                    'similarity': float(similarities[idx]),
                    'score': float(similarities[idx])
                })
            
            return results
            
        except Exception as e:
            st.error(f"本地搜索失败: {e}")
            return []
    
    def search_milvus(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        在Milvus数据库中搜索
        """
        if not self.vector_processor or not self.milvus_manager:
            st.error("搜索引擎未正确初始化")
            return []
        
        if not self.milvus_manager.is_connected:
            st.error("Milvus未连接")
            return []
        
        try:
            # 对查询文本进行向量化
            query_vector = self.vector_processor.encode_single_text(query)
            if query_vector.size == 0:
                return []
            
            # 在Milvus中搜索
            results = self.milvus_manager.search_similar(query_vector, top_k)
            return results
            
        except Exception as e:
            st.error(f"Milvus搜索失败: {e}")
            return []
    
    def batch_search(self, queries: List[str], top_k: int = 5, use_milvus: bool = False) -> Dict[str, List[Dict]]:
        """
        批量搜索
        """
        results = {}
        
        progress_bar = st.progress(0)
        
        for i, query in enumerate(queries):
            if use_milvus and self.milvus_manager:
                search_results = self.search_milvus(query, top_k)
            else:
                search_results = self.search_local(query, top_k)
            
            results[query] = search_results
            progress_bar.progress((i + 1) / len(queries))
        
        progress_bar.empty()
        return results
    
    def semantic_search_with_filters(self, query: str, filters: Dict[str, Any] = None, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        带过滤条件的语义搜索
        """
        # 先进行基本搜索
        if self.milvus_manager and self.milvus_manager.is_connected:
            results = self.search_milvus(query, top_k * 2)  # 获取更多结果用于过滤
        else:
            results = self.search_local(query, top_k * 2)
        
        # 应用过滤条件
        if filters:
            filtered_results = []
            for result in results:
                metadata = result.get('metadata', {})
                
                # 检查是否满足过滤条件
                match = True
                for key, value in filters.items():
                    if key in metadata:
                        if isinstance(value, str):
                            if value.lower() not in str(metadata[key]).lower():
                                match = False
                                break
                        elif metadata[key] != value:
                            match = False
                            break
                
                if match:
                    filtered_results.append(result)
                
                if len(filtered_results) >= top_k:
                    break
            
            return filtered_results
        
        return results[:top_k]
    
    def find_similar_texts(self, reference_text: str, top_k: int = 10, exclude_self: bool = True) -> List[Dict[str, Any]]:
        """
        找到与参考文本相似的文本
        """
        results = self.search_local(reference_text, top_k + (1 if exclude_self else 0))
        
        if exclude_self:
            # 排除完全相同的文本
            filtered_results = []
            for result in results:
                if result['text'].strip() != reference_text.strip():
                    filtered_results.append(result)
                if len(filtered_results) >= top_k:
                    break
            return filtered_results
        
        return results[:top_k]
    
    def get_search_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        获取搜索结果统计信息
        """
        if not results:
            return {}
        
        scores = [result['score'] for result in results]
        
        stats = {
            'total_results': len(results),
            'avg_score': np.mean(scores),
            'max_score': np.max(scores),
            'min_score': np.min(scores),
            'score_std': np.std(scores)
        }
        
        return stats
    
    def export_search_results(self, results: List[Dict[str, Any]], query: str = "") -> pd.DataFrame:
        """
        导出搜索结果为DataFrame
        """
        if not results:
            return pd.DataFrame()
        
        try:
            df_data = []
            for i, result in enumerate(results):
                df_data.append({
                    'rank': i + 1,
                    'query': query,
                    'text': result['text'],
                    'score': result['score'],
                    'metadata': str(result.get('metadata', {}))
                })
            
            return pd.DataFrame(df_data)
            
        except Exception as e:
            st.error(f"导出搜索结果失败: {e}")
            return pd.DataFrame()
