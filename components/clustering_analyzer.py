import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from umap import UMAP
import streamlit as st
from typing import List, Dict, Any, Tuple

class ClusteringAnalyzer:
    def __init__(self):
        """
        初始化聚类分析器
        """
        self.vectors = None
        self.texts = None
        self.metadata = None
        self.cluster_labels = None
        self.reduced_vectors = None
    
    def load_data(self, vectors: np.ndarray, texts: List[str], metadata: List[Dict]):
        """
        加载数据
        """
        self.vectors = vectors
        self.texts = texts
        self.metadata = metadata
    
    def perform_kmeans_clustering(self, n_clusters: int = 8, random_state: int = 42) -> np.ndarray:
        """
        执行K-means聚类
        """
        if self.vectors is None:
            st.error("请先加载数据")
            return np.array([])
        
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
            self.cluster_labels = kmeans.fit_predict(self.vectors)
            
            # 计算轮廓系数
            if len(set(self.cluster_labels)) > 1:
                silhouette_avg = silhouette_score(self.vectors, self.cluster_labels)
                st.info(f"K-means聚类完成，轮廓系数: {silhouette_avg:.3f}")
            
            return self.cluster_labels
            
        except Exception as e:
            st.error(f"K-means聚类失败: {e}")
            return np.array([])
    
    def perform_dbscan_clustering(self, eps: float = 0.5, min_samples: int = 5) -> np.ndarray:
        """
        执行DBSCAN聚类
        """
        if self.vectors is None:
            st.error("请先加载数据")
            return np.array([])
        
        try:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
            self.cluster_labels = dbscan.fit_predict(self.vectors)
            
            n_clusters = len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0)
            n_noise = list(self.cluster_labels).count(-1)
            
            st.info(f"DBSCAN聚类完成，发现 {n_clusters} 个聚类，{n_noise} 个噪声点")
            
            return self.cluster_labels
            
        except Exception as e:
            st.error(f"DBSCAN聚类失败: {e}")
            return np.array([])
    
    def reduce_dimensions(self, n_components: int = 2, random_state: int = 42) -> np.ndarray:
        """
        使用UMAP进行降维
        """
        if self.vectors is None:
            st.error("请先加载数据")
            return np.array([])
        
        try:
            umap_reducer = UMAP(n_components=n_components, random_state=random_state, metric='cosine')
            self.reduced_vectors = umap_reducer.fit_transform(self.vectors)
            
            st.success(f"UMAP降维完成，从 {self.vectors.shape[1]} 维降至 {n_components} 维")
            return self.reduced_vectors
            
        except Exception as e:
            st.error(f"UMAP降维失败: {e}")
            return np.array([])
    
    def create_cluster_visualization(self) -> go.Figure:
        """
        创建聚类可视化图
        """
        if self.reduced_vectors is None or self.cluster_labels is None:
            st.error("请先进行降维和聚类")
            return go.Figure()
        
        try:
            # 创建DataFrame
            df = pd.DataFrame({
                'x': self.reduced_vectors[:, 0],
                'y': self.reduced_vectors[:, 1],
                'cluster': self.cluster_labels.astype(str),
                'text': [text[:100] + '...' if len(text) > 100 else text for text in self.texts]
            })
            
            # 创建散点图
            fig = px.scatter(
                df, 
                x='x', 
                y='y', 
                color='cluster',
                hover_data=['text'],
                title='文本聚类可视化',
                labels={'x': 'UMAP维度1', 'y': 'UMAP维度2'}
            )
            
            fig.update_layout(
                width=800,
                height=600,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            st.error(f"创建可视化失败: {e}")
            return go.Figure()
    
    def get_cluster_summary(self) -> Dict[str, Any]:
        """
        获取聚类摘要信息
        """
        if self.cluster_labels is None:
            return {}
        
        try:
            unique_labels = np.unique(self.cluster_labels)
            cluster_summary = {}
            
            for label in unique_labels:
                mask = self.cluster_labels == label
                cluster_texts = [self.texts[i] for i in range(len(self.texts)) if mask[i]]
                
                cluster_summary[str(label)] = {
                    'size': int(np.sum(mask)),
                    'percentage': float(np.sum(mask) / len(self.cluster_labels) * 100),
                    'sample_texts': cluster_texts[:5]  # 显示前5个样本
                }
            
            return cluster_summary
            
        except Exception as e:
            st.error(f"获取聚类摘要失败: {e}")
            return {}
    
    def find_optimal_k(self, max_k: int = 20) -> Tuple[List[int], List[float]]:
        """
        使用肘部法则找到最优的K值
        """
        if self.vectors is None:
            st.error("请先加载数据")
            return [], []
        
        try:
            k_range = range(2, min(max_k + 1, len(self.vectors)))
            inertias = []
            silhouette_scores = []
            
            progress_bar = st.progress(0)
            
            for i, k in enumerate(k_range):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(self.vectors)
                
                inertias.append(kmeans.inertia_)
                silhouette_scores.append(silhouette_score(self.vectors, labels))
                
                progress_bar.progress((i + 1) / len(k_range))
            
            progress_bar.empty()
            return list(k_range), silhouette_scores
            
        except Exception as e:
            st.error(f"寻找最优K值失败: {e}")
            return [], []
    
    def export_cluster_results(self) -> pd.DataFrame:
        """
        导出聚类结果
        """
        if self.cluster_labels is None:
            st.error("请先进行聚类")
            return pd.DataFrame()
        
        try:
            results_df = pd.DataFrame({
                'text': self.texts,
                'cluster': self.cluster_labels,
                'metadata': [str(meta) for meta in self.metadata]
            })
            
            if self.reduced_vectors is not None:
                results_df['umap_x'] = self.reduced_vectors[:, 0]
                results_df['umap_y'] = self.reduced_vectors[:, 1]
            
            return results_df
            
        except Exception as e:
            st.error(f"导出结果失败: {e}")
            return pd.DataFrame()
