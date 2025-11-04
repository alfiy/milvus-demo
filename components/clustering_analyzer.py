import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
import plotly.express as px
import plotly.graph_objects as go
from umap import UMAP
import streamlit as st
from typing import List, Dict, Any, Tuple
import warnings
import hdbscan
from itertools import product
from sklearn.metrics import davies_bouldin_score, silhouette_score, davies_bouldin_score



# 抑制警告
warnings.filterwarnings('ignore', message='.*n_jobs.*overridden.*')
warnings.filterwarnings('ignore', message='.*TBB threading layer.*')
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')

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
                st.info(f"✅ K-means聚类完成，轮廓系数: {silhouette_avg:.3f}")
            
            return self.cluster_labels
            
        except Exception as e:
            st.error(f"❌ K-means聚类失败: {e}")
            return np.array([])
    
    def perform_dbscan_clustering(self, eps: float = 0.3, min_samples: int = 5) -> np.ndarray:
        """
        执行DBSCAN聚类
        
        Args:
            eps: 邻域半径
                - cosine距离：推荐范围 0.1-0.5（值越小越严格）
                - euclidean距离：需要根据数据尺度调整
            min_samples: 核心点的最小邻居数，推荐 5-10
        """
        if self.vectors is None:
            st.error("❌ 请先加载数据")
            return np.array([])
        
        try:
            #  关键修复1：先进行降维，再聚类
            if self.reduced_vectors is None:
                st.warning("⚠️ 建议先进行降维以提高DBSCAN效果，正在使用原始向量...")
                vectors_to_cluster = self.vectors
            else:
                vectors_to_cluster = self.reduced_vectors
                st.info(" 使用降维后的向量进行DBSCAN聚类")
            
            #  关键修复2：根据数据维度选择合适的距离度量
            if vectors_to_cluster.shape[1] > 50:
                # 高维数据使用余弦距离
                metric = 'cosine'
                # 自动调整eps（如果用户使用默认值）
                if eps == 0.5:
                    eps = 0.3  # 更合理的默认值
                    st.info(f" 高维数据自动调整 eps={eps}")
            else:
                # 低维数据可以使用欧氏距离
                metric = 'euclidean'
                st.info(f" 低维数据使用欧氏距离")
            
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
            self.cluster_labels = dbscan.fit_predict(vectors_to_cluster)
            
            #  关键修复3：详细的聚类诊断信息
            unique_labels = set(self.cluster_labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            n_noise = list(self.cluster_labels).count(-1)
            
            # 显示每个聚类的大小
            cluster_sizes = {}
            for label in unique_labels:
                if label != -1:
                    cluster_sizes[f"簇 {label}"] = int((self.cluster_labels == label).sum())
            
            st.info(f"**✅ DBSCAN聚类完成**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("聚类数量", n_clusters)
            with col2:
                st.metric("噪声点", f"{n_noise} ({n_noise/len(self.cluster_labels)*100:.1f}%)")
            with col3:
                st.metric("有效点", len(self.cluster_labels) - n_noise)
            
            st.write(f"**参数设置**: eps={eps}, min_samples={min_samples}, metric={metric}")
            
            if n_clusters > 0:
                st.write("**各聚类大小**:", cluster_sizes)
            
            #  关键修复4：给出参数调整建议
            if n_clusters == 0:
                st.warning("⚠️ **未发现任何聚类**，建议:")
                st.markdown(f"""
                -  增大 `eps` 值（当前: {eps}，建议尝试: {eps*1.5:.2f}）
                -  减小 `min_samples` 值（当前: {min_samples}，建议尝试: {max(2, min_samples-2)}）
                -  或先使用 UMAP 降维至 2-3 维
                """)
            elif n_clusters == 1:
                st.warning("⚠️ **只发现1个聚类**，建议:")
                st.markdown(f"""
                -  减小 `eps` 值以分离更多聚类（建议尝试: {eps*0.7:.2f}）
                -  或增大 `min_samples` 以提高密度要求（建议尝试: {min_samples+3}）
                """)
            elif n_noise / len(self.cluster_labels) > 0.5:
                st.warning("⚠️ **噪声点过多**（>{n_noise/len(self.cluster_labels)*100:.0f}%），建议:")
                st.markdown(f"""
                -  增大 `eps` 值（建议尝试: {eps*1.3:.2f}）
                -  或减小 `min_samples` 值（建议尝试: {max(2, min_samples-2)}）
                """)
            else:
                st.success("✅ 聚类结果良好！")
            
            return self.cluster_labels
            
        except Exception as e:
            st.error(f"❌ DBSCAN聚类失败: {e}")
            import traceback
            st.code(traceback.format_exc())
            return np.array([])
    
    def find_optimal_dbscan_params(self, eps_range: List[float] = None, 
                                   min_samples_range: List[int] = None) -> Dict:
        """
        自动搜索最优的DBSCAN参数
        """
        if self.vectors is None:
            st.error("❌ 请先加载数据")
            return {}
        
        # 使用降维后的向量（如果可用）
        vectors = self.reduced_vectors if self.reduced_vectors is not None else self.vectors
        
        st.info(f" 开始搜索最优DBSCAN参数（数据维度: {vectors.shape[1]}）")
        
        # 默认搜索范围
        if eps_range is None:
            if vectors.shape[1] > 50:  # 高维
                eps_range = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
                metric = 'cosine'
            else:  # 低维
                eps_range = np.linspace(0.3, 2.0, 7).tolist()
                metric = 'euclidean'
        else:
            metric = 'cosine' if vectors.shape[1] > 50 else 'euclidean'
        
        if min_samples_range is None:
            min_samples_range = [3, 5, 7, 10, 15]
        
        results = []
        best_score = -1
        best_params = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_iterations = len(eps_range) * len(min_samples_range)
        iteration = 0
        
        for eps in eps_range:
            for min_samples in min_samples_range:
                status_text.text(f"测试参数: eps={eps:.2f}, min_samples={min_samples}")
                
                dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
                labels = dbscan.fit_predict(vectors)
                
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                noise_ratio = n_noise / len(labels)
                
                # 评分：优先考虑有多个聚类且噪声适中的结果
                if n_clusters > 1 and noise_ratio < 0.5:
                    try:
                        score = silhouette_score(vectors, labels)
                        # 综合评分：轮廓系数 - 噪声惩罚
                        score = score * (1 - noise_ratio * 0.5)
                    except:
                        score = 0
                else:
                    score = -1
                
                results.append({
                    'eps': eps,
                    'min_samples': min_samples,
                    'n_clusters': n_clusters,
                    'noise_ratio': f"{noise_ratio*100:.1f}%",
                    'noise_count': n_noise,
                    'score': score
                })
                
                if score > best_score:
                    best_score = score
                    best_params = {'eps': eps, 'min_samples': min_samples, 
                                  'n_clusters': n_clusters, 'score': score}
                
                iteration += 1
                progress_bar.progress(iteration / total_iterations)
        
        progress_bar.empty()
        status_text.empty()
        
        # 显示结果
        results_df = pd.DataFrame(results).sort_values('score', ascending=False)
        
        st.write("###  参数搜索结果（按得分排序，前10名）")
        st.dataframe(
            results_df.head(10).style.format({
                'eps': '{:.3f}',
                'score': '{:.3f}'
            }),
            use_container_width=True
        )
        
        if best_score > -1:
            st.success(f"""
            ### ✅ 推荐参数
            - **eps**: {best_params['eps']}
            - **min_samples**: {best_params['min_samples']}
            - **预期聚类数**: {best_params['n_clusters']}
            - **得分**: {best_params['score']:.3f}
            """)
        else:
            st.warning("""
            ### ⚠️ 未找到理想参数
            建议:
            - 先进行降维 (UMAP降至2-3维)
            - 或使用 K-means 聚类
            - 或调整搜索范围
            """)
        
        return best_params
    
    # hdbscan聚类算法
    def perform_hdbscan_clustering(
        self, 
        min_cluster_size: int = 5, 
        min_samples: int = None,
        cluster_selection_epsilon: float = 0.0
    ) -> np.ndarray:
        """
        执行 HDBSCAN 聚类
        Args:
            min_cluster_size: 每个聚类的最小样本数（推荐 5-15）
            min_samples: 控制噪声敏感度（默认等于 min_cluster_size）
            cluster_selection_epsilon: 控制聚类边界的宽松程度（默认 0）
        """
        

        if self.vectors is None:
            st.error("❌ 请先加载数据")
            return np.array([])

        try:
            # 选择聚类输入向量
            if self.reduced_vectors is not None:
                vectors_to_cluster = self.reduced_vectors
                st.info(" 使用降维后的向量进行 HDBSCAN 聚类")
            else:
                st.warning("⚠️ 未进行降维，正在使用原始向量进行 HDBSCAN 聚类（建议先 UMAP 降维）")
                vectors_to_cluster = self.vectors

            # 自动选择度量方式
            if vectors_to_cluster.shape[1] > 50:
                metric = 'cosine'
                st.info(" 高维数据使用余弦距离 (cosine)")
            else:
                metric = 'euclidean'
                st.info(" 低维数据使用欧氏距离 (euclidean)")

            # 创建并拟合 HDBSCAN 模型
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric=metric,
                cluster_selection_epsilon=cluster_selection_epsilon,
                cluster_selection_method='eom'
            )

            self.cluster_labels = clusterer.fit_predict(vectors_to_cluster)
            unique_labels = set(self.cluster_labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            n_noise = list(self.cluster_labels).count(-1)

            st.info("**✅ HDBSCAN 聚类完成**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("聚类数量", n_clusters)
            with col2:
                st.metric("噪声点", f"{n_noise} ({n_noise/len(self.cluster_labels)*100:.1f}%)")
            with col3:
                st.metric("有效点", len(self.cluster_labels) - n_noise)

            # 各聚类大小
            cluster_sizes = {
                f"簇 {label}": int((self.cluster_labels == label).sum())
                for label in unique_labels if label != -1
            }
            if cluster_sizes:
                st.write("**各聚类大小:**", cluster_sizes)

            # 尝试计算轮廓系数（需多个聚类且噪声点 < 90%）
            silhouette_avg = None
            if n_clusters > 1 and n_noise / len(self.cluster_labels) < 0.9:
                from sklearn.metrics import silhouette_score
                silhouette_avg = silhouette_score(vectors_to_cluster, self.cluster_labels)
                st.success(f"平均轮廓系数: {silhouette_avg:.3f}")

            # 参数建议
            if n_clusters == 0:
                st.warning("⚠️ 未发现任何聚类。建议：")
                st.markdown(f"""
                - 减小 `min_cluster_size`（当前 {min_cluster_size} → 建议 {max(2, min_cluster_size // 2)}）
                - 或增加降维维度
                """)
            elif n_clusters == 1:
                st.warning("⚠️ 只发现 1 个聚类。建议：")
                st.markdown(f"""
                - 增大 `min_cluster_size` 或 `min_samples`
                - 调整 `cluster_selection_epsilon` (当前 {cluster_selection_epsilon} → 建议 {cluster_selection_epsilon + 0.05})
                """)
            elif n_noise / len(self.cluster_labels) > 0.5:
                st.warning(f"⚠️ 噪声点过多 ({n_noise/len(self.cluster_labels)*100:.1f}%)。建议：")
                st.markdown(f"""
                - 减小 `min_samples`（当前 {min_samples or min_cluster_size} → 建议 {max(2, (min_samples or min_cluster_size)//2)}）
                - 增加 `cluster_selection_epsilon`
                """)

            else:
                st.success("✅ 聚类结果良好！")

            return self.cluster_labels

        except Exception as e:
            st.error(f"❌ HDBSCAN 聚类失败: {e}")
            import traceback
            st.code(traceback.format_exc())
            return np.array([])

    # 自动优化hdbscan参数 
    def find_optimal_hdbscan_params(
        self,
        min_cluster_size_range: Tuple[int, int, int] = (5, 50, 5),
        min_samples_range: Tuple[int, int, int] = (1, 10, 2),
        metrics: Tuple[str, ...] = ("euclidean", "manhattan"),
        cluster_selection_methods: Tuple[str, ...] = ("eom", "leaf"),
        scoring: str = "silhouette",
    ):
        """
        自动搜索最佳 HDBSCAN 聚类参数（带 Streamlit 实时进度与表格展示）
        """

        if self.vectors is None:
            st.error("❌ 请先加载或生成嵌入向量！")
            return None, None, None, None

        st.subheader("🔍 HDBSCAN 参数搜索中...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        result_table = st.empty()

        embeddings = self.reduced_vectors if self.reduced_vectors is not None else self.vectors

        min_cluster_sizes = range(*min_cluster_size_range)
        min_samples = range(*min_samples_range)
        total = len(list(product(min_cluster_sizes, min_samples, metrics, cluster_selection_methods)))
        current = 0

        all_results = []
        best_score = -np.inf if scoring == "silhouette" else np.inf
        best_model, best_params = None, None

        for mcs, ms, metric, method in product(min_cluster_sizes, min_samples, metrics, cluster_selection_methods):
            current += 1
            progress = current / total
            status_text.text(f"正在搜索: min_cluster_size={mcs}, min_samples={ms}, metric={metric}, method={method} ({current}/{total})")
            progress_bar.progress(progress)

            try:
                model = hdbscan.HDBSCAN(
                    min_cluster_size=mcs,
                    min_samples=ms,
                    metric=metric,
                    cluster_selection_method=method,
                    prediction_data=True,
                    gen_min_span_tree=False
                ).fit(embeddings)

                labels = model.labels_
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

                # 跳过全噪声或仅一个簇的情况
                if n_clusters < 2:
                    continue

                if scoring == "silhouette":
                    score = silhouette_score(embeddings, labels)
                    better = score > best_score
                else:
                    score = davies_bouldin_score(embeddings, labels)
                    better = score < best_score

                result = {
                    "min_cluster_size": mcs,
                    "min_samples": ms,
                    "metric": metric,
                    "method": method,
                    "n_clusters": n_clusters,
                    "score": round(score, 4),
                }
                all_results.append(result)

                if better:
                    best_score = score
                    best_model = model
                    best_params = {
                        "min_cluster_size": mcs,
                        "min_samples": ms,
                        "metric": metric,
                        "cluster_selection_method": method,
                        "n_clusters": n_clusters,
                    }

                    st.success(f"✨ 新最优组合: {best_params} | score={score:.4f}")

                # 每次更新表格
                df = pd.DataFrame(all_results).sort_values(
                    "score", ascending=(scoring != "silhouette"), ignore_index=True
                )
                result_table.dataframe(df)

            except Exception as e:
                st.warning(f"⚠️ 参数组合失败: min_cluster_size={mcs}, min_samples={ms}, metric={metric} ({e})")
                continue

        progress_bar.progress(1.0)
        status_text.text("✅ 搜索完成！")

        if not all_results:
            st.error("❌ 未找到合适的聚类结果，请调整搜索范围或参数。")
            return None, None, None, None

        st.success(f"🏁 最佳参数: {best_params}")
        st.metric("最佳分数", f"{best_score:.4f}")
        st.metric("最佳聚类数", best_params['n_clusters'])

        return best_model, best_params, best_score, all_results


    def reduce_dimensions(self, n_components: int = 2, random_state: int = 42) -> np.ndarray:
        """
        使用UMAP进行降维
        
        """
        if self.vectors is None:
            st.error("❌ 请先加载数据")
            return np.array([])
        
        try:
            st.info(f" 正在进行UMAP降维...")
            
            # low_memory=True 可以避免某些 TBB 相关警告
            umap_reducer = UMAP(
                n_components=n_components, 
                random_state=random_state,  # 保留以确保可重复性
                metric='cosine', 
                n_neighbors=15, 
                min_dist=0.1,
                low_memory=True  # 减少内存使用，避免 TBB 警告
            )
            
            self.reduced_vectors = umap_reducer.fit_transform(self.vectors)
            
            st.success(f"✅ UMAP降维完成: {self.vectors.shape[1]} 维 → {n_components} 维")
            return self.reduced_vectors
            
        except Exception as e:
            st.error(f"❌ UMAP降维失败: {e}")
            import traceback
            st.code(traceback.format_exc())
            return np.array([])
    
    def create_cluster_visualization(self, use_3d: bool = False) -> go.Figure:
        """
        创建聚类可视化图
        
        Args:
            use_3d: 是否创建3D可视化（需要降维到3维）
        """
        if self.reduced_vectors is None or self.cluster_labels is None:
            st.error("❌ 请先进行降维和聚类")
            return go.Figure()
        
        try:
            # 创建DataFrame
            df = pd.DataFrame({
                'x': self.reduced_vectors[:, 0],
                'y': self.reduced_vectors[:, 1],
                'cluster': self.cluster_labels.astype(str),
                'text': [text[:100] + '...' if len(text) > 100 else text for text in self.texts]
            })
            
            # 标记噪声点
            df['cluster'] = df['cluster'].apply(lambda x: '噪声点' if x == '-1' else f'簇 {x}')
            
            # 创建散点图
            if use_3d and self.reduced_vectors.shape[1] >= 3:
                df['z'] = self.reduced_vectors[:, 2]
                fig = px.scatter_3d(
                    df, 
                    x='x', 
                    y='y',
                    z='z',
                    color='cluster',
                    hover_data=['text'],
                    title='文本聚类可视化 (3D)',
                    labels={'x': 'UMAP-1', 'y': 'UMAP-2', 'z': 'UMAP-3'}
                )
            else:
                fig = px.scatter(
                    df, 
                    x='x', 
                    y='y', 
                    color='cluster',
                    hover_data=['text'],
                    title='文本聚类可视化 (2D)',
                    labels={'x': 'UMAP维度1', 'y': 'UMAP维度2'}
                )
            
            fig.update_layout(
                width=900,
                height=700,
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            
            fig.update_traces(marker=dict(size=8, opacity=0.7))
            
            return fig
            
        except Exception as e:
            st.error(f"❌ 创建可视化失败: {e}")
            import traceback
            st.code(traceback.format_exc())
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
                
                label_name = "噪声点" if label == -1 else f"簇 {label}"
                
                cluster_summary[label_name] = {
                    'size': int(np.sum(mask)),
                    'percentage': float(np.sum(mask) / len(self.cluster_labels) * 100),
                    'sample_texts': cluster_texts[:5]  # 显示前5个样本
                }
            
            return cluster_summary
            
        except Exception as e:
            st.error(f"❌ 获取聚类摘要失败: {e}")
            return {}
    
    def find_optimal_k(self, max_k: int = 20) -> Tuple[List[int], List[float]]:
        """
        使用肘部法则找到最优的K值
        """
        if self.vectors is None:
            st.error("❌ 请先加载数据")
            return [], []
        
        try:
            st.subheader("🔍 K-means 最优K值搜索中...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            result_table = st.empty()
            
            k_range = range(2, min(max_k + 1, len(self.vectors)))
            total = len(k_range)
            current = 0
            
            all_results = []
            best_score = -np.inf
            best_k = None
            
            for k in k_range:
                current += 1
                progress = current / total
                status_text.text(f"正在搜索: K={k} ({current}/{total})")
                progress_bar.progress(progress)
                
                try:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(self.vectors)
                    
                    inertia = kmeans.inertia_
                    silhouette = silhouette_score(self.vectors, labels)
                    
                    result = {
                        "K": k,
                        "inertia": round(inertia, 2),
                        "silhouette_score": round(silhouette, 4),
                    }
                    all_results.append(result)
                    
                    if silhouette > best_score:
                        best_score = silhouette
                        best_k = k
                        st.success(f"✨ 新最优K值: K={k} | silhouette={silhouette:.4f}")
                    
                    # 每次更新表格
                    df = pd.DataFrame(all_results).sort_values(
                        "silhouette_score", ascending=False, ignore_index=True
                    )
                    result_table.dataframe(df)
                    
                except Exception as e:
                    st.warning(f"⚠️ K={k} 测试失败: ({e})")
                    continue
            
            progress_bar.progress(1.0)
            status_text.text("✅ 搜索完成!")
            
            if not all_results:
                st.error("❌ 未找到合适的K值,请检查数据。")
                return [], []
            
            st.success(f"🎯 最佳K值: {best_k}")
            st.metric("最佳轮廓系数", f"{best_score:.4f}")
            
            # 提取数据用于返回
            k_values = [r["K"] for r in all_results]
            silhouette_scores = [r["silhouette_score"] for r in all_results]
            
            return k_values, silhouette_scores        
            
        except Exception as e:
            st.error(f"❌ 寻找最优K值失败: {e}")
            return [], []
    
    def export_cluster_results(self) -> pd.DataFrame:
        """
        导出聚类结果
        """
        if self.cluster_labels is None:
            st.error("❌ 请先进行聚类")
            return pd.DataFrame()
        
        try:
            results_df = pd.DataFrame({
                'text': self.texts,
                'cluster': self.cluster_labels,
                'cluster_name': ['噪声点' if c == -1 else f'簇 {c}' for c in self.cluster_labels],
                'metadata': [str(meta) for meta in self.metadata]
            })
            
            if self.reduced_vectors is not None:
                results_df['umap_x'] = self.reduced_vectors[:, 0]
                results_df['umap_y'] = self.reduced_vectors[:, 1]
                if self.reduced_vectors.shape[1] >= 3:
                    results_df['umap_z'] = self.reduced_vectors[:, 2]
            
            st.success(f"✅ 成功导出 {len(results_df)} 条聚类结果")
            return results_df
            
        except Exception as e:
            st.error(f"❌ 导出结果失败: {e}")
            return pd.DataFrame()
    
    def compare_clustering_methods(self, k_values: List[int] = [5, 8, 10]) -> pd.DataFrame:
        """
        比较不同聚类方法的效果
        """
        if self.vectors is None:
            st.error("❌ 请先加载数据")
            return pd.DataFrame()
        
        results = []
        st.info(" 正在比较不同聚类方法...")
        
        # 测试K-means
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.vectors)
            score = silhouette_score(self.vectors, labels)
            results.append({
                'method': f'K-means (k={k})',
                'n_clusters': k,
                'silhouette_score': score
            })
        
        # 测试DBSCAN（如果已降维）
        if self.reduced_vectors is not None:
            for eps in [0.2, 0.3, 0.5]:
                dbscan = DBSCAN(eps=eps, min_samples=5, metric='euclidean')
                labels = dbscan.fit_predict(self.reduced_vectors)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                if n_clusters > 1:
                    try:
                        score = silhouette_score(self.reduced_vectors, labels)
                    except:
                        score = -1
                else:
                    score = -1
                results.append({
                    'method': f'DBSCAN (eps={eps})',
                    'n_clusters': n_clusters,
                    'silhouette_score': score
                })
        
        results_df = pd.DataFrame(results).sort_values('silhouette_score', ascending=False)
        st.dataframe(results_df, use_container_width=True)
        
        return results_df
