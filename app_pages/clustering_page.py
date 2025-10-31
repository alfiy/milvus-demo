import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np  # 🔧 添加这一行

def clustering_page():
    st.markdown("## 🔍 聚类分析")
    
    # 检查Milvus连接和实体数量是否大于0
    milvus_manager = st.session_state.components['milvus_manager']
    if not milvus_manager.is_connected:
        st.error("❌ 未连接到 Milvus 数据库，请检查连接配置。")
        return

    persistence_status = milvus_manager.verify_data_persistence()
    milvus_count = persistence_status.get('num_entities', 0)

    if milvus_count == 0:
        st.warning("⚠️ Milvus 数据库中无可用数据，请先完成数据上传与持久化。")
        return

    
    # 聚类方法选择
    st.markdown("### ⚙️ 聚类设置")
    
    clustering_method = st.selectbox(
        "选择聚类算法",
        ["K-means聚类", "DBSCAN聚类"],
        help="K-means适用于球形聚类，DBSCAN适用于任意形状的聚类"
    )
    
    # 聚类参数设置
    if clustering_method == "K-means聚类":
        col1, col2 = st.columns(2)
        with col1:
            n_clusters = st.slider("聚类数量 (K)", 2, 20, 8, help="设置要分成多少个聚类")
        with col2:
            if st.button("🔍 寻找最优K值", help="使用轮廓系数寻找最佳聚类数"):
                with st.spinner("正在分析最优K值..."):
                    # 先加载数据
                    try:
                        vectors, texts, metadata = milvus_manager.get_all_vectors_and_metadata()
                        if vectors is None or len(vectors) == 0:
                            st.error("❌ 未查询到数据，请先上传数据。")
                        else:
                            clustering_analyzer = st.session_state.components['clustering_analyzer']
                            clustering_analyzer.load_data(vectors, texts, metadata)
                            
                            k_range, silhouette_scores = clustering_analyzer.find_optimal_k()
                            if k_range and silhouette_scores:
                                fig = px.line(
                                    x=k_range, 
                                    y=silhouette_scores,
                                    title="轮廓系数 vs K值",
                                    labels={'x': 'K值', 'y': '轮廓系数'},
                                    markers=True
                                )
                                fig.update_layout(
                                    xaxis_title="K值",
                                    yaxis_title="轮廓系数",
                                    showlegend=False,
                                    width=700,
                                    height=400
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                optimal_k = k_range[np.argmax(silhouette_scores)]
                                max_score = max(silhouette_scores)
                                st.success(f"✅ 建议的最优K值: {optimal_k} (轮廓系数: {max_score:.3f})")
                    except Exception as e:
                        st.error(f"❌ 寻找最优K值失败: {e}")
                        st.exception(e)
    
    else:  # DBSCAN
        st.markdown("#### DBSCAN 参数设置")
        
        # 添加降维选项
        use_dimension_reduction = st.checkbox(
            "📉 先进行降维（推荐）", 
            value=True,
            help="降维可以显著提高DBSCAN的聚类效果"
        )
        
        if use_dimension_reduction:
            n_components = st.slider("降维目标维度", 2, 10, 3, help="推荐2-3维")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            eps = st.slider("邻域半径 (eps)", 0.01, 2.0, 0.3, 0.01, help="定义邻域的半径大小，值越小越严格")
        with col2:
            min_samples = st.slider("最小样本数", 2, 20, 5, help="形成聚类所需的最小样本数")
        with col3:
            if st.button("🔍 自动搜索参数", help="自动寻找最优的eps和min_samples"):
                with st.spinner("正在搜索最优参数..."):
                    try:
                        vectors, texts, metadata = milvus_manager.get_all_vectors_and_metadata()
                        if vectors is None or len(vectors) == 0:
                            st.error("❌ 未查询到数据，请先上传数据。")
                        else:
                            clustering_analyzer = st.session_state.components['clustering_analyzer']
                            clustering_analyzer.load_data(vectors, texts, metadata)
                            
                            # 如果需要降维
                            if use_dimension_reduction:
                                clustering_analyzer.reduce_dimensions(n_components=n_components)
                            
                            best_params = clustering_analyzer.find_optimal_dbscan_params()
                            if best_params:
                                st.session_state['suggested_eps'] = best_params.get('eps', eps)
                                st.session_state['suggested_min_samples'] = best_params.get('min_samples', min_samples)
                    except Exception as e:
                        st.error(f"❌ 参数搜索失败: {e}")
    
    # 执行聚类
    st.markdown("### 🚀 开始聚类")

    if st.button("▶️ 执行聚类分析", type="primary"):
        with st.spinner("正在进行聚类分析..."):
            try:
                # Step 1. 从 Milvus 数据库获得所有向量和文本、元数据
                vectors, texts, metadata = milvus_manager.get_all_vectors_and_metadata()

                # Step 2. 判空和异常处理
                if vectors is None or len(vectors) == 0:
                    st.error("❌ 未查询到聚类分析所需的数据，请先上传并持久化。")
                    return
                if texts is None or len(texts) != len(vectors):
                    st.warning("⚠️ 部分文本或元数据缺失，将使用占位数据。")
                    texts = [f"文本_{i}" for i in range(len(vectors))]
                if metadata is None or len(metadata) != len(vectors):
                    metadata = [{}] * len(vectors)

                # Step 3. 加载数据到 clustering_analyzer
                clustering_analyzer = st.session_state.components['clustering_analyzer']
                clustering_analyzer.load_data(vectors, texts, metadata)

                # Step 4. 执行聚类
                if clustering_method == "K-means聚类":
                    labels = clustering_analyzer.perform_kmeans_clustering(n_clusters)
                else:
                    # DBSCAN: 先降维（如果需要）
                    if use_dimension_reduction:
                        clustering_analyzer.reduce_dimensions(n_components=n_components)
                    labels = clustering_analyzer.perform_dbscan_clustering(eps, min_samples)
                
                if len(labels) > 0:
                    # 可视化
                    st.markdown("### 📊 聚类可视化")
                    with st.spinner("正在生成可视化图表..."):
                        # 如果还没降维，现在降维用于可视化
                        if clustering_analyzer.reduced_vectors is None:
                            reduced_vectors = clustering_analyzer.reduce_dimensions(n_components=2)
                        
                        if clustering_analyzer.reduced_vectors is not None and clustering_analyzer.reduced_vectors.size > 0:
                            # 选择2D或3D可视化
                            use_3d = clustering_analyzer.reduced_vectors.shape[1] >= 3
                            fig = clustering_analyzer.create_cluster_visualization(use_3d=use_3d)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # 聚类摘要
                    st.markdown("### 📋 聚类摘要")
                    cluster_summary = clustering_analyzer.get_cluster_summary()
                    
                    # 统计信息
                    n_clusters_found = len([k for k in cluster_summary.keys() if k != '噪声点'])
                    n_noise = cluster_summary.get('噪声点', {}).get('size', 0)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("发现聚类数", n_clusters_found)
                    with col2:
                        st.metric("噪声点数", n_noise)
                    with col3:
                        coverage = ((len(labels) - n_noise) / len(labels) * 100) if len(labels) > 0 else 0
                        st.metric("聚类覆盖率", f"{coverage:.1f}%")

                    # 详细信息
                    for cluster_id, info in cluster_summary.items():
                        if cluster_id == '噪声点':
                            title = f"🔹 噪声点 ({info['size']} 个样本, {info['percentage']:.1f}%)"
                            icon = "🔹"
                        else:
                            title = f"📦 {cluster_id} ({info['size']} 个样本, {info['percentage']:.1f}%)"
                            icon = "📦"
                        
                        with st.expander(title):
                            st.markdown("**📝 样本文本:**")
                            for j, text in enumerate(info['sample_texts'], 1):
                                st.write(f"{j}. {text[:200]}{'...' if len(text) > 200 else ''}")
                    
                    # 导出结果
                    st.markdown("### 💾 导出结果")
                    export_df = clustering_analyzer.export_cluster_results()
                    if not export_df.empty:
                        csv = export_df.to_csv(index=False, encoding='utf-8-sig')
                        st.download_button(
                            label="📥 下载聚类结果 (CSV)",
                            data=csv,
                            file_name="cluster_results.csv",
                            mime="text/csv"
                        )
                else:
                    st.error("❌ 聚类未产生结果，请检查数据。")
                    
            except Exception as e:
                st.error(f"❌ 聚类分析失败: {e}")
                st.exception(e)
