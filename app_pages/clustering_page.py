import streamlit as st

def clustering_page():
    st.markdown("## 🎯 聚类分析")
    
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
                    k_range, silhouette_scores = st.session_state.components['clustering_analyzer'].find_optimal_k()
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
                            showlegend=False
                        )
                        st.plotly_chart(fig)
                        
                        optimal_k = k_range[np.argmax(silhouette_scores)]
                        max_score = max(silhouette_scores)
                        st.success(f"🎯 建议的最优K值: {optimal_k} (轮廓系数: {max_score:.3f})")
    
    else:  # DBSCAN
        col1, col2 = st.columns(2)
        with col1:
            eps = st.slider("邻域半径 (eps)", 0.1, 2.0, 0.5, 0.1, help="定义邻域的半径大小")
        with col2:
            min_samples = st.slider("最小样本数", 2, 20, 5, help="形成聚类所需的最小样本数")
    
    # 执行聚类
    st.markdown("### 🚀 开始聚类")
    
    if st.button("🎯 执行聚类分析", type="primary"):
        with st.spinner("正在进行聚类分析..."):
            try:
                if clustering_method == "K-means聚类":
                    labels = st.session_state.components['clustering_analyzer'].perform_kmeans_clustering(n_clusters)
                else:
                    labels = st.session_state.components['clustering_analyzer'].perform_dbscan_clustering(eps, min_samples)
                
                if len(labels) > 0:
                    # 降维可视化
                    st.markdown("### 📊 聚类可视化")
                    with st.spinner("正在生成可视化图表..."):
                        reduced_vectors = st.session_state.components['clustering_analyzer'].reduce_dimensions()
                        if reduced_vectors.size > 0:
                            fig = st.session_state.components['clustering_analyzer'].create_cluster_visualization()
                            st.plotly_chart(fig)
                    
                    # 聚类摘要
                    st.markdown("### 📋 聚类摘要")
                    cluster_summary = st.session_state.components['clustering_analyzer'].get_cluster_summary()
                    
                    # 显示聚类统计
                    n_clusters_found = len(cluster_summary)
                    n_noise = cluster_summary.get('-1', {}).get('size', 0) if '-1' in cluster_summary else 0
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("发现聚类数", n_clusters_found - (1 if n_noise > 0 else 0))
                    with col2:
                        st.metric("噪声点数", n_noise)
                    with col3:
                        st.metric("聚类覆盖率", f"{((len(labels) - n_noise) / len(labels) * 100):.1f}%")
                    
                    # 显示每个聚类的详细信息
                    for cluster_id, info in cluster_summary.items():
                        if cluster_id == '-1':
                            title = f"🔹 噪声点 ({info['size']} 个样本, {info['percentage']:.1f}%)"
                        else:
                            title = f"🎯 聚类 {cluster_id} ({info['size']} 个样本, {info['percentage']:.1f}%)"
                        
                        with st.expander(title):
                            st.markdown("**📝 样本文本:**")
                            for j, text in enumerate(info['sample_texts']):
                                st.write(f"{j+1}. {text}")
                        
            except Exception as e:
                st.error(f"❌ 聚类分析失败: {e}")
                st.exception(e)