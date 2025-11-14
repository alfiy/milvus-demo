import streamlit as st
from components.milvus_mongo_insert import get_milvus_collection, get_mongo_collection
from components.utils import vector_search, VectorSearchError
import numpy as np

def safe_get(key, default=None):
    return st.session_state['components'].get(key, default) if 'components' in st.session_state else default

def get_dependencies():
    return {
        "model_loaded": st.session_state.get('model_loaded', False),
        "mongodb_connected": st.session_state.get('mongodb_connected', False),
        "milvus_manager": safe_get('milvus_manager'),
        "vector_processor": safe_get('vector_processor'),
    }

def search_page():
    """文本搜索页面 - 修复版本"""
    st.markdown("## 🔍 文本搜索")

    deps = get_dependencies()
    
    # 🔧 第一步：检查模型是否已加载
    if not deps["model_loaded"]:
        st.warning("⚠️ 尚未加载嵌入模型！")
        st.info("🔥 请先到 '🔥 嵌入模型管理' 页面加载模型，然后再进行搜索。")
        return
    
    # 🔧 第二步：检查 MongoDB 连接状态
    if not deps["mongodb_connected"]:
        st.error("❌ MongoDB 未连接")
        st.info("📌 请先到 '🍃 MongoDB配置管理' 页面配置并连接 MongoDB")
        
        # 显示配置按钮
        if st.button("🔗 前往 MongoDB 配置", type="primary"):
            st.info("请在左侧菜单选择 '🍃 MongoDB配置管理'")
        return
    
    # 🔧 第三步：检查 Milvus 连接状态
    if not deps["milvus_manager"] or not deps["milvus_manager"].is_connected:
        st.error("❌ Milvus 未连接")
        st.info("📌 请先到 '🗄️ Milvus数据库管理' 页面配置并连接 Milvus")
        return
    
    # 🔧 第四步：安全地初始化搜索组件
    try:
        # 获取向量维度
        vectors = st.session_state.get('vectors')
        if vectors is not None and vectors.size > 0:
            dim = vectors.shape[1]
        else:
            # 如果没有向量数据，使用模型的默认维度
            vp = st.session_state['components'].get('vector_processor')
            model_info = vp.get_model_info() if vp else {}
            dim = model_info.get('dimension', 384)
        
        # 获取 Milvus 集合
        milvus_collection = get_milvus_collection(
            collection_name="text_vectors",
            dim=dim
        )
        
        if milvus_collection is None:
            st.error("❌ Milvus 集合未初始化")
            st.info("📌 请先到 '📊 数据上传与处理' 页面上传数据")
            return
        
        # 获取 MongoDB 集合
        mongo_col = get_mongo_collection()
        
        if mongo_col is None:
            st.error("❌ MongoDB 集合获取失败")
            return
        
        # 获取向量处理器
        vector_processor = st.session_state.components["vector_processor"]


    except Exception as e:
        st.error(f"❌ 初始化搜索组件失败: {e}")
        st.info("📌 请确保 Milvus 和 MongoDB 都已正确配置和连接")
        
        # 显示详细的错误信息
        with st.expander("🔍 查看详细错误信息"):
            st.exception(e)
        return

    # 🔧 第五步：搜索界面（只有在所有组件都准备好后才显示）
    st.markdown("### 🔍 搜索查询")
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(
            "输入搜索查询",
            placeholder="例如：描述春天的诗句",
            help="输入您想要搜索的文本内容，系统会找到语义相似的文本"
        )
    with col2:
        st.write("")  # 占位
        search_button = st.button("🔍 开始搜索", type="primary")

    # 搜索参数
    col1, col2, col3 = st.columns(3)
    with col1:
        top_k = st.slider("返回结果数量", 1, 50, 10, help="设置返回的搜索结果数量")
    with col2:
        similarity_threshold = st.slider(
            "相似度阈值", 
            0.0, 1.0, 0.0, 0.1, 
            help="过滤低相似度的结果，0表示不过滤"
        )
    with col3:
        enable_stats = st.checkbox("显示性能统计", value=False, help="显示搜索耗时等统计信息")

    # 执行搜索
    if search_button and query:
        with st.spinner("🔍 正在搜索相关内容..."):
            try:
                # 调用优化后的vector_search函数
                results = vector_search(
                    query=query,
                    top_k=top_k,
                    milvus_collection=milvus_collection,
                    mongo_col=mongo_col,
                    vector_processor=vector_processor,
                    filter_mode="similarity",
                    filter_threshold=similarity_threshold,
                    output_fields=["text", "metadata"],
                    enable_stats=enable_stats
                )

                results = sorted(results, key=lambda x: x.get("score", 0), reverse=True)
                
                # 提取搜索统计（如果启用）
                search_stats = None
                if enable_stats and results and "_search_stats" in results[0]:
                    search_stats = results[0].pop("_search_stats")  # 移除统计信息，避免显示在结果中
                
                if results:
                    st.success(f"✅ 找到 {len(results)} 个相关结果")
                    
                    # 显示性能统计
                    if search_stats:
                        st.markdown("### ⚡ 性能统计")
                        cols = st.columns(5)
                        with cols[0]:
                            st.metric("总耗时", f"{search_stats.get('total_time', 0):.3f}秒")
                        with cols[1]:
                            st.metric("向量化", f"{search_stats.get('encode_time', 0):.3f}秒")
                        with cols[2]:
                            st.metric("Milvus搜索", f"{search_stats.get('milvus_time', 0):.3f}秒")
                        with cols[3]:
                            st.metric("MongoDB查询", f"{search_stats.get('mongo_time', 0):.3f}秒")
                        with cols[4]:
                            st.metric("缺失记录", search_stats.get('mongo_missing', 0))
                    
                    # 计算结果统计
                    scores = [r['score'] for r in results]
                    result_stats = {
                        "total_results": len(results),
                        "avg_score": np.mean(scores) if scores else 0,
                        "max_score": np.max(scores) if scores else 0,
                        "min_score": np.min(scores) if scores else 0,
                    }
                    
                    st.markdown("### 📊 搜索统计")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("结果数量", result_stats['total_results'])
                    with col2:
                        st.metric("平均相似度", f"{result_stats['avg_score']:.3f}")
                    with col3:
                        st.metric("最高相似度", f"{result_stats['max_score']:.3f}")
                    with col4:
                        st.metric("最低相似度", f"{result_stats['min_score']:.3f}")

                    # 显示搜索结果
                    st.markdown("### 📋 搜索结果")
                    for i, result in enumerate(results):
                        similarity_pct = result['score'] * 100
                        
                        # 根据相似度设置颜色
                        if similarity_pct >= 80:
                            color = "#28a745"  # 绿色
                            badge = "🟢 高度相关"
                        elif similarity_pct >= 60:
                            color = "#ffc107"  # 黄色
                            badge = "🟡 中度相关"
                        else:
                            color = "#dc3545"  # 红色
                            badge = "🔴 低度相关"
                            
                        with st.expander(
                            f"📄 结果 {i+1} - {badge} - 相似度: {similarity_pct:.1f}%", 
                            expanded=(i < 3)  # 默认展开前3个结果
                        ):
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.markdown("**📝 文本内容:**")
                                # 截断过长的文本
                                text = result.get('text', '')
                                if len(text) > 500:
                                    st.write(text[:500] + "...")
                                    with st.expander("查看完整文本"):
                                        st.write(text)
                                else:
                                    st.write(text if text else "❌ 无文本内容")
                                
                                # 显示元数据
                                metadata = result.get('metadata', {})
                                if metadata and not metadata.get('_missing'):
                                    st.markdown("**📋 元数据:**")
                                    st.json(metadata)
                                elif metadata.get('_missing'):
                                    st.warning("⚠️ MongoDB中未找到该记录的元数据")
                            
                            with col2:
                                st.markdown(f"""
                                <div style="text-align: center; padding: 1rem; background: {color}20; 
                                     border-radius: 8px; border: 2px solid {color};">
                                    <h3 style="color: {color}; margin: 0;">{similarity_pct:.1f}%</h3>
                                    <p style="margin: 0; color: {color};">相似度</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # 显示记录ID
                                st.markdown(f"""
                                <div style="margin-top: 1rem; padding: 0.5rem; background: #f0f0f0; 
                                     border-radius: 4px; font-size: 0.8em;">
                                    <strong>ID:</strong><br>{result['id'][:16]}...
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # 导出结果功能
                    st.markdown("---")
                    st.markdown("### 💾 导出结果")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # 导出为JSON
                        import json
                        json_str = json.dumps(results, ensure_ascii=False, indent=2)
                        st.download_button(
                            label="📥 导出为 JSON",
                            data=json_str,
                            file_name=f"search_results_{query[:20]}.json",
                            mime="application/json"
                        )
                    
                    with col2:
                        # 导出为CSV
                        import pandas as pd
                        df_data = []
                        for r in results:
                            df_data.append({
                                "ID": r['id'],
                                "相似度": f"{r['score']:.4f}",
                                "文本": r['text'][:100] + "..." if len(r['text']) > 100 else r['text'],
                                "元数据": str(r.get('metadata', {}))
                            })
                        df = pd.DataFrame(df_data)
                        csv = df.to_csv(index=False).encode('utf-8-sig')
                        st.download_button(
                            label="📥 导出为 CSV",
                            data=csv,
                            file_name=f"search_results_{query[:20]}.csv",
                            mime="text/csv"
                        )
                
                else:
                    st.info("ℹ️ 未找到满足条件的结果，请尝试：")
                    st.markdown("""
                    - 🔻 降低相似度阈值（当前: {:.1f}）
                    - 🔄 使用不同的关键词
                    - 📝 检查输入的查询内容
                    - 📊 确保数据库中有相关数据
                    """.format(similarity_threshold))
                    
            except VectorSearchError as e:
                st.error(f"❌ 搜索失败: {str(e)}")
                st.info("💡 建议检查：")
                st.markdown("""
                - Milvus 和 MongoDB 连接是否正常
                - 集合中是否有数据
                - 向量维度是否匹配
                """)
                with st.expander("🔍 查看详细错误"):
                    st.exception(e)
                    
            except Exception as e:
                st.error(f"❌ 未知错误: {str(e)}")
                with st.expander("🔍 查看详细错误"):
                    st.exception(e)
    
    elif search_button and not query:
        st.warning("⚠️ 请输入搜索查询内容")
