import streamlit as st
from components.milvus_mongo_insert import get_milvus_collection, get_mongo_collection
from components.utils import milvus_mongo_semantic_search
import numpy as np


def search_page():
    """文本搜索页面 - 修复版本"""
    st.markdown("## 🔍 文本搜索")
    
    # 🔧 第一步：检查模型是否已加载
    if not st.session_state.get('model_loaded', False):
        st.warning("⚠️ 尚未加载嵌入模型！")
        st.info("🔥 请先到 '🔥 嵌入模型管理' 页面加载模型，然后再进行搜索。")
        return
    
    # 🔧 第二步：检查 MongoDB 连接状态
    if not st.session_state.get("mongodb_connected", False):
        st.error("❌ MongoDB 未连接")
        st.info("📌 请先到 '🍃 MongoDB配置管理' 页面配置并连接 MongoDB")
        
        # 显示配置按钮
        if st.button("🔗 前往 MongoDB 配置", type="primary"):
            # 这里可以添加页面跳转逻辑
            st.info("请在左侧菜单选择 '🍃 MongoDB配置管理'")
        return
    
    # 🔧 第三步：检查 Milvus 连接状态
    milvus_manager = st.session_state['components'].get('milvus_manager')
    if not milvus_manager or not milvus_manager.is_connected:
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
    col1, col2 = st.columns(2)
    with col1:
        top_k = st.slider("返回结果数量", 1, 50, 10, help="设置返回的搜索结果数量")
    with col2:
        similarity_threshold = st.slider("相似度阈值", 0.0, 1.0, 0.0, 0.1, help="过滤低相似度的结果")

    # 执行搜索
    if search_button and query:
        with st.spinner("🔍 正在搜索相关内容..."):
            try:
                results = milvus_mongo_semantic_search(
                    query, 
                    top_k, 
                    milvus_collection, 
                    mongo_col, 
                    vector_processor
                )
                
                # 过滤结果
                filtered_results = [r for r in results if r['score'] >= similarity_threshold]
                
                if filtered_results:
                    st.success(f"✅ 找到 {len(filtered_results)} 个相关结果")
                    
                    # 显示搜索统计
                    stats = {
                        "total_results": len(filtered_results),
                        "avg_score": np.mean([r['score'] for r in filtered_results]) if filtered_results else 0,
                        "max_score": np.max([r['score'] for r in filtered_results]) if filtered_results else 0,
                        "min_score": np.min([r['score'] for r in filtered_results]) if filtered_results else 0,
                    }
                    
                    st.markdown("### 📊 搜索统计")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("结果数量", stats.get('total_results', 0))
                    with col2:
                        st.metric("平均相似度", f"{stats.get('avg_score', 0):.3f}")
                    with col3:
                        st.metric("最高相似度", f"{stats.get('max_score', 0):.3f}")
                    with col4:
                        st.metric("最低相似度", f"{stats.get('min_score', 0):.3f}")

                    # 显示搜索结果
                    st.markdown("### 📋 搜索结果")
                    for i, result in enumerate(filtered_results):
                        similarity_pct = result['score'] * 100
                        if similarity_pct >= 80:
                            color = "#28a745"  # 绿色
                        elif similarity_pct >= 60:
                            color = "#ffc107"  # 黄色
                        else:
                            color = "#dc3545"  # 红色
                            
                        with st.expander(f"📄 结果 {i+1} - 相似度: {similarity_pct:.1f}%", expanded=i < 3):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown("**📝 文本内容:**")
                                st.write(result['text'])
                                if result.get('metadata'):
                                    st.markdown("**📋 元数据:**")
                                    st.json(result['metadata'])
                            with col2:
                                st.markdown(f"""
                                <div style="text-align: center; padding: 1rem; background: {color}20; border-radius: 8px; border: 2px solid {color};">
                                    <h3 style="color: {color}; margin: 0;">{similarity_pct:.1f}%</h3>
                                    <p style="margin: 0; color: {color};">相似度</p>
                                </div>
                                """, unsafe_allow_html=True)
                else:
                    st.info("ℹ️ 未找到满足条件的结果，请尝试：")
                    st.markdown("""
                    - 降低相似度阈值
                    - 使用不同的关键词
                    - 检查输入的查询内容
                    """)
            except Exception as e:
                st.error(f"❌ 搜索失败: {e}")
                with st.expander("🔍 查看详细错误"):
                    st.exception(e)
