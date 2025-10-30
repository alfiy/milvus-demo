import streamlit as st
from components.config_manager import config_manager
from components.vector_processor import VectorProcessor

if "components" not in st.session_state:
    st.session_state["components"] = {}

def home_page():
    st.markdown("## 🏠 系统概览")

    # 保证所有关键变量初始化风格一致
    milvus_config = st.session_state.get('milvus_config', {})
    mongodb_config = st.session_state.get('mongodb_config', {})
    model_config = st.session_state.get('model_config', {})
    mongo_data = st.session_state.get('mongo_data', {})
    current_config = st.session_state.get('current_config', {})

    # 基础配置卡片
    st.markdown("### ⚙️ 配置状态")

    col1, col2, col3 = st.columns(3)
    with col1:
        milvus_status = "✅ 已配置" if milvus_config.get("host") else "❌ 未配置"
        auto_connect = "🔄 自动连接" if milvus_config.get("auto_connect", False) else "⚠️ 手动连接"
        st.markdown(f"""
        <div class="metric-card">
            <h3>🗄️ Milvus</h3>
            <h2>{milvus_status}</h2>
            <p>{auto_connect}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        mongodb_status = "✅ 已配置" if mongodb_config.get("host") else "❌ 未配置"
        mongo_auto = "🔄 自动连接" if mongodb_config.get("auto_connect", False) else "⚠️ 手动连接"
        st.markdown(f"""
        <div class="metric-card">
            <h3>🍃 MongoDB</h3>
            <h2>{mongodb_status}</h2>
            <p>{mongo_auto}</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        model_status = "✅ 已加载" if st.session_state.get('model_loaded', False) else "❌ 未加载"
        model_auto = "🔄 自动加载" if model_config.get("auto_load", False) else "⚠️ 手动加载"
        st.markdown(f"""
        <div class="metric-card">
            <h3>🤖 模型</h3>
            <h2>{model_status}</h2>
            <p>{model_auto}</p>
        </div>
        """, unsafe_allow_html=True)

    # 连接状态
    st.markdown("### 🔗 连接状态")
    mongodb_config = st.session_state.get("mongodb_config", {})  # 拿到配置（用于展示）

    col1, col2 = st.columns(2)
    # MongoDB连接状态
    with col1:
        if st.session_state.get('mongodb_connected', False):
            st.markdown(f"""
            <div class="persistence-status status-success">
                <h4>✅ MongoDB连接正常</h4>
                <p>已连接到 <strong>{mongodb_config.get('host', '')}:{mongodb_config.get('port', '')}</strong></p>
                <p>数据库: {mongodb_config.get('db_name', '')}, 集合: {mongodb_config.get('col_name', '')}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            error_info = st.session_state.get('mongodb_connect_error')
            st.markdown(f"""
            <div class="persistence-status status-warning">
                <h4>⚠️ MongoDB未连接</h4>
                <p>请到 ' MongoDB配置管理' 页面配置连接</p>
                {"<p style='color:red'>" + error_info + "</p>" if error_info else ""}
            </div>
            """, unsafe_allow_html=True)

    # Milvus连接状态
    with col2:
        milvus_manager = st.session_state['components'].get('milvus_manager')
        if milvus_manager and milvus_manager.is_connected:
            persistence_status = milvus_manager.verify_data_persistence()
            if persistence_status['status'] == 'success':
                st.markdown(f"""
                <div class="persistence-status status-success">
                    <h4>✅ Milvus数据库正常</h4>
                    <p>已保存 <strong>{persistence_status['num_entities']:,}</strong> 条记录</p>
                    <p>配置已保存，重启后自动恢复</p>
                </div>
                """, unsafe_allow_html=True)
            elif persistence_status['status'] == 'no_collection':
                st.markdown("""
                <div class="persistence-status status-warning">
                    <h4>⚠️ Milvus已连接，暂无数据</h4>
                    <p>数据库已连接，但尚未创建数据集合</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="persistence-status status-error">
                    <h4>❌ Milvus数据状态异常</h4>
                    <p>{persistence_status['message']}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="persistence-status status-warning">
                <h4>🗄️ Milvus未连接</h4>
                <p>请到 '🗄️ Milvus数据库管理' 页面配置连接</p>
            </div>
            """, unsafe_allow_html=True)

    # 系统状态卡片
    st.markdown("### 📊 系统状态")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if mongo_data.get("connected", False):
            data_count = len(mongo_data.get('texts', []))
            status_text = "数据库记录数量"
        else:
            data_count = 0
            status_text = "连接失败" if mongo_data.get("error") else "未连接"
        st.markdown(f"""
        <div class="metric-card">
            <h3> MongoDB数据</h3>
            <h2>{data_count}</h2>
            <p>{status_text}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        vectors_val = st.session_state.get('vectors')
        vector_size = vectors_val.nbytes / 1024 / 1024 if vectors_val is not None else 0
        st.markdown(f"""
        <div class="metric-card">
            <h3>💾 内存占用</h3>
            <h2>{vector_size:.1f} MB</h2>
            <p>向量数据大小</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        milvus_count = 0
        if milvus_manager and milvus_manager.is_connected:
            persistence_status = milvus_manager.verify_data_persistence()
            milvus_count = persistence_status.get('num_entities', 0)
        status_color = "#28a745" if milvus_count > 0 else "#dc3545"
        st.markdown(f"""
        <div class="metric-card">
            <h3>🗄️ 持久化数据</h3>
            <h2 style="color: {status_color}">{milvus_count:,}</h2>
            <p>Milvus中的记录</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        vector_processor = st.session_state['components'].get('vector_processor')
        model_info = vector_processor.get_model_info() if vector_processor else {}
        embedding_dim = model_info.get('dimension', 'N/A') if st.session_state.get('model_loaded', False) else 'N/A'
        st.markdown(f"""
        <div class="metric-card">
            <h3>🔢 向量维度</h3>
            <h2>{embedding_dim}</h2>
            <p>模型输出维度</p>
        </div>
        """, unsafe_allow_html=True)

    
    st.markdown("---")
    
    # 功能介绍
    st.markdown("## 🚀 主要功能")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>📊 数据处理</h4>
            <ul>
                <li>支持JSON/JSONL格式数据上传</li>
                <li>自动文本向量化处理</li>
                <li>支持大规模数据处理（38万条+）</li>
                <li>多语言文本支持</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4>🔍 智能搜索</h4>
            <ul>
                <li>语义相似度搜索</li>
                <li>本地向量搜索</li>
                <li>Milvus数据库搜索</li>
                <li>批量搜索功能</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>🗄️ 数据管理</h4>
            <ul>
                <li><strong>Milvus向量数据库集成</strong></li>
                <li><strong>MongoDB元数据存储</strong></li>
                <li><strong>配置自动保存和恢复</strong></li>
                <li>高效向量存储和检索</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4>🎯 聚类分析</h4>
            <ul>
                <li>K-means聚类算法</li>
                <li>DBSCAN密度聚类</li>
                <li>UMAP降维可视化</li>
                <li>聚类结果分析</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # 快速开始提示
    if not milvus_config.get("host") or not st.session_state.model_loaded or not mongodb_config.get("host"):
        st.markdown("---")
        st.markdown("## 🚀 快速开始")
        
        if not st.session_state.get('model_loaded', False):
            st.info("💡 请先到 '🤖 嵌入模型管理' 页面选择并加载向量化模型")
        
        if not milvus_config.get("host"):
            st.info("💡 请到 '🗄️ Milvus数据库管理' 页面配置数据库连接")
        
        if not mongodb_config.get("host"):
            st.info("💡 请到 '🍃 MongoDB配置管理' 页面配置元数据存储")