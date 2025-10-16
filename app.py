import streamlit as st
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from vector_processor import VectorProcessor
from milvus_manager import MilvusManager
from clustering_analyzer import ClusteringAnalyzer
from search_engine import SearchEngine

# 页面配置
st.set_page_config(
    page_title="文本向量化与Milvus数据库解决方案",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .persistence-status {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 2px solid;
    }
    .status-success {
        background-color: #d4edda;
        border-color: #28a745;
        color: #155724;
    }
    .status-warning {
        background-color: #fff3cd;
        border-color: #ffc107;
        color: #856404;
    }
    .status-error {
        background-color: #f8d7da;
        border-color: #dc3545;
        color: #721c24;
    }
    .record-item {
        background: #f8f9fa;
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 4px;
        border-left: 3px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# 初始化session state
@st.cache_resource
def init_components():
    return {
        'vector_processor': VectorProcessor(),
        'milvus_manager': MilvusManager(),
        'clustering_analyzer': ClusteringAnalyzer(),
        'search_engine': SearchEngine()
    }

def init_session_state():
    if 'components' not in st.session_state:
        st.session_state.components = init_components()
    
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'vectors' not in st.session_state:
        st.session_state.vectors = None
    if 'texts' not in st.session_state:
        st.session_state.texts = []
    if 'metadata' not in st.session_state:
        st.session_state.metadata = []
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False

def main():
    init_session_state()
    
    # 主标题
    st.markdown("""
    <div class="main-header">
        <h1>🔍 文本向量化与Milvus数据库解决方案</h1>
        <p>支持38万条数据的向量化、存储、搜索和聚类分析 - 数据持久化版本</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 侧边栏导航
    with st.sidebar:
        st.markdown("### 📋 功能菜单")
        
        # 显示当前状态
        if st.session_state.data_loaded:
            st.success(f"✅ 已加载 {len(st.session_state.texts)} 条数据")
        else:
            st.info("📁 请先上传数据")
        
        # 模型加载状态
        if st.session_state.model_loaded:
            st.success("🤖 模型已加载")
        else:
            st.warning("⚠️ 模型未加载")
        
        # Milvus连接状态和数据持久化验证
        if st.session_state.components['milvus_manager'].is_connected:
            st.success("🔗 Milvus已连接")
            
            # 验证数据持久化状态
            persistence_status = st.session_state.components['milvus_manager'].verify_data_persistence()
            if persistence_status['status'] == 'success':
                st.success(f"💾 持久化数据: {persistence_status['num_entities']:,} 条")
            elif persistence_status['status'] == 'no_collection':
                st.info("📦 暂无持久化集合")
            else:
                st.warning("⚠️ 数据状态未知")
        else:
            st.warning("⚠️ Milvus未连接")
        
        st.markdown("---")
        
        page = st.selectbox(
            "选择功能模块",
            ["🏠 首页概览", "📁 数据上传与处理", "🗄️ Milvus数据库管理", "🔍 文本搜索", "🎯 聚类分析", "ℹ️ 系统信息"],
            index=0
        )
    
    # 主要内容区域
    if page == "🏠 首页概览":
        home_page()
    elif page == "📁 数据上传与处理":
        data_upload_page()
    elif page == "🗄️ Milvus数据库管理":
        milvus_management_page()
    elif page == "🔍 文本搜索":
        search_page()
    elif page == "🎯 聚类分析":
        clustering_page()
    elif page == "ℹ️ 系统信息":
        system_info_page()

def home_page():
    st.markdown("## 🏠 系统概览")
    
    # 数据持久化状态检查
    if st.session_state.components['milvus_manager'].is_connected:
        persistence_status = st.session_state.components['milvus_manager'].verify_data_persistence()
        
        if persistence_status['status'] == 'success':
            st.markdown(f"""
            <div class="persistence-status status-success">
                <h4>✅ 数据持久化状态良好</h4>
                <p>Milvus数据库中保存了 <strong>{persistence_status['num_entities']:,}</strong> 条记录</p>
                <p>即使重启应用，数据仍然安全保存在数据库中</p>
            </div>
            """, unsafe_allow_html=True)
        elif persistence_status['status'] == 'no_collection':
            st.markdown("""
            <div class="persistence-status status-warning">
                <h4>⚠️ 暂无持久化数据</h4>
                <p>Milvus数据库已连接，但尚未创建数据集合</p>
                <p>上传数据并插入到Milvus后，数据将永久保存</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="persistence-status status-error">
                <h4>❌ 数据状态检查失败</h4>
                <p>{persistence_status['message']}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="persistence-status status-warning">
            <h4>🔌 Milvus未连接</h4>
            <p>请先连接到Milvus数据库以启用数据持久化功能</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 系统状态卡片
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 本地数据</h3>
            <h2>{}</h2>
            <p>当前加载数量</p>
        </div>
        """.format(len(st.session_state.texts) if st.session_state.data_loaded else 0), unsafe_allow_html=True)
    
    with col2:
        vector_size = 0
        if st.session_state.vectors is not None:
            vector_size = st.session_state.vectors.nbytes / 1024 / 1024
        st.markdown("""
        <div class="metric-card">
            <h3>💾 内存占用</h3>
            <h2>{:.1f} MB</h2>
            <p>向量数据大小</p>
        </div>
        """.format(vector_size), unsafe_allow_html=True)
    
    with col3:
        # Milvus持久化数据统计
        milvus_count = 0
        if st.session_state.components['milvus_manager'].is_connected:
            persistence_status = st.session_state.components['milvus_manager'].verify_data_persistence()
            milvus_count = persistence_status.get('num_entities', 0)
        
        status_color = "#28a745" if milvus_count > 0 else "#dc3545"
        st.markdown("""
        <div class="metric-card">
            <h3>🗄️ 持久化数据</h3>
            <h2 style="color: {}">{:,}</h2>
            <p>Milvus中的记录</p>
        </div>
        """.format(status_color, milvus_count), unsafe_allow_html=True)
    
    with col4:
        model_info = st.session_state.components['vector_processor'].get_model_info()
        embedding_dim = model_info.get('dimension', 'N/A')
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 向量维度</h3>
            <h2>{}</h2>
            <p>模型输出维度</p>
        </div>
        """.format(embedding_dim), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 功能介绍
    st.markdown("## 🚀 主要功能")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>📁 数据处理</h4>
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
                <li><strong>数据永久保存，重启不丢失</strong></li>
                <li><strong>支持记录删除和数据清理</strong></li>
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

def data_upload_page():
    st.markdown("## 📁 数据上传与处理")
    
    # 模型选择和加载
    st.markdown("### 🤖 模型选择与加载")
    
    # 可用模型列表
    available_models = {
        "paraphrase-multilingual-MiniLM-L12-v2": {
            "name": "多语言MiniLM-L12-v2",
            "description": "支持多语言，384维向量，平衡性能与质量",
            "dimension": 384
        },
        "all-MiniLM-L6-v2": {
            "name": "MiniLM-L6-v2",
            "description": "轻量级模型，384维向量，速度快",
            "dimension": 384
        },
        "paraphrase-MiniLM-L3-v2": {
            "name": "MiniLM-L3-v2", 
            "description": "超轻量级模型，384维向量，最快速度",
            "dimension": 384
        },
        "all-mpnet-base-v2": {
            "name": "MPNet-base-v2",
            "description": "高质量模型，768维向量，效果较佳",
            "dimension": 768
        },
        "Qwen/Qwen3-Embedding-4B": {
            "name": "Qwen3-Embedding-4B",
            "description": "高质量模型，32-2560维向量可灵活调整，应用更广",
            "dimension": 2560
        }
    }
    
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_model = st.selectbox(
            "选择向量化模型",
            options=list(available_models.keys()),
            format_func=lambda x: f"{available_models[x]['name']} - {available_models[x]['description']}",
            index=0,
            help="选择不同的模型会影响向量化质量和速度"
        )
        
        # 显示模型信息
        model_info = available_models[selected_model]
        st.info(f"📊 模型维度: {model_info['dimension']} | 描述: {model_info['description']}")
        
    
    with col2:
        st.write("")  # 占位
        st.write("")  # 占位
        if not st.session_state.model_loaded:
            if st.button("🤖 加载模型", type="primary", use_container_width=True):
                with st.spinner("正在加载模型..."):
                    if st.session_state.components['vector_processor'].load_model():
                        st.session_state.model_loaded = True
                        st.success("✅ 模型加载成功！")
                        st.rerun()
        else:
            st.success("✅ 模型已加载")
            if st.button("🔄 重新加载", use_container_width=True):
                st.session_state.components['vector_processor'].model = None
                st.session_state.model_loaded = False
                st.rerun()
    
    st.markdown("---")
    
    # 数据上传选项
    upload_method = st.radio(
        "选择数据输入方式",
        ["📤 上传JSON文件", "✏️ 手动输入JSON数据", "📋 使用示例数据"],
        horizontal=True
    )
    
    json_data = None
    
    if upload_method == "📤 上传JSON文件":
        uploaded_file = st.file_uploader(
            "选择JSON文件",
            type=['json', 'jsonl', 'txt'],
            help="支持JSON、JSONL格式文件。JSON格式：[{\"text1\":\"内容\"}]，JSONL格式：每行一个JSON对象"
        )
        
        if uploaded_file is not None:
            try:
                # 读取文件内容
                file_content = uploaded_file.read().decode('utf-8')
                
                # 使用改进的JSON解析方法
                json_data = st.session_state.components['vector_processor'].parse_json_file(file_content)
                
                if not isinstance(json_data, list):
                    json_data = [json_data]
                
                st.success(f"✅ 成功加载 {len(json_data)} 条数据")
                
                # 显示文件信息
                file_size = uploaded_file.size / 1024 / 1024
                st.info(f"📊 文件大小: {file_size:.2f} MB")
                
                # 显示数据格式检测结果
                sample_item = json_data[0] if json_data else {}
                if isinstance(sample_item, dict):
                    keys = list(sample_item.keys())
                    # 修复f-string语法错误
                    keys_display = ', '.join(keys[:5])
                    if len(keys) > 5:
                        keys_display += '...'
                    st.info(f"🔍 检测到字段: {keys_display}")
                
            except Exception as e:
                st.error(f"❌ 文件加载失败: {e}")
                st.markdown("""
                **支持的文件格式：**
                1. **标准JSON数组**: `[{"text1":"内容1"}, {"text1":"内容2"}]`
                2. **JSONL格式**: 每行一个JSON对象
                   ```
                   {"text1":"内容1"}
                   {"text1":"内容2"}
                   ```
                3. **单个JSON对象**: `{"text1":"内容"}`
                """)
    
    elif upload_method == "✏️ 手动输入JSON数据":
        json_text = st.text_area(
            "输入JSON数据",
            height=200,
            placeholder='[{"text1":"半生长以客为家，罢直初来瀚海槎。始信人间行不尽，天涯更复有天涯。"}]',
            help="请输入有效的JSON格式数据"
        )
        
        if json_text.strip():
            try:
                json_data = st.session_state.components['vector_processor'].parse_json_file(json_text)
                if not isinstance(json_data, list):
                    json_data = [json_data]
                st.success(f"✅ 成功解析 {len(json_data)} 条数据")
            except Exception as e:
                st.error(f"❌ JSON解析失败: {e}")
    
    elif upload_method == "📋 使用示例数据":
        sample_data = [
            {"text1": "半生长以客为家，罢直初来瀚海槎。始信人间行不尽，天涯更复有天涯。"},
            {"text1": "春风得意马蹄疾，一日看尽长安花。"},
            {"text1": "山重水复疑无路，柳暗花明又一村。"},
            {"text1": "海内存知己，天涯若比邻。"},
            {"text1": "落红不是无情物，化作春泥更护花。"},
            {"text1": "会当凌绝顶，一览众山小。"},
            {"text1": "采菊东篱下，悠然见南山。"},
            {"text1": "明月几时有，把酒问青天。"}
        ]
        json_data = sample_data
        st.info(f"📋 使用示例数据，共 {len(json_data)} 条古诗词")
    
    # 数据预览和处理
    if json_data:
        st.markdown("### 📊 数据预览")
        
        # 显示数据统计
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("数据条数", len(json_data))
        with col2:
            total_chars = sum(len(str(item)) for item in json_data)
            st.metric("总字符数", f"{total_chars:,}")
        with col3:
            avg_length = total_chars / len(json_data) if json_data else 0
            st.metric("平均长度", f"{avg_length:.1f}")
        
        # 数据表格预览
        df_preview = pd.DataFrame(json_data[:10])
        st.dataframe(df_preview, use_container_width=True)
        
        if len(json_data) > 10:
            st.info(f"显示前10条数据，总共{len(json_data)}条")
        
        # 向量化处理
        st.markdown("### 🚀 向量化处理")
        
        if not st.session_state.model_loaded:
            st.warning("⚠️ 请先加载向量化模型")
        else:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info("点击下方按钮开始文本向量化处理，处理后的数据可以保存到Milvus数据库中永久存储")
            with col2:
                process_button = st.button("🚀 开始向量化处理", type="primary", use_container_width=True)
            
            if process_button:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.text("🔄 正在处理文本数据...")
                    progress_bar.progress(30)
                    
                    texts, vectors, metadata = st.session_state.components['vector_processor'].process_json_data(json_data)
                    progress_bar.progress(80)
                    
                    if len(texts) > 0:
                        # 保存到session state
                        st.session_state.texts = texts
                        st.session_state.vectors = vectors
                        st.session_state.metadata = metadata
                        st.session_state.data_loaded = True
                        
                        # 设置搜索引擎数据
                        st.session_state.components['search_engine'].load_data(vectors, texts, metadata)
                        st.session_state.components['search_engine'].set_vector_processor(st.session_state.components['vector_processor'])
                        
                        # 设置聚类分析器数据
                        st.session_state.components['clustering_analyzer'].load_data(vectors, texts, metadata)
                        
                        progress_bar.progress(100)
                        status_text.text("✅ 向量化处理完成！")
                        
                        st.success(f"🎉 向量化完成！成功处理了 {len(texts)} 条文本")
                        
                        # 显示处理结果统计
                        st.markdown("### 📈 处理结果")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("文本数量", len(texts))
                        with col2:
                            st.metric("向量维度", vectors.shape[1])
                        with col3:
                            st.metric("数据大小", f"{vectors.nbytes / 1024 / 1024:.2f} MB")
                        with col4:
                            st.metric("处理状态", "✅ 完成")
                        
                        # 数据持久化提醒
                        st.markdown("### 💾 数据持久化")
                        st.info("💡 数据已在内存中处理完成。为确保数据不会在重启后丢失，请前往 '🗄️ Milvus数据库管理' 页面将数据保存到数据库中。")
                        
                        # 显示向量化样本
                        with st.expander("🔍 查看向量化样本", expanded=False):
                            sample_idx = 0
                            st.write(f"**原文本:** {texts[sample_idx]}")
                            st.write(f"**向量维度:** {len(vectors[sample_idx])}")
                            st.write(f"**向量前10维:** {vectors[sample_idx][:10].tolist()}")
                            
                    else:
                        st.error("❌ 未找到有效的文本数据")
                        
                except Exception as e:
                    st.error(f"❌ 向量化处理失败: {e}")
                    st.exception(e)
                
                finally:
                    progress_bar.empty()
                    status_text.empty()

def milvus_management_page():
    st.markdown("## 🗄️ Milvus数据库管理")
    
    # 数据持久化状态显示
    st.markdown("### 💾 数据持久化状态")
    
    if st.session_state.components['milvus_manager'].is_connected:
        persistence_status = st.session_state.components['milvus_manager'].verify_data_persistence()
        
        if persistence_status['status'] == 'success':
            st.success(f"✅ 数据库中已保存 {persistence_status['num_entities']:,} 条记录")
        elif persistence_status['status'] == 'no_collection':
            st.info("📦 数据库已连接，但尚未创建数据集合")
        else:
            st.error(f"❌ {persistence_status['message']}")
    else:
        st.warning("⚠️ 尚未连接到Milvus数据库")
    
    # 连接设置
    st.markdown("### 🔌 数据库连接")
    
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        host = st.text_input("Milvus主机地址", value="localhost", help="Milvus服务器的IP地址或域名")
    with col2:
        port = st.text_input("端口", value="19530", help="Milvus服务器端口，默认19530")
    with col3:
        user = st.text_input("用户名", value="", help="Milvus用户名（可选）", placeholder="可选")
    with col4:
        password = st.text_input("密码", value="", type="password", help="Milvus密码（可选）", placeholder="可选")
    
    # 连接按钮
    col1, col2 = st.columns([4, 1])
    with col1:
        st.info("💡 如果Milvus服务器未设置认证，用户名和密码可以留空")
    with col2:
        connect_button = st.button("🔌 连接数据库", type="primary", use_container_width=True)
    
    # 连接操作
    if connect_button:
        with st.spinner("正在连接到Milvus数据库..."):
            # 更新连接参数
            st.session_state.components['milvus_manager'].host = host
            st.session_state.components['milvus_manager'].port = port
            
            # 如果提供了用户名和密码，设置认证信息
            if user.strip() and password.strip():
                st.session_state.components['milvus_manager'].user = user
                st.session_state.components['milvus_manager'].password = password
            
            success = st.session_state.components['milvus_manager'].connect()
            if success:
                st.session_state.components['search_engine'].set_milvus_manager(st.session_state.components['milvus_manager'])
                st.rerun()
    
    # 显示连接状态
    if st.session_state.components['milvus_manager'].is_connected:
        connection_info = f"{host}:{port}"
        if hasattr(st.session_state.components['milvus_manager'], 'user') and st.session_state.components['milvus_manager'].user:
            connection_info += f" (用户: {st.session_state.components['milvus_manager'].user})"
        
        st.success(f"✅ 已成功连接到Milvus数据库 ({connection_info})")
        
        # 集合管理
        st.markdown("### 📦 集合管理")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📦 创建/连接集合", use_container_width=True):
                if st.session_state.data_loaded:
                    with st.spinner("正在创建/连接集合..."):
                        dimension = st.session_state.vectors.shape[1]
                        success = st.session_state.components['milvus_manager'].create_collection(dimension)
                        if success:
                            st.rerun()
                else:
                    st.warning("⚠️ 请先上传并处理数据")
        
        with col2:
            if st.button("📥 插入数据到Milvus", use_container_width=True):
                if st.session_state.data_loaded and st.session_state.components['milvus_manager'].collection:
                    with st.spinner("正在插入数据到Milvus..."):
                        success = st.session_state.components['milvus_manager'].insert_vectors(
                            st.session_state.texts,
                            st.session_state.vectors,
                            st.session_state.metadata
                        )
                        if success:
                            st.rerun()
                else:
                    st.warning("⚠️ 请先创建集合并加载数据")
        
        with col3:
            if st.button("🗑️ 删除集合", use_container_width=True):
                # 使用session state来管理确认状态
                if 'confirm_delete_collection' not in st.session_state:
                    st.session_state.confirm_delete_collection = False
                
                if not st.session_state.confirm_delete_collection:
                    st.session_state.confirm_delete_collection = True
                    st.rerun()
                else:
                    if st.button("⚠️ 确认删除", type="secondary"):
                        with st.spinner("正在删除集合..."):
                            st.session_state.components['milvus_manager'].delete_collection()
                            st.session_state.confirm_delete_collection = False
                            st.rerun()
        
        # 数据管理功能
        if st.session_state.components['milvus_manager'].collection:
            st.markdown("### 🗂️ 数据管理")
            
            # 获取集合统计信息
            stats = st.session_state.components['milvus_manager'].get_collection_stats()
            if stats and stats.get('num_entities', 0) > 0:
                
                # 数据管理选项卡
                tab1, tab2, tab3 = st.tabs(["📋 数据预览", "🔍 搜索删除", "🗑️ 批量删除"])
                
                with tab1:
                    st.markdown("#### 📋 数据预览")
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.info("查看数据库中的记录，可以选择特定记录进行删除")
                    with col2:
                        if st.button("🔄 刷新数据", use_container_width=True):
                            st.rerun()
                    
                    # 获取样本记录
                    sample_records = st.session_state.components['milvus_manager'].get_sample_records(20)
                    
                    if sample_records:
                        st.write(f"显示前20条记录（总共 {stats.get('num_entities', 0):,} 条）：")
                        
                        # 选择要删除的记录
                        selected_ids = []
                        for i, record in enumerate(sample_records):
                            col1, col2, col3 = st.columns([1, 6, 1])
                            
                            with col1:
                                # 修复checkbox标签问题
                                if st.checkbox("选择", key=f"select_{record['id']}", label_visibility="collapsed"):
                                    selected_ids.append(record['id'])
                            
                            with col2:
                                st.markdown(f"""
                                <div class="record-item">
                                    <strong>ID:</strong> {record['id']}<br>
                                    <strong>文本:</strong> {record['text']}<br>
                                    <strong>元数据:</strong> {json.dumps(record['metadata'], ensure_ascii=False)}
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col3:
                                if st.button("🗑️", key=f"delete_{record['id']}", help="删除此记录"):
                                    with st.spinner("正在删除记录..."):
                                        success = st.session_state.components['milvus_manager'].delete_records_by_ids([record['id']])
                                        if success:
                                            st.rerun()
                        
                        # 批量删除选中的记录
                        if selected_ids:
                            st.markdown("---")
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.info(f"已选择 {len(selected_ids)} 条记录")
                            with col2:
                                if st.button("🗑️ 删除选中", type="secondary", use_container_width=True):
                                    with st.spinner("正在删除选中的记录..."):
                                        success = st.session_state.components['milvus_manager'].delete_records_by_ids(selected_ids)
                                        if success:
                                            st.rerun()
                    else:
                        st.info("暂无记录可显示")
                
                with tab2:
                    st.markdown("#### 🔍 搜索并删除记录")
                    
                    # 文本搜索
                    search_text = st.text_input(
                        "搜索文本内容",
                        placeholder="输入要搜索的文本内容...",
                        help="支持模糊匹配，会搜索包含该文本的所有记录"
                    )
                    
                    if search_text:
                        # 搜索记录
                        search_results = st.session_state.components['milvus_manager'].search_records_by_text(search_text)
                        
                        if search_results:
                            st.success(f"🔍 找到 {len(search_results)} 条匹配记录")
                            
                            # 显示搜索结果
                            for i, record in enumerate(search_results):
                                col1, col2 = st.columns([5, 1])
                                
                                with col1:
                                    st.markdown(f"""
                                    <div class="record-item">
                                        <strong>ID:</strong> {record['id']}<br>
                                        <strong>文本:</strong> {record['text']}<br>
                                        <strong>元数据:</strong> {json.dumps(record['metadata'], ensure_ascii=False)}
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col2:
                                    if st.button("🗑️ 删除", key=f"search_delete_{record['id']}"):
                                        with st.spinner("正在删除记录..."):
                                            success = st.session_state.components['milvus_manager'].delete_records_by_ids([record['id']])
                                            if success:
                                                st.rerun()
                            
                            # 批量删除搜索结果
                            st.markdown("---")
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.warning(f"⚠️ 将删除所有包含 '{search_text}' 的记录")
                            with col2:
                                # 使用session state管理确认状态
                                if f'confirm_delete_search_{search_text}' not in st.session_state:
                                    st.session_state[f'confirm_delete_search_{search_text}'] = False
                                
                                if st.button("🗑️ 删除所有匹配", type="secondary", use_container_width=True):
                                    st.session_state[f'confirm_delete_search_{search_text}'] = True
                                    st.rerun()
                                
                                if st.session_state.get(f'confirm_delete_search_{search_text}', False):
                                    if st.button("⚠️ 确认删除所有", key="confirm_delete_all_search"):
                                        with st.spinner("正在删除所有匹配记录..."):
                                            success = st.session_state.components['milvus_manager'].delete_records_by_text_pattern(search_text)
                                            if success:
                                                st.session_state[f'confirm_delete_search_{search_text}'] = False
                                                st.rerun()
                        else:
                            st.info("🔍 未找到匹配的记录")
                
                with tab3:
                    st.markdown("#### 🗑️ 批量删除操作")
                    
                    st.warning("⚠️ 以下操作将永久删除数据，请谨慎操作！")
                    
                    # 清空所有数据
                    st.markdown("##### 🗑️ 清空所有数据")
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.error(f"⚠️ 将删除集合中的所有 {stats.get('num_entities', 0):,} 条记录")
                    with col2:
                        # 使用session state管理确认状态
                        if 'confirm_clear_all' not in st.session_state:
                            st.session_state.confirm_clear_all = False
                        
                        if st.button("🗑️ 清空所有数据", type="secondary", use_container_width=True):
                            st.session_state.confirm_clear_all = True
                            st.rerun()
                        
                        if st.session_state.confirm_clear_all:
                            if st.button("⚠️ 确认清空", key="confirm_clear_all_final"):
                                success = st.session_state.components['milvus_manager'].clear_all_data()
                                if success:
                                    st.session_state.confirm_clear_all = False
                                    st.rerun()
            
            else:
                st.info("📦 集合为空，暂无数据需要管理")
        
        # 集合统计信息
        if st.session_state.components['milvus_manager'].collection:
            st.markdown("### 📈 集合统计")
            stats = st.session_state.components['milvus_manager'].get_collection_stats()
            if stats:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("集合名称", stats.get('name', 'N/A'))
                with col2:
                    st.metric("数据条数", f"{stats.get('num_entities', 0):,}")
                with col3:
                    st.metric("集合状态", "✅ 活跃" if stats.get('is_loaded', False) else "⚠️ 未加载")
                
                # 详细信息
                with st.expander("📋 详细信息"):
                    st.json(stats)
    else:
        st.warning("⚠️ 未连接到Milvus数据库")
        st.info("💡 请确保Milvus服务器正在运行，并检查网络连接")

def search_page():
    st.markdown("## 🔍 文本搜索")
    
    # 检查数据源
    local_data_available = st.session_state.data_loaded
    milvus_data_available = False
    milvus_count = 0
    
    if st.session_state.components['milvus_manager'].is_connected:
        persistence_status = st.session_state.components['milvus_manager'].verify_data_persistence()
        milvus_data_available = persistence_status['status'] == 'success'
        milvus_count = persistence_status.get('num_entities', 0)
    
    if not local_data_available and not milvus_data_available:
        st.warning("⚠️ 请先上传并处理数据，或连接到包含数据的Milvus数据库")
        return
    
    # 数据源选择
    st.markdown("### 📊 可用数据源")
    
    col1, col2 = st.columns(2)
    with col1:
        if local_data_available:
            st.success(f"✅ 本地数据：{len(st.session_state.texts)} 条记录")
        else:
            st.info("📁 本地数据：暂无")
    
    with col2:
        if milvus_data_available:
            st.success(f"✅ Milvus数据库：{milvus_count:,} 条记录")
        else:
            st.info("🗄️ Milvus数据库：暂无")
    
    # 搜索设置
    search_methods = []
    if local_data_available:
        search_methods.append("🏠 本地搜索")
    if milvus_data_available:
        search_methods.append("🗄️ Milvus搜索")
    
    search_method = st.radio(
        "选择搜索方式",
        search_methods,
        horizontal=True,
        help="本地搜索使用内存中的向量，Milvus搜索使用数据库中的向量"
    )
    
    # 搜索界面
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
        search_button = st.button("🚀 开始搜索", type="primary", use_container_width=True)
    
    # 搜索参数
    col1, col2 = st.columns(2)
    with col1:
        top_k = st.slider("返回结果数量", 1, 50, 10, help="设置返回的搜索结果数量")
    with col2:
        similarity_threshold = st.slider("相似度阈值", 0.0, 1.0, 0.0, 0.1, help="过滤低相似度的结果")
    
    # 执行搜索
    if search_button and query:
        if not st.session_state.model_loaded:
            st.error("❌ 请先加载向量化模型")
            return
            
        with st.spinner("🔍 正在搜索相关内容..."):
            try:
                if "Milvus" in search_method:
                    results = st.session_state.components['search_engine'].search_milvus(query, top_k)
                else:
                    results = st.session_state.components['search_engine'].search_local(query, top_k)
                
                # 过滤结果
                filtered_results = [r for r in results if r['score'] >= similarity_threshold]
                
                if filtered_results:
                    st.success(f"🎉 找到 {len(filtered_results)} 个相关结果")
                    
                    # 显示搜索统计
                    stats = st.session_state.components['search_engine'].get_search_statistics(filtered_results)
                    
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
                        # 计算相似度百分比和颜色
                        similarity_pct = result['score'] * 100
                        if similarity_pct >= 80:
                            color = "#28a745"  # 绿色
                        elif similarity_pct >= 60:
                            color = "#ffc107"  # 黄色
                        else:
                            color = "#dc3545"  # 红色
                        
                        with st.expander(f"🔍 结果 {i+1} - 相似度: {similarity_pct:.1f}%", expanded=i < 3):
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
                    st.info("🔍 未找到满足条件的结果，请尝试：")
                    st.markdown("""
                    - 降低相似度阈值
                    - 使用不同的关键词
                    - 检查输入的查询内容
                    """)
                    
            except Exception as e:
                st.error(f"❌ 搜索失败: {e}")
                st.exception(e)

def clustering_page():
    st.markdown("## 🎯 聚类分析")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ 请先在'数据上传与处理'页面上传并处理数据")
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
                        st.plotly_chart(fig, use_container_width=True)
                        
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
    
    if st.button("🎯 执行聚类分析", type="primary", use_container_width=True):
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
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # 聚类摘要
                    st.markdown("### 📈 聚类摘要")
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
                            title = f"🔴 噪声点 ({info['size']} 个样本, {info['percentage']:.1f}%)"
                        else:
                            title = f"🎯 聚类 {cluster_id} ({info['size']} 个样本, {info['percentage']:.1f}%)"
                        
                        with st.expander(title):
                            st.markdown("**📝 样本文本:**")
                            for j, text in enumerate(info['sample_texts']):
                                st.write(f"{j+1}. {text}")
                        
            except Exception as e:
                st.error(f"❌ 聚类分析失败: {e}")
                st.exception(e)

def system_info_page():
    st.markdown("## ℹ️ 系统信息")
    
    # 数据持久化状态
    st.markdown("### 💾 数据持久化状态")
    
    if st.session_state.components['milvus_manager'].is_connected:
        persistence_status = st.session_state.components['milvus_manager'].verify_data_persistence()
        
        if persistence_status['status'] == 'success':
            st.success(f"✅ Milvus数据库：{persistence_status['num_entities']:,} 条记录")
        elif persistence_status['status'] == 'no_collection':
            st.info("📦 Milvus数据库：已连接，暂无数据")
        else:
            st.error(f"❌ Milvus数据库：{persistence_status['message']}")
    else:
        st.warning("⚠️ Milvus数据库：未连接")
    
    # 模型信息
    st.markdown("### 🤖 向量化模型信息")
    model_info = st.session_state.components['vector_processor'].get_model_info()
    if model_info:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("模型名称", model_info.get('model_name', 'N/A'))
        with col2:
            st.metric("加载状态", "✅ 已加载" if model_info.get('status') == 'loaded' else "❌ 未加载")
        with col3:
            st.metric("向量维度", model_info.get('dimension', 'N/A'))
        
        with st.expander("📋 模型详细信息"):
            st.json(model_info)
    
    # 数据状态
    st.markdown("### 📊 数据状态")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        status = "✅ 已加载" if st.session_state.data_loaded else "❌ 未加载"
        st.metric("本地数据状态", status)
    with col2:
        st.metric("本地文本数量", f"{len(st.session_state.texts):,}" if st.session_state.data_loaded else "0")
    with col3:
        if st.session_state.data_loaded and st.session_state.vectors is not None:
            vector_size = st.session_state.vectors.nbytes / 1024 / 1024
            st.metric("内存占用", f"{vector_size:.2f} MB")
        else:
            st.metric("内存占用", "0 MB")
    with col4:
        if st.session_state.data_loaded and st.session_state.vectors is not None:
            st.metric("向量维度", st.session_state.vectors.shape[1])
        else:
            st.metric("向量维度", "N/A")

if __name__ == "__main__":
    main()
