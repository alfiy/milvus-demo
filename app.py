# app_enhanced.py
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
from milvus_mongo_insert import milvus_mongo_upload,get_milvus_collection,get_mongo_collection
from pymongo import MongoClient
from pymilvus import connections, Collection
from config_manager import config_manager

# 页面配置
st.set_page_config(
    page_title="文本向量化与Milvus数据库解决方案",
    page_icon="🚀",
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
    .config-card {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# 初始化session state
@st.cache_resource
def init_components():
    # 从配置文件加载Milvus设置
    milvus_config = config_manager.get_milvus_config()
    
    return {
        'vector_processor': VectorProcessor(),
        'milvus_manager': MilvusManager(
            host=milvus_config.get("host", "localhost"),
            port=milvus_config.get("port", "19530"),
            user=milvus_config.get("user", ""),
            password=milvus_config.get("password", "")
        ),
        'clustering_analyzer': ClusteringAnalyzer(),
        'search_engine': SearchEngine()
    }

def config_management_page():
    """配置管理页面"""
    st.markdown("## ⚙️ 系统配置管理")
    
    # 显示当前配置状态
    st.markdown("### 📋 当前配置状态")
    
    # 获取当前配置
    current_config = config_manager.load_config()
    milvus_config = current_config.get("milvus", {})
    mongodb_config = current_config.get("mongodb", {})
    model_config = current_config.get("model", {})
    
    # 显示配置卡片
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="config-card">
            <h4>🗄️ Milvus配置</h4>
            <p><strong>主机:</strong> {milvus_config.get('host', 'N/A')}</p>
            <p><strong>端口:</strong> {milvus_config.get('port', 'N/A')}</p>
            <p><strong>集合名:</strong> {milvus_config.get('collection_name', 'N/A')}</p>
            <p><strong>自动连接:</strong> {'✅' if milvus_config.get('auto_connect', False) else '❌'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="config-card">
            <h4>🍃 MongoDB配置</h4>
            <p><strong>主机:</strong> {mongodb_config.get('host', 'N/A')}</p>
            <p><strong>端口:</strong> {mongodb_config.get('port', 'N/A')}</p>
            <p><strong>数据库:</strong> {mongodb_config.get('db_name', 'N/A')}</p>
            <p><strong>自动连接:</strong> {'✅' if mongodb_config.get('auto_connect', False) else '❌'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 配置操作
    st.markdown("---")
    st.markdown("### 🛠️ 配置操作")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📥 导入配置", use_container_width=True):
            st.session_state.show_import = True
    
    with col2:
        if st.button("📤 导出配置", use_container_width=True):
            st.session_state.show_export = True
    
    with col3:
        if st.button("🔄 重置配置", use_container_width=True):
            st.session_state.show_reset = True
    
    with col4:
        if st.button("🔄 重新加载", use_container_width=True):
            st.rerun()
    
    # 导入配置
    if st.session_state.get('show_import', False):
        st.markdown("#### 📥 导入配置文件")
        uploaded_config = st.file_uploader(
            "选择配置文件",
            type=['json'],
            help="选择之前导出的配置JSON文件"
        )
        
        if uploaded_config is not None:
            try:
                config_content = json.loads(uploaded_config.read().decode('utf-8'))
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.json(config_content)
                with col2:
                    if st.button("✅ 确认导入"):
                        # 保存临时文件
                        temp_path = "temp_config.json"
                        with open(temp_path, 'w', encoding='utf-8') as f:
                            json.dump(config_content, f, indent=2, ensure_ascii=False)
                        
                        if config_manager.import_config(temp_path):
                            st.success("✅ 配置导入成功！")
                            st.session_state.show_import = False
                            st.rerun()
                        else:
                            st.error("❌ 配置导入失败")
                        
                        # 清理临时文件
                        import os
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                            
            except Exception as e:
                st.error(f"❌ 配置文件格式错误: {e}")
    
    # 导出配置
    if st.session_state.get('show_export', False):
        st.markdown("#### 📤 导出配置")
        
        export_path = st.text_input(
            "导出文件名",
            value="milvus_config_backup.json",
            help="输入要保存的配置文件名"
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("将当前配置导出为JSON文件，可用于备份或在其他环境中导入")
        with col2:
            if st.button("📤 导出"):
                if config_manager.export_config(export_path):
                    st.success(f"✅ 配置已导出到: {export_path}")
                    
                    # 提供下载链接
                    with open(export_path, 'r', encoding='utf-8') as f:
                        config_data = f.read()
                    
                    st.download_button(
                        label="📥 下载配置文件",
                        data=config_data,
                        file_name=export_path,
                        mime="application/json"
                    )
                    st.session_state.show_export = False
                else:
                    st.error("❌ 配置导出失败")
    
    # 重置配置
    if st.session_state.get('show_reset', False):
        st.markdown("#### 🔄 重置配置")
        st.warning("⚠️ 此操作将重置所有配置为默认值，无法撤销！")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.error("确认要重置所有配置吗？这将清除所有保存的连接信息。")
        with col2:
            if st.button("⚠️ 确认重置"):
                if config_manager.reset_config():
                    st.success("✅ 配置已重置为默认值")
                    st.session_state.show_reset = False
                    st.rerun()
                else:
                    st.error("❌ 配置重置失败")
    
    # 详细配置信息
    with st.expander("🔍 查看完整配置", expanded=False):
        st.json(current_config)

def model_manager_page():
    st.markdown("## 🔥 嵌入模型管理")
    if 'vector_processor' not in st.session_state.components:
        st.session_state.components['vector_processor'] = VectorProcessor()
    vp = st.session_state.components['vector_processor']

    st.markdown("### 添加/下载嵌入模型")
    vp.select_and_add_model_ui()
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info("选择模型后点击加载，系统会自动保存您的选择")
    with col2:
        if st.button("🔥 加载并检测模型", type="primary", use_container_width=True):
            with st.spinner("正在加载/检测模型..."):
                success = vp.load_model()
                if success:
                    # 保存模型配置
                    config_manager.update_model_config(
                        last_used_model=vp.model_name,
                        auto_load=True
                    )
                    st.success("✅ 模型加载成功，配置已保存！")
                else:
                    st.error("❌ 模型加载失败，请检查模型目录或网络状态。")

    st.markdown("---")
    st.markdown("### 当前可用本地模型")
    if vp.available_models:
        st.write(vp.available_models)
        
        # 显示上次使用的模型
        model_config = config_manager.get_model_config()
        last_model = model_config.get("last_used_model", "")
        if last_model:
            st.info(f"💡 上次使用的模型: {last_model}")
    else:
        st.info("暂无本地模型，请先添加。")

def mongodb_config_page():
    st.markdown("## 🍃 MongoDB配置管理")
    st.markdown("### 请输入MongoDB连接信息")

    # 从配置文件加载MongoDB设置
    mongodb_config = config_manager.get_mongodb_config()

    # 初始化配置信息和连接对象
    if "mongodb_config" not in st.session_state:
        st.session_state.mongodb_config = {
            "host": mongodb_config.get("host", "localhost"),
            "port": mongodb_config.get("port", 27017),
            "username": mongodb_config.get("username", ""),
            "password": mongodb_config.get("password", ""),
            "db_name": mongodb_config.get("db_name", "textdb"),
            "col_name": mongodb_config.get("col_name", "metadata"),
            "connected": False,
            "error": ""
        }
    if "mongodb_client" not in st.session_state:
        st.session_state.mongodb_client = None

    config = st.session_state.mongodb_config

    col1, col2 = st.columns(2)
    with col1:
        config["host"] = st.text_input("MongoDB主机地址", value=config["host"])
        config["port"] = st.number_input("MongoDB端口", value=config["port"], min_value=1, max_value=65535)
        config["db_name"] = st.text_input("数据库名", value=config["db_name"])
        config["col_name"] = st.text_input("集合名", value=config["col_name"])
    with col2:
        config["username"] = st.text_input("用户名", value=config["username"], placeholder="可选")
        config["password"] = st.text_input("密码", value=config["password"], type="password", placeholder="可选")
        
        # 自动连接选项
        auto_connect = st.checkbox("启动时自动连接", value=mongodb_config.get("auto_connect", False))

    col1, col2 = st.columns([3, 1])
    with col1:
        st.info("测试连接成功后，配置将自动保存")
    with col2:
        test_button = st.button("🍃 测试连接", type="primary", use_container_width=True)

    if test_button:
        st.session_state.mongodb_config["connected"] = False
        st.session_state.mongodb_config["error"] = ""
        try:
            if config["username"] and config["password"]:
                uri = f"mongodb://{config['username']}:{config['password']}@{config['host']}:{config['port']}/{config['db_name']}?authSource=admin"
            else:
                uri = f"mongodb://{config['host']}:{config['port']}/"
            client = MongoClient(uri, serverSelectionTimeoutMS=3000)
            # 测试连接 & 读集合
            db = client[config["db_name"]]
            col = db[config["col_name"]]
            _ = col.estimated_document_count()
            st.session_state.mongodb_config["connected"] = True
            st.session_state.mongodb_client = client
            
            # 保存MongoDB配置
            config_manager.update_mongodb_config(
                host=config["host"],
                port=config["port"],
                username=config["username"],
                password=config["password"],
                db_name=config["db_name"],
                col_name=config["col_name"],
                auto_connect=auto_connect
            )
            
            st.success("✅ 连接成功！配置已保存，该连接对象已保存，可被其他模块自动复用。")
        except Exception as e:
            st.session_state.mongodb_config["error"] = str(e)
            st.session_state.mongodb_client = None
            st.error(f"❌ 连接失败: {e}")

    # 显示连接状态
    if config["connected"]:
        st.info(f"✅ 已连接MongoDB：{config['host']}:{config['port']}，数据库：{config['db_name']}，集合：{config['col_name']}")
    elif config["error"]:
        st.warning(f"⚠️ 连接错误：{config['error']}")

    st.markdown("---")
    st.markdown("💡 连接成功后，连接对象将自动用于后续数据写入、搜索等功能。配置会自动保存，下次启动时可选择自动连接。")

# 通用获取MongoDB集合对象的函数，自动复用连接对象
def get_shared_mongo_collection():
    config = st.session_state.get("mongodb_config", None)
    client = st.session_state.get("mongodb_client", None)
    if config and client:
        db = client[config["db_name"]]
        col = db[config["col_name"]]
        return col
    else:
        st.error("MongoDB未配置或未连接，请先在🍃 MongoDB配置管理页面完成连接。")
        return None

def milvus_mongo_semantic_search(query, top_k, milvus_collection, mongo_col, vector_processor):
    """
    使用 Milvus + MongoDB 进行语义搜索
    """
    try:
        # 1️⃣ 获取向量
        query_vector = vector_processor.encode([query])[0]

        # 2️⃣ 检查集合是否存在且已连接
        milvus_manager = st.session_state.components['milvus_manager']
        if not milvus_manager.collection:
            st.error("❌ Milvus 集合未初始化，请先创建集合或导入数据")
            return []

        collection = milvus_manager.collection

        # 3️⃣ 执行 Milvus 搜索
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }

        results = collection.search(
            data=[query_vector.tolist()],
            anns_field="vector",           # ✅ 字段名改为 "vector"
            param=search_params,
            limit=top_k,
            output_fields=["text", "metadata"]
        )

        # 4️⃣ 整理 Milvus 搜索结果
        ids, scores = [], []
        for hits in results:
            for hit in hits:
                ids.append(hit.id)
                scores.append(float(hit.distance))

        # 5️⃣ 查询 MongoDB 获取元数据
        docs = list(mongo_col.find({"_id": {"$in": ids}}))
        id2doc = {str(doc["_id"]): doc for doc in docs}

        combined = []
        for id_, score in zip(ids, scores):
            doc = id2doc.get(str(id_), {})
            combined.append({
                "id": id_,
                "score": score,
                "text": doc.get("text", ""),
                "metadata": doc.get("metadata", {}),
            })

        return combined

    except Exception as e:
        st.error(f"❌ 搜索失败: {e}")
        return []

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
        <h1>🚀 文本向量化与Milvus数据库解决方案</h1>
        <p>支持百万级数据的向量化、存储、搜索和聚类分析 - 配置持久化版本</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 侧边栏导航
    with st.sidebar:
        st.markdown("### 🎯 功能菜单")
        
        # 显示当前状态
        if st.session_state.data_loaded:
            st.success(f"✅ 已加载 {len(st.session_state.texts)} 条数据")
        else:
            st.info("📋 请先上传数据")
        
        # 模型加载状态
        if st.session_state.model_loaded:
            st.success("🔥 模型已加载")
        else:
            st.warning("⚠️ 模型未加载")
        
        # Milvus连接状态和数据持久化验证
        if st.session_state.components['milvus_manager'].is_connected:
            st.success("🗄️ Milvus已连接")
            
            # 显示连接信息
            conn_info = st.session_state.components['milvus_manager'].get_connection_info()
            st.caption(f"📍 {conn_info['host']}:{conn_info['port']}")
            
            # 验证数据持久化状态
            persistence_status = st.session_state.components['milvus_manager'].verify_data_persistence()
            if persistence_status['status'] == 'success':
                st.success(f"💾 持久化数据: {persistence_status['num_entities']:,} 条")
            elif persistence_status['status'] == 'no_collection':
                st.info("📝 暂无持久化集合")
            else:
                st.warning("⚠️ 数据状态未知")
        else:
            st.warning("⚠️ Milvus未连接")
        
        st.markdown("---")
        
        page = st.selectbox(
            "选择功能模块",
            ["🏠 首页概览", "⚙️ 系统配置管理", "🔥 嵌入模型管理", "📊 数据上传与处理", "🗄️ Milvus数据库管理","🍃 MongoDB配置管理", "🔍 文本搜索", "🎯 聚类分析", "ℹ️ 系统信息"],
            index=0
        )

    # 主要内容区域
    if page == "🏠 首页概览":
        home_page()
    elif page == "⚙️ 系统配置管理":
        config_management_page()
    elif page == "🔥 嵌入模型管理":
        model_manager_page()
    elif page == "📊 数据上传与处理":
        data_upload_page()
    elif page == "🗄️ Milvus数据库管理":
        milvus_management_page()
    elif page == "🍃 MongoDB配置管理":
        mongodb_config_page()
    elif page == "🔍 文本搜索":
        search_page()
    elif page == "🎯 聚类分析":
        clustering_page()
    elif page == "ℹ️ 系统信息":
        system_info_page()

def home_page():
    st.markdown("## 🏠 系统概览")
    
    # 配置状态显示
    st.markdown("### ⚙️ 配置状态")
    
    # 获取当前配置
    milvus_config = config_manager.get_milvus_config()
    mongodb_config = config_manager.get_mongodb_config()
    model_config = config_manager.get_model_config()
    
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
        model_status = "✅ 已配置" if model_config.get("last_used_model") else "❌ 未配置"
        model_auto = "🔄 自动加载" if model_config.get("auto_load", False) else "⚠️ 手动加载"
        st.markdown(f"""
        <div class="metric-card">
            <h3>🔥 模型</h3>
            <h2>{model_status}</h2>
            <p>{model_auto}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 数据持久化状态检查
    if st.session_state.components['milvus_manager'].is_connected:
        persistence_status = st.session_state.components['milvus_manager'].verify_data_persistence()
        
        if persistence_status['status'] == 'success':
            st.markdown(f"""
            <div class="persistence-status status-success">
                <h4>✅ 数据持久化状态良好</h4>
                <p>Milvus数据库中保存了 <strong>{persistence_status['num_entities']:,}</strong> 条记录</p>
                <p>配置已保存，重启应用后数据和连接设置都会自动恢复</p>
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
            <p>如已配置自动连接，请检查服务器状态</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 系统状态卡片
    st.markdown("### 📊 系统状态")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📋 本地数据</h3>
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
            <h3>🔥 向量维度</h3>
            <h2>{}</h2>
            <p>模型输出维度</p>
        </div>
        """.format(embedding_dim), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 功能介绍
    st.markdown("## 🎯 主要功能")
    
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
                <li><strong>配置自动保存和恢复</strong></li>
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
    
    # 快速开始提示
    if not milvus_config.get("host") or not model_config.get("last_used_model"):
        st.markdown("---")
        st.markdown("## 🚀 快速开始")
        
        if not milvus_config.get("host"):
            st.info("💡 首次使用？请先到 '🗄️ Milvus数据库管理' 页面配置数据库连接")
        
        if not model_config.get("last_used_model"):
            st.info("💡 请到 '🔥 嵌入模型管理' 页面选择并加载向量化模型")

def data_upload_page():
    st.markdown("## 📊 数据上传与处理")
    
    # 模型选择和加载
    st.markdown("### 🔥 模型选择与加载")
    
    # 嵌入模型管理页面已处理模型管理，这里只需用vector_processor的模型列表
    vp = st.session_state.components['vector_processor']
    available_models = vp.available_models
    if not available_models:
        st.warning("当前无可用嵌入模型！请前往'嵌入模型管理'页面添加模型。")
        return

    # 从配置中获取上次使用的模型
    model_config = config_manager.get_model_config()
    last_used_model = model_config.get("last_used_model", "")
    
    # 设置默认选择
    default_index = 0
    if last_used_model and last_used_model in available_models:
        default_index = available_models.index(last_used_model)

    current_model = st.selectbox(
        "选择本地嵌入模型",
        options=available_models,
        index=default_index,
        help="选择你要用于向量化的嵌入模型，系统会记住您的选择"
    )
    vp.model_name = current_model

    col1, col2 = st.columns([3, 1])
    with col1:
        if last_used_model:
            st.info(f"💡 上次使用的模型: {last_used_model}")
        else:
            st.info("选择模型后，系统会自动保存您的选择")
    with col2:
        if st.button("🔥 加载模型", type="primary", use_container_width=True):
            with st.spinner("正在加载模型..."):
                if vp.load_model():
                    st.session_state.model_loaded = True
                    # 保存模型配置
                    config_manager.update_model_config(
                        last_used_model=current_model,
                        auto_load=True
                    )
                    st.success("✅ 模型加载成功，配置已保存！")
                    st.rerun()
                else:
                    st.session_state.model_loaded = False

    if not st.session_state.model_loaded:
        st.warning("⚠️ 当前模型未加载，请先加载模型！")
        return

    st.info(f"✅ 已选模型: {vp.model_name}，向量维度: {vp.dimension if vp.dimension else '未知'}")

    st.markdown("---")
    
    # 数据上传选项
    upload_method = st.radio(
        "选择数据输入方式",
        ["📁 上传JSON文件", "✏️ 手动输入JSON数据", "🎯 使用示例数据"],
        horizontal=True
    )
    
    json_data = None
    
    if upload_method == "📁 上传JSON文件":
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
                st.info(f"📁 文件大小: {file_size:.2f} MB")
                
                # 显示数据格式检测结果
                sample_item = json_data[0] if json_data else {}
                if isinstance(sample_item, dict):
                    keys = list(sample_item.keys())
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
    
    elif upload_method == "🎯 使用示例数据":
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
        st.info(f"🎯 使用示例数据，共 {len(json_data)} 条古诗词")
    
    # 数据预览和处理
    if json_data:
        st.markdown("### 📋 数据预览")
        
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
                process_button = st.button("🚀 开始向量化处理并持久化", type="primary", use_container_width=True)
            
            if process_button:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.text("📊 正在处理文本数据...")
                    progress_bar.progress(30)
                    
                    texts, vectors, metadata = st.session_state.components['vector_processor'].process_json_data(json_data)
                    progress_bar.progress(70)
                    
                    if len(texts) > 0:
                        # 保存到session state
                        st.session_state.texts = texts
                        st.session_state.vectors = vectors
                        st.session_state.metadata = metadata
                        st.session_state.data_loaded = True
                        
                        # ------ 新增：自动批量插入Milvus和MongoDB ------
                        embedding_dim = vectors.shape[1]
                        status_text.text("💾 正在批量插入 Milvus & MongoDB ...")
                        inserted_ids = milvus_mongo_upload(texts, vectors, metadata, milvus_dim=embedding_dim)
                        progress_bar.progress(100)
                        status_text.text(f"✅ 向量化及持久化完成！已插入 {len(inserted_ids)} 条数据。")
                        st.success(f"🎉 向量化和持久化完成！成功处理并写入 {len(inserted_ids)} 条文本数据。")

                        # 搜索引擎设置
                        st.session_state.components['search_engine'].load_data(vectors, texts, metadata)
                        st.session_state.components['search_engine'].set_vector_processor(st.session_state.components['vector_processor'])
                        
                        # 设置聚类分析器数据
                        st.session_state.components['clustering_analyzer'].load_data(vectors, texts, metadata)
                        
                        progress_bar.progress(100)
                        status_text.text("✅ 向量化处理完成！")
                        
                        st.success(f"🎉 向量化完成！成功处理了 {len(texts)} 条文本")
                        
                        # 显示处理结果统计
                        st.markdown("### 📊 处理结果")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("文本数量", len(texts))
                        with col2:
                            st.metric("向量维度", vectors.shape[1])
                        with col3:
                            st.metric("数据大小", f"{vectors.nbytes / 1024 / 1024:.2f} MB")
                        with col4:
                            st.metric("处理状态", "✅ 完成")
                        
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
            st.info("📝 数据库已连接，但尚未创建数据集合")
        else:
            st.error(f"❌ {persistence_status['message']}")
    else:
        st.warning("⚠️ 尚未连接到Milvus数据库")
    
    # 连接设置
    st.markdown("### 🔌 数据库连接")
    
    # 从配置文件加载设置
    milvus_config = config_manager.get_milvus_config()
    
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        host = st.text_input("Milvus主机地址", value=milvus_config.get("host", "localhost"), help="Milvus服务器的IP地址或域名")
    with col2:
        port = st.text_input("端口", value=str(milvus_config.get("port", "19530")), help="Milvus服务器端口，默认19530")
    with col3:
        user = st.text_input("用户名", value=milvus_config.get("user", ""), help="Milvus用户名（可选）", placeholder="可选")
    with col4:
        password = st.text_input("密码", value="", type="password", help="Milvus密码（可选）", placeholder="可选")
    
    # 集合名称和自动连接选项
    col1, col2 = st.columns(2)
    with col1:
        collection_name = st.text_input("集合名称", value=milvus_config.get("collection_name", "text_vectors"), help="向量集合的名称")
    with col2:
        auto_connect = st.checkbox("启动时自动连接", value=milvus_config.get("auto_connect", False), help="下次启动应用时自动连接到此Milvus服务器")
    
    # 连接按钮
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info("💡 连接成功后，配置将自动保存。如果Milvus服务器未设置认证，用户名和密码可以留空")
    with col2:
        connect_button = st.button("🔌 连接数据库", type="primary", use_container_width=True)
    
    # 连接操作
    if connect_button:
        with st.spinner("正在连接到Milvus数据库..."):
            # 更新连接参数
            st.session_state.components['milvus_manager'].update_connection_params(
                host=host,
                port=port,
                user=user,
                password=password,
                collection_name=collection_name
            )
            
            success = st.session_state.components['milvus_manager'].connect(save_config=True)
            if success:
                # 额外保存自动连接设置
                config_manager.update_milvus_config(
                    host=host,
                    port=port,
                    user=user,
                    password=password,
                    collection_name=collection_name,
                    auto_connect=auto_connect
                )
                st.session_state.components['search_engine'].set_milvus_manager(st.session_state.components['milvus_manager'])
                st.rerun()
    
    # 显示连接状态
    if st.session_state.components['milvus_manager'].is_connected:
        conn_info = st.session_state.components['milvus_manager'].get_connection_info()
        connection_display = f"{conn_info['host']}:{conn_info['port']}"
        if conn_info['user']:
            connection_display += f" (用户: {conn_info['user']})"
        
        st.success(f"✅ 已成功连接到Milvus数据库 ({connection_display})")
        st.info(f"📋 当前集合: {conn_info['collection_name']}")
        
        # 集合管理
        st.markdown("### 📋 集合管理")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📋 创建/连接集合", use_container_width=True):
                if st.session_state.data_loaded:
                    with st.spinner("正在创建/连接集合..."):
                        dimension = st.session_state.vectors.shape[1]
                        success = st.session_state.components['milvus_manager'].create_collection(dimension)
                        if success:
                            st.rerun()
                else:
                    st.warning("⚠️ 请先上传并处理数据")
        
        with col2:
            if st.button("💾 插入数据到Milvus", use_container_width=True):
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
                tab1, tab2, tab3 = st.tabs(["👁️ 数据预览", "🔍 搜索删除", "🗑️ 批量删除"])
                
                with tab1:
                    st.markdown("#### 👁️ 数据预览")
                    
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
                st.info("📝 集合为空，暂无数据需要管理")
        
        # 集合统计信息
        if st.session_state.components['milvus_manager'].collection:
            st.markdown("### 📊 集合统计")
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
                with st.expander("🔍 详细信息"):
                    st.json(stats)
    else:
        st.warning("⚠️ 未连接到Milvus数据库")
        st.info("💡 请确保Milvus服务器正在运行，并检查网络连接")

def search_page():
    st.markdown("## 🔍 文本搜索")
    
    # 检查MongoDB和Milvus是否已连接
    milvus_collection = get_milvus_collection(
        collection_name="text_embeddings",
        dim=st.session_state.vectors.shape[1] if st.session_state.vectors is not None else 384
    )
    mongo_col = get_mongo_collection()
    vector_processor = st.session_state.components["vector_processor"]

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
        search_button = st.button("🔍 开始搜索", type="primary", use_container_width=True)

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
                results = milvus_mongo_semantic_search(query, top_k, milvus_collection, mongo_col, vector_processor)
                # 过滤结果
                filtered_results = [r for r in results if r['score'] >= similarity_threshold]
                if filtered_results:
                    st.success(f"🎯 找到 {len(filtered_results)} 个相关结果")
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
                    st.markdown("### 🎯 搜索结果")
                    for i, result in enumerate(filtered_results):
                        similarity_pct = result['score'] * 100
                        if similarity_pct >= 80:
                            color = "#28a745"  # 绿色
                        elif similarity_pct >= 60:
                            color = "#ffc107"  # 黄色
                        else:
                            color = "#dc3545"  # 红色
                        with st.expander(f"🎯 结果 {i+1} - 相似度: {similarity_pct:.1f}%", expanded=i < 3):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown("**📄 文本内容:**")
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
        st.warning("⚠️ 请先在'📊 数据上传与处理'页面上传并处理数据")
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
            if st.button("🎯 寻找最优K值", help="使用轮廓系数寻找最佳聚类数"):
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
                            title = f"🔴 噪声点 ({info['size']} 个样本, {info['percentage']:.1f}%)"
                        else:
                            title = f"🎯 聚类 {cluster_id} ({info['size']} 个样本, {info['percentage']:.1f}%)"
                        
                        with st.expander(title):
                            st.markdown("**📄 样本文本:**")
                            for j, text in enumerate(info['sample_texts']):
                                st.write(f"{j+1}. {text}")
                        
            except Exception as e:
                st.error(f"❌ 聚类分析失败: {e}")
                st.exception(e)

def system_info_page():
    st.markdown("## ℹ️ 系统信息")
    
    # 配置信息
    st.markdown("### ⚙️ 配置信息")
    
    current_config = config_manager.load_config()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🗄️ Milvus配置")
        milvus_config = current_config.get("milvus", {})
        st.json(milvus_config)
    
    with col2:
        st.markdown("#### 🍃 MongoDB配置")
        mongodb_config = current_config.get("mongodb", {})
        # 隐藏密码
        display_config = mongodb_config.copy()
        if display_config.get("password"):
            display_config["password"] = "***"
        st.json(display_config)
    
    # 数据持久化状态
    st.markdown("### 💾 数据持久化状态")
    
    if st.session_state.components['milvus_manager'].is_connected:
        persistence_status = st.session_state.components['milvus_manager'].verify_data_persistence()
        
        if persistence_status['status'] == 'success':
            st.success(f"✅ Milvus数据库：{persistence_status['num_entities']:,} 条记录")
        elif persistence_status['status'] == 'no_collection':
            st.info("📝 Milvus数据库：已连接，暂无数据")
        else:
            st.error(f"❌ Milvus数据库：{persistence_status['message']}")
    else:
        st.warning("⚠️ Milvus数据库：未连接")
    
    # 模型信息
    st.markdown("### 🔥 向量化模型信息")
    model_info = st.session_state.components['vector_processor'].get_model_info()
    if model_info:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("模型名称", model_info.get('model_name', 'N/A'))
        with col2:
            st.metric("加载状态", "✅ 已加载" if model_info.get('status') == 'loaded' else "❌ 未加载")
        with col3:
            st.metric("向量维度", model_info.get('dimension', 'N/A'))
        
        with st.expander("🔍 模型详细信息"):
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
    
    # 连接信息
    st.markdown("### 🔌 连接信息")
    
    if st.session_state.components['milvus_manager'].is_connected:
        conn_info = st.session_state.components['milvus_manager'].get_connection_info()
        st.json(conn_info)
    else:
        st.info("Milvus未连接")

if __name__ == "__main__":
    main()
