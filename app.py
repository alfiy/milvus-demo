# app_enhanced.py - 修复版本
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
# 使用修复后的模块
from milvus_mongo_insert import milvus_mongo_upload, get_milvus_collection, get_mongo_collection, debug_collection_info
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
    .model-card {
        background: #f3e5f5;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #9c27b0;
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

def check_and_load_model_on_startup():
    """检查并在启动时自动加载模型"""
    model_config = config_manager.get_model_config()
    if model_config.get("auto_load", False) and model_config.get("last_used_model"):
        vp = st.session_state.components['vector_processor']
        if not st.session_state.get('model_loaded', False):
            # 设置模型名称
            vp.model_name = model_config.get("last_used_model")
            # 尝试加载模型
            if vp.load_model():
                st.session_state.model_loaded = True
                return True
    return False

def check_and_connect_mongodb_on_startup():
    """检查并在启动时自动连接MongoDB"""
    mongodb_config = config_manager.get_mongodb_config()
    if mongodb_config.get("auto_connect", False) and mongodb_config.get("host"):
        if "mongodb_config" not in st.session_state or not st.session_state.get("mongodb_client"):
            try:
                # 初始化配置
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
                
                # 尝试连接
                config = st.session_state.mongodb_config
                if config["username"] and config["password"]:
                    uri = f"mongodb://{config['username']}:{config['password']}@{config['host']}:{config['port']}/{config['db_name']}?authSource=admin"
                else:
                    uri = f"mongodb://{config['host']}:{config['port']}/"
                
                client = MongoClient(uri, serverSelectionTimeoutMS=3000)
                # 测试连接
                db = client[config["db_name"]]
                col = db[config["col_name"]]
                _ = col.estimated_document_count()
                
                # 连接成功
                st.session_state.mongodb_config["connected"] = True
                st.session_state.mongodb_client = client
                return True
                
            except Exception as e:
                st.session_state.mongodb_config["error"] = str(e)
                st.session_state.mongodb_client = None
                return False
    return False

def config_management_page():
    """配置管理页面"""
    st.markdown("## ⚙️ 系统配置管理")
    
    # 显示当前配置状态
    st.markdown("### 📊 当前配置状态")
    
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
        if st.button("📥 导入配置", width="stretch"):
            st.session_state.show_import = True
    
    with col2:
        if st.button("📤 导出配置", width="stretch"):
            st.session_state.show_export = True
    
    with col3:
        if st.button("🔄 重置配置", width="stretch"):
            st.session_state.show_reset = True
    
    with col4:
        if st.button("🔃 重新加载", width="stretch"):
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
    st.markdown("## 🤖 嵌入模型管理")
    
    if 'vector_processor' not in st.session_state.components:
        st.session_state.components['vector_processor'] = VectorProcessor()
    vp = st.session_state.components['vector_processor']

    # 显示当前模型状态
    st.markdown("### 📊 当前模型状态")
    
    model_config = config_manager.get_model_config()
    current_loaded_model = model_config.get("last_used_model", "") if st.session_state.get('model_loaded', False) else ""
    
    col1, col2 = st.columns(2)
    
    with col1:
        if current_loaded_model:
            st.markdown(f"""
            <div class="model-card">
                <h4>🤖 当前加载的模型</h4>
                <p><strong>模型名称:</strong> {current_loaded_model}</p>
                <p><strong>加载状态:</strong> ✅ 已加载</p>
                <p><strong>自动加载:</strong> {'✅' if model_config.get('auto_load', False) else '❌'}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="model-card">
                <h4>🤖 当前模型状态</h4>
                <p><strong>加载状态:</strong> ❌ 未加载</p>
                <p><strong>提示:</strong> 请选择并加载模型</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # 模型信息
        if current_loaded_model:
            model_info = vp.get_model_info()
            if model_info:
                st.markdown(f"""
                <div class="model-card">
                    <h4>📋 模型详情</h4>
                    <p><strong>向量维度:</strong> {model_info.get('dimension', 'N/A')}</p>
                    <p><strong>模型类型:</strong> {model_info.get('model_type', 'Sentence Transformer')}</p>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🛠️ 模型选择与管理")

    # 显示可用模型和添加新模型功能
    vp.select_and_add_model_ui()
    
    # 模型选择和加载
    if vp.available_models:
        st.markdown("#### 🎯 选择并加载模型")
        
        # 从配置中获取上次使用的模型
        last_used_model = model_config.get("last_used_model", "")
        
        # 设置默认选择
        default_index = 0
        if last_used_model and last_used_model in vp.available_models:
            default_index = vp.available_models.index(last_used_model)

        selected_model = st.selectbox(
            "选择要加载的嵌入模型",
            options=vp.available_models,
            index=default_index,
            help="选择你要用于向量化的嵌入模型"
        )
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            auto_load = st.checkbox(
                "启动时自动加载此模型", 
                value=model_config.get("auto_load", False),
                help="勾选后，应用启动时会自动加载此模型"
            )
        
        with col2:
            load_button = st.button("🚀 加载模型", type="primary", width="stretch")
        
        with col3:
            if current_loaded_model:
                unload_button = st.button("🔄 卸载模型", width="stretch")
            else:
                unload_button = False
        
        # 加载模型
        if load_button:
            with st.spinner("正在加载模型..."):
                # 设置选中的模型
                vp.model_name = selected_model
                success = vp.load_model()
                
                if success:
                    # 更新全局状态
                    st.session_state.model_loaded = True
                    
                    # 保存模型配置
                    config_manager.update_model_config(
                        last_used_model=selected_model,
                        auto_load=auto_load
                    )
                    
                    st.success("✅ 模型加载成功，配置已保存！")
                    st.rerun()
                else:
                    st.session_state.model_loaded = False
                    st.error("❌ 模型加载失败，请检查模型文件或网络状态。")
        
        # 卸载模型
        if unload_button:
            st.session_state.model_loaded = False
            # 更新配置但不改变auto_load设置
            config_manager.update_model_config(
                last_used_model="",
                auto_load=auto_load
            )
            st.success("✅ 模型已卸载")
            st.rerun()
        
        # 更新自动加载设置（当复选框状态改变时）
        if auto_load != model_config.get("auto_load", False):
            config_manager.update_model_config(
                last_used_model=model_config.get("last_used_model", ""),
                auto_load=auto_load
            )
    
    else:
        st.warning("⚠️ 暂无可用模型，请先添加模型。")
        st.info("💡 使用上方的模型添加功能来下载或添加本地模型。")
    
    # 显示所有可用模型列表
    if vp.available_models:
        st.markdown("---")
        st.markdown("### 📋 所有可用模型")
        
        for i, model in enumerate(vp.available_models):
            col1, col2 = st.columns([4, 1])
            with col1:
                status = "🟢 当前加载" if model == current_loaded_model else "⚪ 未加载"
                st.write(f"{i+1}. **{model}** - {status}")
            with col2:
                if model != current_loaded_model:
                    if st.button("⚡ 快速加载", key=f"quick_load_{i}", width="stretch"):
                        with st.spinner(f"正在加载 {model}..."):
                            vp.model_name = model
                            if vp.load_model():
                                st.session_state.model_loaded = True
                                config_manager.update_model_config(
                                    last_used_model=model,
                                    auto_load=model_config.get("auto_load", False)
                                )
                                st.success(f"✅ {model} 加载成功！")
                                st.rerun()

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

    # 显示当前连接状态
    st.markdown("### 📊 当前连接状态")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if config.get("connected", False):
            st.markdown(f"""
            <div class="config-card">
                <h4>✅ MongoDB 已连接</h4>
                <p><strong>主机:</strong> {config['host']}:{config['port']}</p>
                <p><strong>数据库:</strong> {config['db_name']}</p>
                <p><strong>集合:</strong> {config['col_name']}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="config-card">
                <h4>❌ MongoDB 未连接</h4>
                <p>请配置连接信息并测试连接</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        auto_connect_status = "✅ 开启" if mongodb_config.get("auto_connect", False) else "❌ 关闭"
        st.markdown(f"""
        <div class="config-card">
            <h4>⚙️ 配置状态</h4>
            <p><strong>自动连接:</strong> {auto_connect_status}</p>
            <p><strong>配置保存:</strong> {'✅ 已保存' if mongodb_config.get('host') else '❌ 未保存'}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔧 连接配置")

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
        st.info("💡 测试连接成功后，配置将自动保存到配置文件，重启应用后可自动恢复连接")
    with col2:
        test_button = st.button("🔗 测试连接", type="primary", width="stretch")

    if test_button:
        st.session_state.mongodb_config["connected"] = False
        st.session_state.mongodb_config["error"] = ""
        
        with st.spinner("正在测试MongoDB连接..."):
            try:
                if config["username"] and config["password"]:
                    uri = f"mongodb://{config['username']}:{config['password']}@{config['host']}:{config['port']}/{config['db_name']}?authSource=admin"
                else:
                    uri = f"mongodb://{config['host']}:{config['port']}/"
                
                client = MongoClient(uri, serverSelectionTimeoutMS=5000)
                # 测试连接 & 读集合
                db = client[config["db_name"]]
                col = db[config["col_name"]]
                _ = col.estimated_document_count()
                
                # 连接成功
                st.session_state.mongodb_config["connected"] = True
                st.session_state.mongodb_client = client
                
                # 保存MongoDB配置到配置文件
                config_manager.update_mongodb_config(
                    host=config["host"],
                    port=config["port"],
                    username=config["username"],
                    password=config["password"],
                    db_name=config["db_name"],
                    col_name=config["col_name"],
                    auto_connect=auto_connect
                )
                
                st.success("✅ 连接成功！配置已保存到配置文件，重启应用后可自动连接。")
                
            except Exception as e:
                st.session_state.mongodb_config["error"] = str(e)
                st.session_state.mongodb_client = None
                st.error(f"❌ 连接失败: {e}")
                
                # 提供常见问题解决方案
                st.markdown("### 🔧 常见问题解决")
                st.markdown("""
                **可能的原因：**
                - MongoDB服务未启动
                - 主机地址或端口错误
                - 用户名密码错误
                - 网络连接问题
                - 防火墙阻止连接
                
                **解决建议：**
                1. 检查MongoDB服务是否运行：`sudo systemctl status mongod`
                2. 确认连接信息是否正确
                3. 如果是本地连接，尝试使用 `127.0.0.1` 而不是 `localhost`
                4. 检查防火墙设置
                """)

    # 显示连接状态详情
    if config.get("error"):
        st.markdown("### ❌ 连接错误详情")
        st.error(f"错误信息：{config['error']}")
    
    # 连接管理操作
    if config.get("connected", False):
        st.markdown("---")
        st.markdown("### 🛠️ 连接管理")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 重新连接", width="stretch"):
                st.session_state.mongodb_config["connected"] = False
                st.session_state.mongodb_client = None
                st.rerun()
        
        with col2:
            if st.button("🧪 测试数据库", width="stretch"):
                try:
                    client = st.session_state.mongodb_client
                    db = client[config["db_name"]]
                    col = db[config["col_name"]]
                    count = col.estimated_document_count()
                    st.success(f"✅ 数据库连接正常，集合中有 {count:,} 条记录")
                except Exception as e:
                    st.error(f"❌ 数据库测试失败: {e}")
        
        with col3:
            if st.button("🔌 断开连接", width="stretch"):
                if st.session_state.mongodb_client:
                    st.session_state.mongodb_client.close()
                st.session_state.mongodb_config["connected"] = False
                st.session_state.mongodb_client = None
                st.info("✅ 已断开MongoDB连接")
                st.rerun()

    st.markdown("---")
    st.markdown("### 📚 使用说明")
    st.markdown("""
    **配置持久化：**
    - 连接成功后，配置会自动保存到 `config.json` 文件
    - 重启应用时，如果启用了"自动连接"，会自动尝试连接
    - 配置文件包含连接信息（密码会加密存储）
    
    **在其他功能中使用：**
    - 搜索功能会自动使用这里配置的MongoDB连接
    - 数据上传功能会将元数据保存到MongoDB
    - 如果连接断开，系统会提示重新连接
    """)

# 通用获取MongoDB集合对象的函数，自动复用连接对象
def get_shared_mongo_collection():
    config = st.session_state.get("mongodb_config", None)
    client = st.session_state.get("mongodb_client", None)
    if config and client and config.get("connected", False):
        try:
            db = client[config["db_name"]]
            col = db[config["col_name"]]
            # 测试连接是否仍然有效
            _ = col.estimated_document_count()
            return col
        except Exception as e:
            st.error(f"❌ MongoDB连接已断开: {e}")
            st.info("💡 请到 '🍃 MongoDB配置管理' 页面重新连接")
            return None
    else:
        st.error("❌ MongoDB未配置或未连接，请先在 '🍃 MongoDB配置管理' 页面完成连接。")
        return None

def milvus_mongo_semantic_search(query, top_k, milvus_collection, mongo_col, vector_processor):
    """
    使用 Milvus + MongoDB 进行语义搜索 - 修复版本
    """
    try:
        # 1️⃣ 获取向量
        query_vector = vector_processor.encode([query])[0]

        # 2️⃣ 检查集合是否存在且已连接
        if not milvus_collection:
            st.error("❌ Milvus 集合未初始化，请先创建集合或导入数据")
            return []

        # 3️⃣ 执行 Milvus 搜索
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }

        results = milvus_collection.search(
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
    
    # 检查并自动加载模型
    if not st.session_state.model_loaded:
        check_and_load_model_on_startup()
    
    # 检查并自动连接MongoDB
    check_and_connect_mongodb_on_startup()

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
        st.markdown("### 🧭 功能菜单")
        
        # 显示当前状态
        if st.session_state.data_loaded:
            st.success(f"✅ 已加载 {len(st.session_state.texts)} 条数据")
        else:
            st.info("💡 请先上传数据")
        
        # 模型加载状态
        if st.session_state.model_loaded:
            model_config = config_manager.get_model_config()
            current_model = model_config.get("last_used_model", "")
            st.success("🤖 模型已加载")
            if current_model:
                st.caption(f"📋 {current_model}")
        else:
            st.warning("⚠️ 模型未加载")
        
        # MongoDB连接状态
        mongodb_config = st.session_state.get("mongodb_config", {})
        if mongodb_config.get("connected", False):
            st.success("🍃 MongoDB已连接")
            st.caption(f"🔗 {mongodb_config['host']}:{mongodb_config['port']}")
        else:
            st.warning("⚠️ MongoDB未连接")
        
        # Milvus连接状态和数据持久化验证
        if st.session_state.components['milvus_manager'].is_connected:
            st.success("🗄️ Milvus已连接")
            
            # 显示连接信息
            conn_info = st.session_state.components['milvus_manager'].get_connection_info()
            st.caption(f"🔗 {conn_info['host']}:{conn_info['port']}")
            
            # 验证数据持久化状态
            persistence_status = st.session_state.components['milvus_manager'].verify_data_persistence()
            if persistence_status['status'] == 'success':
                st.success(f"💾 持久化数据: {persistence_status['num_entities']:,} 条")
            elif persistence_status['status'] == 'no_collection':
                st.info("📄 暂无持久化集合")
            else:
                st.warning("⚠️ 数据状态未知")
        else:
            st.warning("⚠️ Milvus未连接")
        
        st.markdown("---")
        
        page = st.selectbox(
            "选择功能模块",
            ["🏠 首页概览", "⚙️ 系统配置管理", "🤖 嵌入模型管理", "📊 数据上传与处理", "🗄️ Milvus数据库管理","🍃 MongoDB配置管理", "🔍 文本搜索", "🎯 聚类分析", "ℹ️ 系统信息"],
            index=0
        )

    # 主要内容区域
    if page == "🏠 首页概览":
        home_page()
    elif page == "⚙️ 系统配置管理":
        config_management_page()
    elif page == "🤖 嵌入模型管理":
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
        model_status = "✅ 已加载" if st.session_state.model_loaded else "❌ 未加载"
        model_auto = "🔄 自动加载" if model_config.get("auto_load", False) else "⚠️ 手动加载"
        st.markdown(f"""
        <div class="metric-card">
            <h3>🤖 模型</h3>
            <h2>{model_status}</h2>
            <p>{model_auto}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 连接状态显示
    st.markdown("### 🔗 连接状态")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # MongoDB连接状态
        mongodb_session_config = st.session_state.get("mongodb_config", {})
        if mongodb_session_config.get("connected", False):
            st.markdown(f"""
            <div class="persistence-status status-success">
                <h4>✅ MongoDB连接正常</h4>
                <p>已连接到 <strong>{mongodb_session_config['host']}:{mongodb_session_config['port']}</strong></p>
                <p>数据库: {mongodb_session_config['db_name']}, 集合: {mongodb_session_config['col_name']}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="persistence-status status-warning">
                <h4>⚠️ MongoDB未连接</h4>
                <p>请到 '🍃 MongoDB配置管理' 页面配置连接</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Milvus连接状态
        if st.session_state.components['milvus_manager'].is_connected:
            persistence_status = st.session_state.components['milvus_manager'].verify_data_persistence()
            
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
        st.markdown("""
        <div class="metric-card">
            <h3>📄 本地数据</h3>
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
        embedding_dim = model_info.get('dimension', 'N/A') if st.session_state.model_loaded else 'N/A'
        st.markdown("""
        <div class="metric-card">
            <h3>🔢 向量维度</h3>
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
        
        if not st.session_state.model_loaded:
            st.info("💡 请先到 '🤖 嵌入模型管理' 页面选择并加载向量化模型")
        
        if not milvus_config.get("host"):
            st.info("💡 请到 '🗄️ Milvus数据库管理' 页面配置数据库连接")
        
        if not mongodb_config.get("host"):
            st.info("💡 请到 '🍃 MongoDB配置管理' 页面配置元数据存储")

def data_upload_page():
    st.markdown("## 📊 数据上传与处理")
    
    # 检查模型加载状态
    if not st.session_state.model_loaded:
        st.warning("⚠️ 尚未加载嵌入模型！")
        st.info("💡 请先到 '🤖 嵌入模型管理' 页面加载模型，然后再回到此页面进行数据处理。")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("""
            **为什么需要先加载模型？**
            - 文本向量化需要使用嵌入模型
            - 模型加载后可以处理任何文本数据
            - 统一的模型管理确保配置一致性
            """)
        with col2:
            if st.button("🚀 前往模型管理", type="primary", width="stretch"):
                st.switch_page("🤖 嵌入模型管理")
        return
    
    # 显示当前使用的模型
    model_config = config_manager.get_model_config()
    current_model = model_config.get("last_used_model", "")
    
    st.markdown("### 🤖 当前模型状态")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.success(f"✅ 已加载模型: **{current_model}**")
        vp = st.session_state.components['vector_processor']
        model_info = vp.get_model_info()
        if model_info:
            st.info(f"🔢 向量维度: {model_info.get('dimension', 'N/A')}")
    with col2:
        if st.button("🔄 切换模型", width="stretch"):
            # 这里可以添加快速切换模型的功能，或者跳转到模型管理页面
            st.info("💡 请到 '🤖 嵌入模型管理' 页面切换模型")

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
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("点击下方按钮开始文本向量化处理，处理后的数据可以保存到Milvus数据库中永久存储")
        with col2:
            process_button = st.button("🚀 开始向量化处理并持久化", type="primary", width="stretch")
        
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
                    
                    try:
                        inserted_ids = milvus_mongo_upload(texts, vectors, metadata, milvus_dim=embedding_dim)
                        progress_bar.progress(100)
                        status_text.text(f"✅ 向量化及持久化完成！已插入 {len(inserted_ids)} 条数据。")
                        st.success(f"🎉 向量化和持久化完成！成功处理并写入 {len(inserted_ids)} 条文本数据。")
                    except Exception as e:
                        progress_bar.progress(100)
                        status_text.text("⚠️ 向量化完成，但持久化失败")
                        st.warning(f"⚠️ 向量化完成，但数据持久化失败: {e}")
                        st.info("💡 数据已保存到内存中，可以进行搜索和聚类分析。要启用持久化，请检查Milvus和MongoDB连接。")

                    # 搜索引擎设置
                    st.session_state.components['search_engine'].load_data(vectors, texts, metadata)
                    st.session_state.components['search_engine'].set_vector_processor(st.session_state.components['vector_processor'])
                    
                    # 设置聚类分析器数据
                    st.session_state.components['clustering_analyzer'].load_data(vectors, texts, metadata)
                    
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
            st.info("📄 数据库已连接，但尚未创建数据集合")
        else:
            st.error(f"❌ {persistence_status['message']}")
    else:
        st.warning("⚠️ 尚未连接到Milvus数据库")
    
    # 连接设置
    st.markdown("### 🔗 数据库连接")
    
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
        connect_button = st.button("🔗 连接数据库", type="primary", width="stretch")
    
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
        st.markdown("### 🛠️ 集合管理")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🆕 创建/连接集合", width="stretch"):
                if st.session_state.data_loaded:
                    with st.spinner("正在创建/连接集合..."):
                        dimension = st.session_state.vectors.shape[1]
                        success = st.session_state.components['milvus_manager'].create_collection(dimension)
                        if success:
                            st.rerun()
                else:
                    st.warning("⚠️ 请先上传并处理数据")
        
        with col2:
            if st.button("📤 插入数据到Milvus", width="stretch"):
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
            # 修复删除集合功能
            delete_collection_key = "confirm_delete_collection"
            
            # 初始化确认状态
            if delete_collection_key not in st.session_state:
                st.session_state[delete_collection_key] = False
            
            # 如果还没有确认，显示删除按钮
            if not st.session_state[delete_collection_key]:
                if st.button("🗑️ 删除集合", width="stretch", key="delete_collection_btn"):
                    st.session_state[delete_collection_key] = True
                    st.rerun()
            else:
                # 已经点击了删除，显示确认按钮
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("⚠️ 确认删除", type="secondary", width="content", key="confirm_delete_btn"):
                        with st.spinner("正在删除集合..."):
                            success = st.session_state.components['milvus_manager'].delete_collection()
                            if success:
                                st.session_state[delete_collection_key] = False
                                st.success("✅ 集合已删除")
                                st.rerun()
                            else:
                                st.error("❌ 删除集合失败")
                with col_b:
                    if st.button("❌ 取消", width="content", key="cancel_delete_btn"):
                        st.session_state[delete_collection_key] = False
                        st.rerun()
        
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
        
        # 调试功能
        st.markdown("### 🔧 调试工具")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔍 调试集合信息", width="stretch"):
                debug_collection_info("text_vectors")
        with col2:
            if st.button("🧪 测试连接", width="stretch"):
                try:
                    collection = get_milvus_collection("text_vectors", 384)
                    if collection:
                        st.success("✅ 集合连接测试成功")
                    else:
                        st.error("❌ 集合连接测试失败")
                except Exception as e:
                    st.error(f"❌ 连接测试失败: {e}")
    else:
        st.warning("⚠️ 未连接到Milvus数据库")
        st.info("💡 请确保Milvus服务器正在运行，并检查网络连接")

def search_page():
    st.markdown("## 🔍 文本搜索")
    
    # 检查模型是否已加载
    if not st.session_state.model_loaded:
        st.warning("⚠️ 尚未加载嵌入模型！")
        st.info("💡 请先到 '🤖 嵌入模型管理' 页面加载模型，然后再进行搜索。")
        return
    
    # 检查MongoDB和Milvus是否已连接
    try:
        milvus_collection = get_milvus_collection(
            collection_name="text_vectors",
            dim=st.session_state.vectors.shape[1] if st.session_state.vectors is not None else 384
        )
        mongo_col = get_mongo_collection()
        vector_processor = st.session_state.components["vector_processor"]
    except Exception as e:
        st.error(f"❌ 初始化搜索组件失败: {e}")
        st.info("💡 请确保Milvus和MongoDB都已正确配置和连接")
        return

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
        search_button = st.button("🚀 开始搜索", type="primary", width="stretch")

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
                        st.plotly_chart(fig, width="stretch")
                        
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
    
    if st.button("🎯 执行聚类分析", type="primary", width="stretch"):
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
                            st.plotly_chart(fig, width="stretch")
                    
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
    
    # 连接状态
    st.markdown("### 🔗 连接状态")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🗄️ Milvus状态")
        if st.session_state.components['milvus_manager'].is_connected:
            persistence_status = st.session_state.components['milvus_manager'].verify_data_persistence()
            
            if persistence_status['status'] == 'success':
                st.success(f"✅ Milvus数据库：{persistence_status['num_entities']:,} 条记录")
            elif persistence_status['status'] == 'no_collection':
                st.info("📄 Milvus数据库：已连接，暂无数据")
            else:
                st.error(f"❌ Milvus数据库：{persistence_status['message']}")
        else:
            st.warning("⚠️ Milvus数据库：未连接")
    
    with col2:
        st.markdown("#### 🍃 MongoDB状态")
        mongodb_session_config = st.session_state.get("mongodb_config", {})
        if mongodb_session_config.get("connected", False):
            try:
                client = st.session_state.get("mongodb_client")
                if client:
                    db = client[mongodb_session_config["db_name"]]
                    col = db[mongodb_session_config["col_name"]]
                    count = col.estimated_document_count()
                    st.success(f"✅ MongoDB数据库：{count:,} 条记录")
                else:
                    st.warning("⚠️ MongoDB：连接对象丢失")
            except Exception as e:
                st.error(f"❌ MongoDB数据库：连接异常 - {e}")
        else:
            st.warning("⚠️ MongoDB数据库：未连接")
    
    # 模型信息
    st.markdown("### 🤖 向量化模型信息")
    model_info = st.session_state.components['vector_processor'].get_model_info()
    if model_info and st.session_state.model_loaded:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("模型名称", model_info.get('model_name', 'N/A'))
        with col2:
            st.metric("加载状态", "✅ 已加载" if st.session_state.model_loaded else "❌ 未加载")
        with col3:
            st.metric("向量维度", model_info.get('dimension', 'N/A'))
        
        with st.expander("🔍 模型详细信息"):
            st.json(model_info)
    else:
        st.info("📄 暂无已加载的模型")
    
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
