import streamlit as st
import json
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pymongo import MongoClient
from pymilvus import connections, Collection
from pathlib import Path
from components.vector_processor import VectorProcessor
from components.milvus_manager import MilvusManager
from components.clustering_analyzer import ClusteringAnalyzer
from components.search_engine import SearchEngine
from components.milvus_mongo_insert import milvus_mongo_upload, get_milvus_collection, get_mongo_collection, debug_collection_info
from components.config_manager import config_manager
from components.utils import get_mongodb_data, auto_connect_mongodb


# 加载全局样式
def local_css(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
# 加载 style.css（可根据实际路径调整）
local_css("style.css")


# 先初始化 components 字典（必须在其它组件之前）
if "components" not in st.session_state:
    st.session_state["components"] = {}

# 初始化 components 内部业务对象
default_components = {
    "milvus_manager": lambda: __import__("components.milvus_manager", fromlist=["MilvusManager"]).MilvusManager(),
    "vector_processor": lambda: __import__("components.vector_processor", fromlist=["VectorProcessor"]).VectorProcessor(),
    "search_engine": lambda: __import__("components.search_engine", fromlist=["SearchEngine"]).SearchEngine(),
    "clustering_analyzer": lambda: __import__("components.clustering_analyzer", fromlist=["ClusteringAnalyzer"]).ClusteringAnalyzer()
}

for comp_name, comp_builder in default_components.items():
    if comp_name not in st.session_state["components"]:
        st.session_state["components"][comp_name] = comp_builder()

# 初始化其余全局 session_state 变量
for key, default in [
    ("current_config", config_manager.load_config()),
    ("milvus_config", config_manager.get_milvus_config()),
    ("mongodb_config", config_manager.get_mongodb_config()),
    ("model_config", config_manager.get_model_config()),
    ("mongo_data", {}),
    ("model_loaded", False),
    ("data_loaded", False),
    # 其它自定义session_state变量加在这里
]:
    if key not in st.session_state:
        st.session_state[key] = default


# 页面配置
st.set_page_config(
    page_title="文本向量化与Milvus数据库解决方案",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化session state
@st.cache_resource
def init_components(milvus_config):
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

# 模型自动加载
def check_and_load_model_on_startup():

    if 'components' not in st.session_state:
        print("组件尚未初始化，跳过模型自动加载")
        return
    
    if 'vector_processor' not in st.session_state['components']:
        print("vector_processor 尚未初始化，跳过模型自动加载")
        return
    
    model_config = st.session_state.get('model_config', {})

    # 检查是否启用了自动加载
    if not model_config.get("auto_load", False):
        print("未启用模型自动加载")
        return
    
    last_used_model = model_config.get("last_used_model", "")
    if not last_used_model:
        print("没有上次使用的模型记录")
        return
    
    # 尝试加载模型
    try:
        vp = st.session_state['components']['vector_processor']
        vp.set_model_name(last_used_model)
        success, msg = vp.load_model()
        
        if success:
            st.session_state['model_loaded'] = True
            print(f"✅ 模型自动加载成功: {last_used_model}")
        else:
            st.session_state['model_loaded'] = False
            print(f"❌ 模型自动加载失败: {msg}")
    except Exception as e:
        st.session_state['model_loaded'] = False
        print(f"❌ 模型自动加载异常: {e}")



# 通用获取MongoDB集合对象的函数，自动复用连接对象
def get_shared_mongo_collection():

    # 连接状态检查
    if not st.session_state.get("mongodb_connected", False):
        st.error("❌ MongoDB未配置或未连接")
        return None
    
    config = st.session_state.get("mongodb_config", {})
    client = st.session_state.get("mongodb_client", None)

    if not client:
        st.error("❌ MongoDB客户端未初始化")
        return None
    
    try:
        db = client[config.get("db_name", "textdb")]
        col = db[config.get("col_name", "metadata")]
        _ = col.estimated_document_count()
        return col
    except Exception as e:
        st.error(f"❌ MongoDB连接已断开: {e}")
        st.session_state['mongodb_connected'] = False
        return None
  

def milvus_mongo_semantic_search(query, top_k, milvus_collection, mongo_col, vector_processor):
    """
    使用 Milvus + MongoDB 进行语义搜索 
    """
    try:
        # 1️⃣ 将查询文本向量化
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

def load_config(config_path='config.json'):
    # config.json 路径可指定
    if not os.path.exists(config_path):
        # 返回空或默认config结构
        return {
            "milvus": {
                "host": "localhost",
                "port": "19530",
                "user": "",
                "password": ""
            },
            "mongodb": {},
            "model": {}
        }
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config
    except Exception as e:
        # 加载失败，记录，并返回默认
        print(f"配置加载异常: {e}")
        return {
            "milvus": {
                "host": "localhost",
                "port": "19530",
                "user": "",
                "password": ""
            },
            "mongodb": {},
            "model": {}
        }

def init_session_state():
    # ---- 配置初始化 ----
    if 'current_config' not in st.session_state:
        st.session_state['current_config'] = load_config()

    if 'milvus_config' not in st.session_state:
        st.session_state['milvus_config'] = st.session_state['current_config'].get('milvus', {
            "host": "localhost", "port": "19530", "user": "", "password": ""
        })
    if 'mongodb_config' not in st.session_state:
        st.session_state['mongodb_config'] = st.session_state['current_config'].get('mongodb', {})
    if 'model_config' not in st.session_state:
        st.session_state['model_config'] = st.session_state['current_config'].get('model', {})
    if 'mongo_data' not in st.session_state:
        # 若没有 get_mongodb_data 函数，临时设置为空字典
        try:
            from components.utils import get_mongodb_data
            st.session_state['mongo_data'] = get_mongodb_data(st.session_state['mongodb_config'])
        except Exception:
            st.session_state['mongo_data'] = {}

    # ---- 组件初始化 ----
    if 'components' not in st.session_state:
        st.session_state['components'] = {
            'vector_processor': VectorProcessor(),
            'milvus_manager': MilvusManager(
                host=st.session_state['milvus_config'].get("host", "localhost"),
                port=st.session_state['milvus_config'].get("port", "19530"),
                user=st.session_state['milvus_config'].get("user", ""),
                password=st.session_state['milvus_config'].get("password", "")
            ),
            'clustering_analyzer': ClusteringAnalyzer(),
            'search_engine': SearchEngine(),
        }

    # 其他全局状态变量
    default_vars = {
        'data_loaded': False,
        'vectors': None,
        'texts': [],
        'metadata': [],
        'model_loaded': False,
        # 你可继续添加需要的变量
    }
    for var_name, default_value in default_vars.items():
        if var_name not in st.session_state:
            st.session_state[var_name] = default_value

    # 自动加载模型 确保 components 已经初始化后再尝试加载模型
    if not st.session_state.get('model_loaded', False):
        try:
            check_and_load_model_on_startup()
        except Exception as e:
            # 记录错误但不中断应用启动
            print(f"模型自动加载失败: {e}")

    # 自动连接MongoDB
    try:
        auto_connect_mongodb()
    except Exception as e:
        print(f"MongoDB自动连接失败: {e}")

# -------rebuild project start --------


# 侧边栏导航
with st.sidebar:
    st.title("导航")
    page = st.radio(
        "选择页面",
        ("🏠 首页", "⚙️ 配置管理", "🤖 模型管理", "📊 数据上传与处理", "🍃 MongoDB管理", "🗄️ Milvus管理", "🔍 搜索", "🎯 聚类分析","ℹ️ 系统信息")
    )

# 页面路由
if page == "🏠 首页":
    from pages.homepage import home_page
    home_page()
elif page == "⚙️ 配置管理":
    from pages.config_management import config_management_page
    config_management_page()
elif page == "🤖 模型管理":
    from pages.model_manager import model_manager_page
    model_manager_page()
elif page == "📊 数据上传与处理":
    from pages.data_upload import data_upload_page
    data_upload_page()
elif page == "🍃 MongoDB管理":
    from pages.mongodb_config import mongodb_config_page
    mongodb_config_page()
elif page == "🗄️ Milvus管理":
    from pages.milvus_management import milvus_management_page
    milvus_management_page()
elif page == "🔍 搜索":
    from pages.search_page import search_page
    search_page()
elif page == "🎯 聚类分析":
    from pages.clustering_page import clustering_page
    clustering_page()
elif page == "ℹ️ 系统信息":
    from pages.system_info import system_info_page
    system_info_page()

# ---------end ------------------------

# def main():
#     init_session_state()
    
#     # 获取配置
#     mongodb_config = st.session_state.get("mongodb_config", {})

#     # 主标题
#     st.markdown("""
#     <div class="main-header">
#         <h1>🚀 文本向量化与Milvus数据库解决方案</h1>
#         <p>支持百万级数据的向量化、存储、搜索和聚类分析 - 配置持久化版本</p>
#     </div>
#     """, unsafe_allow_html=True)
    
#     # 侧边栏导航
#     with st.sidebar:
#         st.markdown("### 🧭 功能菜单")
        
#         # 显示当前状态
#         if st.session_state.get('data_loaded', False):
#             st.success(f"✅ 已加载 {len(st.session_state.texts)} 条数据")
#         else:
#             st.info("💡 请先上传数据")
        
#         # 模型加载状态
#         if st.session_state.get('model_loaded', False):
#             model_config =  config_manager.get_model_config()
#             current_model = model_config.get("last_used_model", "")
#             st.success("🤖 模型已加载")
#             if current_model:
#                 st.caption(f"📋 {current_model}")
#         else:
#             st.warning("⚠️ 模型未加载")
        
#         # MongoDB连接状态
#         mongodb_config = st.session_state.get("mongodb_config", {})
#         if st.session_state.get("mongodb_connected", False):
#             st.success("🍃 MongoDB已连接")
#             st.caption(f"🔗 {mongodb_config.get('host', '')}:{mongodb_config.get('port', '')}")
#         else:
#             st.warning("⚠️ MongoDB未连接")
#             error_msg = st.session_state.get("mongodb_connect_error")
#             if error_msg:
#                 st.caption(f"连接异常信息：{error_msg}")
        
#         # Milvus连接状态和数据持久化验证
#         if st.session_state.components['milvus_manager'].is_connected:
#             st.success("🗄️ Milvus已连接")
            
#             # 显示连接信息
#             conn_info = st.session_state.components['milvus_manager'].get_connection_info()
#             st.caption(f"🔗 {conn_info['host']}:{conn_info['port']}")
            
#             # 验证数据持久化状态
#             persistence_status = st.session_state.components['milvus_manager'].verify_data_persistence()
#             if persistence_status['status'] == 'success':
#                 st.success(f"💾 持久化数据: {persistence_status['num_entities']:,} 条")
#             elif persistence_status['status'] == 'no_collection':
#                 st.info("📄 暂无持久化集合")
#             else:
#                 st.warning("⚠️ 数据状态未知")
#         else:
#             st.warning("⚠️ Milvus未连接")
        
#         st.markdown("---")
        
#         page = st.selectbox(
#             "选择功能模块",
#             ["🏠 首页概览", "⚙️ 系统配置管理", "🤖 嵌入模型管理", "📊 数据上传与处理", "🗄️ Milvus数据库管理","🍃 MongoDB配置管理", "🔍 文本搜索", "🎯 聚类分析", "ℹ️ 系统信息"],
#             index=0
#         )

#     # 主要内容区域
#     if page == "🏠 首页概览":
#         home_page()
#     elif page == "⚙️ 系统配置管理":
#         config_management_page()
#     elif page == "🤖 嵌入模型管理":
#         model_manager_page()
#     elif page == "📊 数据上传与处理":
#         data_upload_page()
#     elif page == "🗄️ Milvus数据库管理":
#         milvus_management_page()
#     elif page == "🍃 MongoDB配置管理":
#         mongodb_config_page()
#     elif page == "🔍 文本搜索":
#         search_page()
#     elif page == "🎯 聚类分析":
#         clustering_page()
#     elif page == "ℹ️ 系统信息":
#         system_info_page()


# def data_upload_page():
#     st.markdown("## 📊 数据上传与处理")
    
#     # 检查模型加载状态
#     if not st.session_state.model_loaded:
#         st.warning("⚠️ 尚未加载嵌入模型！")
#         st.info("💡 请先到 '🤖 嵌入模型管理' 页面加载模型，然后再回到此页面进行数据处理。")
        
#         col1, col2 = st.columns([3, 1])
#         with col1:
#             st.markdown("""
#             **为什么需要先加载模型？**
#             - 文本向量化需要使用嵌入模型
#             - 模型加载后可以处理任何文本数据
#             - 统一的模型管理确保配置一致性
#             """)
#         with col2:
#             if st.button("🚀 前往模型管理", type="primary"):
#                 st.switch_page("🤖 嵌入模型管理")
#         return
    
#     # 显示当前使用的模型
#     model_config =  st.session_state.model_config
#     current_model = model_config.get("last_used_model", "")
    
#     st.markdown("### 🤖 当前模型状态")
#     col1, col2 = st.columns([3, 1])
#     with col1:
#         st.success(f"✅ 已加载模型: **{current_model}**")
#         vp = st.session_state.components['vector_processor']
#         model_info = vp.get_model_info()
#         if model_info:
#             st.info(f"🔢 向量维度: {model_info.get('dimension', 'N/A')}")
#     with col2:
#         if st.button("🔄 切换模型"):
#             # 这里可以添加快速切换模型的功能，或者跳转到模型管理页面
#             st.info("💡 请到 '🤖 嵌入模型管理' 页面切换模型")

#     st.markdown("---")
    
#     # 数据上传选项
#     upload_method = st.radio(
#         "选择数据输入方式",
#         ["📁 上传JSON文件", "✏️ 手动输入JSON数据", "🎯 使用示例数据"],
#         horizontal=True
#     )
    
#     json_data = None
    
#     if upload_method == "📁 上传JSON文件":
#         uploaded_file = st.file_uploader(
#             "选择JSON文件",
#             type=['json', 'jsonl', 'txt'],
#             help="支持JSON、JSONL格式文件。JSON格式：[{\"text1\":\"内容\"}]，JSONL格式：每行一个JSON对象"
#         )
        
#         if uploaded_file is not None:
#             try:
#                 # 读取文件内容
#                 file_content = uploaded_file.read().decode('utf-8')
                
#                 # 使用改进的JSON解析方法
#                 json_data = st.session_state.components['vector_processor'].parse_json_file(file_content)
                
#                 if not isinstance(json_data, list):
#                     json_data = [json_data]
                
#                 st.success(f"✅ 成功加载 {len(json_data)} 条数据")
                
#                 # 显示文件信息
#                 file_size = uploaded_file.size / 1024 / 1024
#                 st.info(f"📁 文件大小: {file_size:.2f} MB")
                
#                 # 显示数据格式检测结果
#                 sample_item = json_data[0] if json_data else {}
#                 if isinstance(sample_item, dict):
#                     keys = list(sample_item.keys())
#                     keys_display = ', '.join(keys[:5])
#                     if len(keys) > 5:
#                         keys_display += '...'
#                     st.info(f"🔍 检测到字段: {keys_display}")
                
#             except Exception as e:
#                 st.error(f"❌ 文件加载失败: {e}")
#                 st.markdown("""
#                 **支持的文件格式：**
#                 1. **标准JSON数组**: `[{"text1":"内容1"}, {"text1":"内容2"}]`
#                 2. **JSONL格式**: 每行一个JSON对象
#                    ```
#                    {"text1":"内容1"}
#                    {"text1":"内容2"}
#                    ```
#                 3. **单个JSON对象**: `{"text1":"内容"}`
#                 """)
    
#     elif upload_method == "✏️ 手动输入JSON数据":
#         json_text = st.text_area(
#             "输入JSON数据",
#             height=200,
#             placeholder='[{"text1":"半生长以客为家，罢直初来瀚海槎。始信人间行不尽，天涯更复有天涯。"}]',
#             help="请输入有效的JSON格式数据"
#         )
        
#         if json_text.strip():
#             try:
#                 json_data = st.session_state.components['vector_processor'].parse_json_file(json_text)
#                 if not isinstance(json_data, list):
#                     json_data = [json_data]
#                 st.success(f"✅ 成功解析 {len(json_data)} 条数据")
#             except Exception as e:
#                 st.error(f"❌ JSON解析失败: {e}")
    
#     elif upload_method == "🎯 使用示例数据":
#         sample_data = [
#             {"text1": "半生长以客为家，罢直初来瀚海槎。始信人间行不尽，天涯更复有天涯。"},
#             {"text1": "春风得意马蹄疾，一日看尽长安花。"},
#             {"text1": "山重水复疑无路，柳暗花明又一村。"},
#             {"text1": "海内存知己，天涯若比邻。"},
#             {"text1": "落红不是无情物，化作春泥更护花。"},
#             {"text1": "会当凌绝顶，一览众山小。"},
#             {"text1": "采菊东篱下，悠然见南山。"},
#             {"text1": "明月几时有，把酒问青天。"}
#         ]
#         json_data = sample_data
#         st.info(f"🎯 使用示例数据，共 {len(json_data)} 条古诗词")
    
#     # 数据预览和处理
#     if json_data:
#         st.markdown("### 📋 数据预览")
        
#         # 显示数据统计
#         col1, col2, col3 = st.columns(3)
#         with col1:
#             st.metric("数据条数", len(json_data))
#         with col2:
#             total_chars = sum(len(str(item)) for item in json_data)
#             st.metric("总字符数", f"{total_chars:,}")
#         with col3:
#             avg_length = total_chars / len(json_data) if json_data else 0
#             st.metric("平均长度", f"{avg_length:.1f}")
        
#         # 数据表格预览
#         df_preview = pd.DataFrame(json_data[:10])
#         st.dataframe(df_preview, use_container_width=True)
        
#         if len(json_data) > 10:
#             st.info(f"显示前10条数据，总共{len(json_data)}条")
        
#         # 向量化处理
#         st.markdown("### 🚀 向量化处理")
        
#         col1, col2 = st.columns([3, 1])
#         with col1:
#             st.info("点击下方按钮开始文本向量化处理，处理后的数据可以保存到Milvus数据库中永久存储")
#         with col2:
#             process_button = st.button("🚀 开始向量化处理并持久化", type="primary")
        
#         if process_button:
#             progress_bar = st.progress(0)
#             status_text = st.empty()
            
#             try:
#                 status_text.text("📊 正在处理文本数据...")
#                 progress_bar.progress(30)
                
#                 texts, vectors, metadata = st.session_state.components['vector_processor'].process_json_data(json_data)
#                 progress_bar.progress(70)
                
#                 if len(texts) > 0:
#                     # 保存到session state
#                     st.session_state.texts = texts
#                     st.session_state.vectors = vectors
#                     st.session_state.metadata = metadata
#                     st.session_state.data_loaded = True
                    
#                     # ------ 新增：自动批量插入Milvus和MongoDB ------
#                     embedding_dim = vectors.shape[1]
#                     status_text.text("💾 正在批量插入 Milvus & MongoDB ...")
                    
#                     try:
#                         inserted_ids = milvus_mongo_upload(texts, vectors, metadata, milvus_dim=embedding_dim)
#                         progress_bar.progress(100)
#                         status_text.text(f"✅ 向量化及持久化完成！已插入 {len(inserted_ids)} 条数据。")
#                         st.success(f"🎉 向量化和持久化完成！成功处理并写入 {len(inserted_ids)} 条文本数据。")
#                     except Exception as e:
#                         progress_bar.progress(100)
#                         status_text.text("⚠️ 向量化完成，但持久化失败")
#                         st.warning(f"⚠️ 向量化完成，但数据持久化失败: {e}")
#                         st.info("💡 数据已保存到内存中，可以进行搜索和聚类分析。要启用持久化，请检查Milvus和MongoDB连接。")

#                     # 搜索引擎设置
#                     st.session_state.components['search_engine'].load_data(vectors, texts, metadata)
#                     st.session_state.components['search_engine'].set_vector_processor(st.session_state.components['vector_processor'])
                    
#                     # 设置聚类分析器数据
#                     st.session_state.components['clustering_analyzer'].load_data(vectors, texts, metadata)
                    
#                     st.success(f"🎉 向量化完成！成功处理了 {len(texts)} 条文本")
                    
#                     # 显示处理结果统计
#                     st.markdown("### 📊 处理结果")
#                     col1, col2, col3, col4 = st.columns(4)
#                     with col1:
#                         st.metric("文本数量", len(texts))
#                     with col2:
#                         st.metric("向量维度", vectors.shape[1])
#                     with col3:
#                         st.metric("数据大小", f"{vectors.nbytes / 1024 / 1024:.2f} MB")
#                     with col4:
#                         st.metric("处理状态", "✅ 完成")
                    
#                     # 显示向量化样本
#                     with st.expander("🔍 查看向量化样本", expanded=False):
#                         sample_idx = 0
#                         st.write(f"**原文本:** {texts[sample_idx]}")
#                         st.write(f"**向量维度:** {len(vectors[sample_idx])}")
#                         st.write(f"**向量前10维:** {vectors[sample_idx][:10].tolist()}")
                        
#                 else:
#                     st.error("❌ 未找到有效的文本数据")
                    
#             except Exception as e:
#                 st.error(f"❌ 向量化处理失败: {e}")
#                 st.exception(e)
            
#             finally:
#                 progress_bar.empty()
#                 status_text.empty()




# if __name__ == "__main__": 
#     main()
