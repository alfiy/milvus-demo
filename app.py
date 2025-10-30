import streamlit as st
import os
from components.vector_processor import VectorProcessor
from components.milvus_manager import MilvusManager
from components.clustering_analyzer import ClusteringAnalyzer
from components.search_engine import SearchEngine
from components.config_manager import config_manager
from components.utils import auto_connect_mongodb

# ===== 1. 全局样式加载 =====
def local_css(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
local_css("style.css")

# ===== 2. 全局变量初始化 =====
initials = [
    ("current_config", config_manager.load_config()),
    ("components", {}),
    ("milvus_config", config_manager.get_milvus_config()),
    ("mongodb_config", config_manager.get_mongodb_config()),
    ("model_config", config_manager.get_model_config()),
    ("mongo_data", {}),
    ("model_loaded", False),
    ("data_loaded", False),
    ("vectors", None),
    ("texts", []),
    ("metadata", []),
    ("mongodb_connected", False),
    ("mongodb_connect_error", None),
    ("mongodb_client", None),
    # 如有新增全局变量，继续补充
]
for key, default in initials:
    if key not in st.session_state:
        st.session_state[key] = default

# ===== 3. 主要组件对象初始化 =====
if "milvus_manager" not in st.session_state["components"]:
    st.session_state["components"]["milvus_manager"] = MilvusManager(
        host=st.session_state["milvus_config"].get("host", "localhost"),
        port=st.session_state["milvus_config"].get("port", "19530"),
        user=st.session_state["milvus_config"].get("user", ""),
        password=st.session_state["milvus_config"].get("password", "")
    )
if "vector_processor" not in st.session_state["components"]:
    st.session_state["components"]["vector_processor"] = VectorProcessor()
if "search_engine" not in st.session_state["components"]:
    st.session_state["components"]["search_engine"] = SearchEngine()
if "clustering_analyzer" not in st.session_state["components"]:
    st.session_state["components"]["clustering_analyzer"] = ClusteringAnalyzer()

# ===== 4. 自动加载模型（如配置开启） =====
def check_and_load_model_on_startup():
    model_config = st.session_state.get('model_config', {})
    if not model_config.get("auto_load", False):
        return
    last_used_model = model_config.get("last_used_model", "")
    if not last_used_model:
        return
    vp = st.session_state["components"]["vector_processor"]
    try:
        vp.set_model_name(last_used_model)
        success, _ = vp.load_model()
        st.session_state["model_loaded"] = success
    except Exception:
        st.session_state["model_loaded"] = False

if not st.session_state.get("model_loaded", False):
    check_and_load_model_on_startup()

# ===== 5. 自动连接 MongoDB（严格根据配置） =====
mongodb_config = st.session_state.get("mongodb_config", config_manager.get_mongodb_config())
if mongodb_config.get("auto_connect", False) and not st.session_state.get("mongodb_connected", False):
    try:
        connected, error_msg, client = auto_connect_mongodb(mongodb_config)
        st.session_state["mongodb_connected"] = connected
        st.session_state["mongodb_connect_error"] = error_msg
        st.session_state["mongodb_client"] = client if connected else None
    except Exception as e:
        st.session_state["mongodb_connected"] = False
        st.session_state["mongodb_connect_error"] = str(e)
        st.session_state["mongodb_client"] = None

# ===== 6. 页面全局配置 =====
st.set_page_config(
    page_title="文本向量化与Milvus数据库解决方案",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== 7. 侧边栏导航 =====
with st.sidebar:
    st.title("导航")
    page = st.radio(
        "选择页面",
        ("🏠 首页", "⚙️ 配置管理", "🤖 模型管理", "📊 数据上传与处理", "🍃 MongoDB管理", "🗄️ Milvus管理", "🔍 搜索", "🎯 聚类分析","ℹ️ 系统信息")
    )

# ===== 8. 页面路由 =====
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



# import streamlit as st
# import json
# import os
# from components.vector_processor import VectorProcessor
# from components.milvus_manager import MilvusManager
# from components.clustering_analyzer import ClusteringAnalyzer
# from components.search_engine import SearchEngine
# from components.config_manager import config_manager
# from components.utils import auto_connect_mongodb


# # 加载全局样式
# def local_css(file_name):
#     with open(file_name, "r", encoding="utf-8") as f:
#         st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
# # 加载 style.css（可根据实际路径调整）
# local_css("style.css")


# # 先初始化 components 字典（必须在其它组件之前）
# if "components" not in st.session_state:
#     st.session_state["components"] = {}

# # 初始化 components 内部业务对象
# default_components = {
#     "milvus_manager": lambda: __import__("components.milvus_manager", fromlist=["MilvusManager"]).MilvusManager(),
#     "vector_processor": lambda: __import__("components.vector_processor", fromlist=["VectorProcessor"]).VectorProcessor(),
#     "search_engine": lambda: __import__("components.search_engine", fromlist=["SearchEngine"]).SearchEngine(),
#     "clustering_analyzer": lambda: __import__("components.clustering_analyzer", fromlist=["ClusteringAnalyzer"]).ClusteringAnalyzer()
# }

# for comp_name, comp_builder in default_components.items():
#     if comp_name not in st.session_state["components"]:
#         st.session_state["components"][comp_name] = comp_builder()

# # 初始化其余全局 session_state 变量
# for key, default in [
#     ("current_config", config_manager.load_config()),
#     ("milvus_config", config_manager.get_milvus_config()),
#     ("mongodb_config", config_manager.get_mongodb_config()),
#     ("model_config", config_manager.get_model_config()),
#     ("mongo_data", {}),
#     ("model_loaded", False),
#     ("data_loaded", False),
#     # 其它自定义session_state变量加在这里
# ]:
#     if key not in st.session_state:
#         st.session_state[key] = default


# # 页面配置
# st.set_page_config(
#     page_title="文本向量化与Milvus数据库解决方案",
#     page_icon="🚀",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # 初始化session state
# @st.cache_resource
# def init_components(milvus_config):
#     return {
#         'vector_processor': VectorProcessor(),
#         'milvus_manager': MilvusManager(
#             host=milvus_config.get("host", "localhost"),
#             port=milvus_config.get("port", "19530"),
#             user=milvus_config.get("user", ""),
#             password=milvus_config.get("password", "")
#         ),
#         'clustering_analyzer': ClusteringAnalyzer(),
#         'search_engine': SearchEngine()
#     }

# # 模型自动加载
# def check_and_load_model_on_startup():

#     if 'components' not in st.session_state:
#         print("组件尚未初始化，跳过模型自动加载")
#         return
    
#     if 'vector_processor' not in st.session_state['components']:
#         print("vector_processor 尚未初始化，跳过模型自动加载")
#         return
    
#     model_config = st.session_state.get('model_config', {})

#     # 检查是否启用了自动加载
#     if not model_config.get("auto_load", False):
#         print("未启用模型自动加载")
#         return
    
#     last_used_model = model_config.get("last_used_model", "")
#     if not last_used_model:
#         print("没有上次使用的模型记录")
#         return
    
#     # 尝试加载模型
#     try:
#         vp = st.session_state['components']['vector_processor']
#         vp.set_model_name(last_used_model)
#         success, msg = vp.load_model()
        
#         if success:
#             st.session_state['model_loaded'] = True
#             print(f"✅ 模型自动加载成功: {last_used_model}")
#         else:
#             st.session_state['model_loaded'] = False
#             print(f"❌ 模型自动加载失败: {msg}")
#     except Exception as e:
#         st.session_state['model_loaded'] = False
#         print(f"❌ 模型自动加载异常: {e}")



# # 通用获取MongoDB集合对象的函数，自动复用连接对象
# def get_shared_mongo_collection():

#     # 连接状态检查
#     if not st.session_state.get("mongodb_connected", False):
#         st.error("❌ MongoDB未配置或未连接")
#         return None
    
#     config = st.session_state.get("mongodb_config", {})
#     client = st.session_state.get("mongodb_client", None)

#     if not client:
#         st.error("❌ MongoDB客户端未初始化")
#         return None
    
#     try:
#         db = client[config.get("db_name", "textdb")]
#         col = db[config.get("col_name", "metadata")]
#         _ = col.estimated_document_count()
#         return col
#     except Exception as e:
#         st.error(f"❌ MongoDB连接已断开: {e}")
#         st.session_state['mongodb_connected'] = False
#         return None
  

# def milvus_mongo_semantic_search(query, top_k, milvus_collection, mongo_col, vector_processor):
#     """
#     使用 Milvus + MongoDB 进行语义搜索 
#     """
#     try:
#         # 1️⃣ 将查询文本向量化
#         query_vector = vector_processor.encode([query])[0]

#         # 2️⃣ 检查集合是否存在且已连接
#         if not milvus_collection:
#             st.error("❌ Milvus 集合未初始化，请先创建集合或导入数据")
#             return []

#         # 3️⃣ 执行 Milvus 搜索
#         search_params = {
#             "metric_type": "COSINE",
#             "params": {"nprobe": 10}
#         }

#         results = milvus_collection.search(
#             data=[query_vector.tolist()],
#             anns_field="vector",           # ✅ 字段名改为 "vector"
#             param=search_params,
#             limit=top_k,
#             output_fields=["text", "metadata"]
#         )

#         # 4️⃣ 整理 Milvus 搜索结果
#         ids, scores = [], []
#         for hits in results:
#             for hit in hits:
#                 ids.append(hit.id)
#                 scores.append(float(hit.distance))

#         # 5️⃣ 查询 MongoDB 获取元数据
#         docs = list(mongo_col.find({"_id": {"$in": ids}}))
#         id2doc = {str(doc["_id"]): doc for doc in docs}

#         combined = []
#         for id_, score in zip(ids, scores):
#             doc = id2doc.get(str(id_), {})
#             combined.append({
#                 "id": id_,
#                 "score": score,
#                 "text": doc.get("text", ""),
#                 "metadata": doc.get("metadata", {}),
#             })

#         return combined

#     except Exception as e:
#         st.error(f"❌ 搜索失败: {e}")
#         return []

# def load_config(config_path='config.json'):
#     # config.json 路径可指定
#     if not os.path.exists(config_path):
#         # 返回空或默认config结构
#         return {
#             "milvus": {
#                 "host": "localhost",
#                 "port": "19530",
#                 "user": "",
#                 "password": ""
#             },
#             "mongodb": {},
#             "model": {}
#         }
#     try:
#         with open(config_path, "r", encoding="utf-8") as f:
#             config = json.load(f)
#         return config
#     except Exception as e:
#         # 加载失败，记录，并返回默认
#         print(f"配置加载异常: {e}")
#         return {
#             "milvus": {
#                 "host": "localhost",
#                 "port": "19530",
#                 "user": "",
#                 "password": ""
#             },
#             "mongodb": {},
#             "model": {}
#         }

# def init_session_state():
#     # ---- 配置初始化 ----
#     if 'current_config' not in st.session_state:
#         st.session_state['current_config'] = load_config()

#     if 'milvus_config' not in st.session_state:
#         st.session_state['milvus_config'] = st.session_state['current_config'].get('milvus', {
#             "host": "localhost", "port": "19530", "user": "", "password": ""
#         })
#     if 'mongodb_config' not in st.session_state:
#         st.session_state['mongodb_config'] = st.session_state['current_config'].get('mongodb', {})
#     if 'model_config' not in st.session_state:
#         st.session_state['model_config'] = st.session_state['current_config'].get('model', {})
#     if 'mongo_data' not in st.session_state:
#         # 若没有 get_mongodb_data 函数，临时设置为空字典
#         try:
#             from components.utils import get_mongodb_data
#             st.session_state['mongo_data'] = get_mongodb_data(st.session_state['mongodb_config'])
#         except Exception:
#             st.session_state['mongo_data'] = {}

#     # ---- 组件初始化 ----
#     if 'components' not in st.session_state:
#         st.session_state['components'] = {
#             'vector_processor': VectorProcessor(),
#             'milvus_manager': MilvusManager(
#                 host=st.session_state['milvus_config'].get("host", "localhost"),
#                 port=st.session_state['milvus_config'].get("port", "19530"),
#                 user=st.session_state['milvus_config'].get("user", ""),
#                 password=st.session_state['milvus_config'].get("password", "")
#             ),
#             'clustering_analyzer': ClusteringAnalyzer(),
#             'search_engine': SearchEngine(),
#         }

#     # 其他全局状态变量
#     default_vars = {
#         'data_loaded': False,
#         'vectors': None,
#         'texts': [],
#         'metadata': [],
#         'model_loaded': False,
#         # 你可继续添加需要的变量
#     }
#     for var_name, default_value in default_vars.items():
#         if var_name not in st.session_state:
#             st.session_state[var_name] = default_value

#     # 自动加载模型 确保 components 已经初始化后再尝试加载模型
#     if not st.session_state.get('model_loaded', False):
#         try:
#             check_and_load_model_on_startup()
#         except Exception as e:
#             # 记录错误但不中断应用启动
#             print(f"模型自动加载失败: {e}")

#     # 自动连接MongoDB
#     try:
#         auto_connect_mongodb()
#     except Exception as e:
#         print(f"MongoDB自动连接失败: {e}")



# # 侧边栏导航
# with st.sidebar:
#     st.title("导航")
#     page = st.radio(
#         "选择页面",
#         ("🏠 首页", "⚙️ 配置管理", "🤖 模型管理", "📊 数据上传与处理", "🍃 MongoDB管理", "🗄️ Milvus管理", "🔍 搜索", "🎯 聚类分析","ℹ️ 系统信息")
#     )

# # 页面路由
# if page == "🏠 首页":
#     from pages.homepage import home_page
#     home_page()
# elif page == "⚙️ 配置管理":
#     from pages.config_management import config_management_page
#     config_management_page()
# elif page == "🤖 模型管理":
#     from pages.model_manager import model_manager_page
#     model_manager_page()
# elif page == "📊 数据上传与处理":
#     from pages.data_upload import data_upload_page
#     data_upload_page()
# elif page == "🍃 MongoDB管理":
#     from pages.mongodb_config import mongodb_config_page
#     mongodb_config_page()
# elif page == "🗄️ Milvus管理":
#     from pages.milvus_management import milvus_management_page
#     milvus_management_page()
# elif page == "🔍 搜索":
#     from pages.search_page import search_page
#     search_page()
# elif page == "🎯 聚类分析":
#     from pages.clustering_page import clustering_page
#     clustering_page()
# elif page == "ℹ️ 系统信息":
#     from pages.system_info import system_info_page
#     system_info_page()

