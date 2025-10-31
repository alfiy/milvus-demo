import streamlit as st
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
if "model_config" not in st.session_state or not isinstance(st.session_state["model_config"], dict):
    st.session_state["model_config"] = config_manager.get_model_config()



# ===== 4. 自动加载模型（如配置开启） =====
def check_and_load_model_on_startup():
    raw_model_config = st.session_state.get('model_config', {})
    # 强制类型校正
    model_config = raw_model_config if isinstance(raw_model_config, dict) else {}
    
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
    from app_pages.homepage import home_page
    home_page()
elif page == "⚙️ 配置管理":
    from app_pages.config_management import config_management_page
    config_management_page()
elif page == "🤖 模型管理":
    from app_pages.model_manager import model_manager_page
    model_manager_page()
elif page == "📊 数据上传与处理":
    from app_pages.data_upload import data_upload_page
    data_upload_page()
elif page == "🍃 MongoDB管理":
    from app_pages.mongodb_config import mongodb_config_page
    mongodb_config_page()
elif page == "🗄️ Milvus管理":
    from app_pages.milvus_management import milvus_management_page
    milvus_management_page()
elif page == "🔍 搜索":
    from app_pages.search_page import search_page
    search_page()
elif page == "🎯 聚类分析":
    from app_pages.clustering_page import clustering_page
    clustering_page()
elif page == "ℹ️ 系统信息":
    from app_pages.system_info import system_info_page
    system_info_page()
