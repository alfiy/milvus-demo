import streamlit as st
from components.utils import get_mongodb_stats


def system_info_page():
    st.markdown("## ℹ️ 系统信息")

    # 配置信息
    st.markdown("### ⚙️ 配置信息")

    current_config = st.session_state.get("current_config", {})
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🗄️ Milvus配置")
        milvus_config = st.session_state.get("milvus_config", {})
        st.json(milvus_config)
    with col2:
        st.markdown("#### 🍃 MongoDB配置")
        mongodb_config = st.session_state.get("mongodb_config", {})
        display_config = mongodb_config.copy()
        if display_config.get("password"):
            display_config["password"] = "***"
        st.json(display_config)

    # 连接状态
    st.markdown("### 🔗 连接状态")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🗄️ Milvus状态")
        milvus_manager = st.session_state["components"].get("milvus_manager")
        if milvus_manager and milvus_manager.is_connected:
            persistence_status = milvus_manager.verify_data_persistence()
            if persistence_status['status'] == 'success':
                st.success(f"✅ Milvus数据库：{persistence_status['num_entities']:,} 条记录")
            elif persistence_status['status'] == 'no_collection':
                st.info("🗄️ Milvus数据库：已连接，暂无数据")
            else:
                st.error(f"❌ Milvus数据库：{persistence_status['message']}")
        else:
            st.warning("⚠️ Milvus数据库：未连接")

    # 统一统计MongoDB业务状态
    mongodb_client = st.session_state.get("mongodb_client", None)
    mongodb_config = st.session_state.get("mongodb_config", {})
    mongo_stats = get_mongodb_stats(mongodb_client, mongodb_config)

    with col2:
        st.markdown("#### 🍃 MongoDB状态")
        if mongo_stats["connected"]:
            st.success(f"✅ MongoDB数据库已连接：{mongo_stats['count']:,} 条记录")
        else:
            st.warning("⚠️ MongoDB数据库：未连接")
            error_msg = mongo_stats.get("error") or st.session_state.get("mongodb_connect_error")
            if error_msg:
                st.error(f"❌ MongoDB数据库连接异常: {error_msg}")

    # 模型信息
    st.markdown("### 🤖 向量化模型信息")
    model_info = st.session_state["components"].get("vector_processor").get_model_info()
    if model_info and st.session_state.get("model_loaded", False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("模型名称", model_info.get('model_name', 'N/A'))
        with col2:
            st.metric("加载状态", "✅ 已加载")
        with col3:
            st.metric("向量维度", model_info.get('dimension', 'N/A'))
        with st.expander("🤖 模型详细信息"):
            st.json(model_info)
    else:
        st.info("🤖 暂无已加载的模型")

    # 数据状态（完全基于MongoDB）
    st.markdown("### 📊 数据状态")
    col1, col2, col3, col4 = st.columns(4)
    status = "✅ 已加载" if mongo_stats["connected"] and mongo_stats["count"] > 0 else "❌ 未加载"
    with col1:
        st.metric("MongoDB数据状态", status)
    with col2:
        st.metric("文本数量", f"{mongo_stats['count']:,}" if mongo_stats["connected"] else "0")
    with col3:
        st.metric("内存占用", f"{mongo_stats['vector_size']:.2f} MB" if mongo_stats["vector_size"] > 0 else "0 MB")
    with col4:
        st.metric("向量维度", mongo_stats["vector_info"])
    if mongo_stats.get("error"):
        st.error(f"❌ MongoDB数据异常: {mongo_stats['error']}")
