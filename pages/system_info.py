import streamlit as st
from components.utils import get_mongodb_data

def system_info_page():
    st.markdown("## ℹ️ 系统信息")

    # 配置信息
    st.markdown("### ⚙️ 配置信息")

    current_config = st.session_state.current_config

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🗄️ Milvus配置")
        milvus_config = st.session_state.milvus_config
        st.json(milvus_config)

    with col2:
        st.markdown("#### 🍃 MongoDB配置")
        mongodb_config = st.session_state.mongodb_config
        # 隐藏密码
        display_config = mongodb_config.copy()
        if display_config.get("password"):
            display_config["password"] = "***"
        st.json(display_config)

    # 连接状态
    st.markdown("### 🔗  连接状态")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🗄️ Milvus状态")
        if st.session_state.components['milvus_manager'].is_connected:
            persistence_status = st.session_state.components['milvus_manager'].verify_data_persistence()

            if persistence_status['status'] == 'success':
                st.success(f"✅ Milvus数据库：{persistence_status['num_entities']:,} 条记录")
            elif persistence_status['status'] == 'no_collection':
                st.info("🗄️ Milvus数据库：已连接，暂无数据")
            else:
                st.error(f"❌ Milvus数据库：{persistence_status['message']}")
        else:
            st.warning("⚠️ Milvus数据库：未连接")

    with col2:
        st.markdown("#### 🍃  MongoDB状态")
        # 新写法：实时查询显示
        mongo_data = st.session_state.get("mongo_data", {})
        if mongo_data.get("connected", False):
            st.success(f"✅ MongoDB数据库：{mongo_data['count']:,} 条记录")
        else:
            st.warning("⚠️ MongoDB数据库：未连接")
        if mongo_data.get("error"):
            st.error(f"❌ MongoDB数据库异常: {mongo_data['error']}")

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

        with st.expander("🤖 模型详细信息"):
            st.json(model_info)
    else:
        st.info("🤖 暂无已加载的模型")

    # 数据状态（完全基于MongoDB）
    st.markdown("### 🗄️ 数据状态")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        is_connected = mongo_data.get("connected", False)
        texts = mongo_data.get("texts", [])
        status = "✅ 已加载" if is_connected and texts else "❌ 未加载"
        st.metric("MongoDB数据状态", status)
    with col2:
        st.metric("文本数量", f"{len(texts):,}" if is_connected else "0")
    with col3:
        vectors = mongo_data.get("vectors")
        if is_connected and vectors is not None and hasattr(vectors, "size") and vectors.size > 0:
            vector_size = vectors.nbytes / 1024 / 1024
            st.metric("内存占用", f"{vector_size:.2f} MB")
        else:
            st.metric("内存占用", "0 MB")
    with col4:
        if is_connected and vectors is not None and hasattr(vectors, "shape") and vectors.size > 0:
            st.metric("向量维度", vectors.shape[1])
        else:
            st.metric("向量维度", "N/A")