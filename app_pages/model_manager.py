import streamlit as st
from components.vector_processor import VectorProcessor
from components.config_manager import config_manager


def model_manager_page():
    st.markdown("## 🤖 嵌入模型管理")

    # 确保 session_state 里有 VectorProcessor 实例
    if 'vector_processor' not in st.session_state['components']:
        st.session_state['components']['vector_processor'] = VectorProcessor()
    vp = st.session_state['components']['vector_processor']

    # 1. 读取模型配置信息（来自config.json）
    model_config = config_manager.get_model_config()
    current_loaded_model = model_config.get("last_used_model", "") if st.session_state.get('model_loaded', False) else ""

    # 2. 当前模型状态区
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
        if current_loaded_model:
            model_info = vp.get_model_info()
            if model_info:
                st.markdown(f"""
                <div class="model-card">
                    <h4>🤖  模型详情</h4>
                    <p><strong>向量维度:</strong> {model_info.get('dimension', 'N/A')}</p>
                    <p><strong>模型类型:</strong> {model_info.get('model_type', 'Sentence Transformer')}</p>
                </div>
                """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### ⚙️ 模型选择与管理")

    # 3. 添加新模型
    st.markdown("#### 🤖 添加新模型")
    new_model_name = st.text_input(
        "输入 HuggingFace 模型名并下载到本地", "",
        help="如：sentence-transformers/paraphrase-MiniLM-L6-v2"
    )
    if st.button("下载模型"):
        if new_model_name:
            with st.spinner("正在下载模型..."):
                ok, msg = vp.download_model(new_model_name, log_callback=lambda l: st.info(l))
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

    available_models = vp.scan_local_models()
    st.markdown("#### ⚙️ 选择并加载模型")
    last_used_model = model_config.get("last_used_model", "")
    default_index = 0
    if last_used_model and last_used_model in available_models:
        default_index = available_models.index(last_used_model)

    selected_model = st.selectbox(
        "选择要加载的嵌入模型",
        options=available_models,
        index=default_index if available_models else 0,
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
        load_button = st.button("⚙️ 加载模型", type="primary")
    with col3:
        if current_loaded_model:
            unload_button = st.button("⚙️ 卸载模型")
        else:
            unload_button = False

    # 4. 加载模型（更新session_state与config）
    if load_button:
        with st.spinner("正在加载模型..."):
            vp.model_name = selected_model
            ok, msg = vp.load_model()
            if ok:
                st.session_state['model_loaded'] = True
                st.session_state['model_config'] = config_manager.update_model_config(selected_model, auto_load)
                st.success("✅ 模型加载成功")
                st.rerun()
            else:
                st.session_state['model_loaded'] = False
                st.error(f"❌ 模型加载失败: {msg}")

    # 5. 卸载模型（更新session_state与config）
    if unload_button:
        st.session_state['model_loaded'] = False
        config_manager.update_model_config("", auto_load)
        st.success("✅ 模型已卸载")
        st.rerun()

    # 6. 检查自动加载复选框更改（只更新auto_load，不动模型名）
    # 注意不要每次都写，只有更改时写
    if auto_load != model_config.get("auto_load", False):
        config_manager.update_model_config(last_used_model, auto_load)

    if not available_models:
        st.warning("⚠️ 暂无可用模型，请先添加模型。")
        st.info("使用上方的模型添加功能来下载或添加本地模型。")
