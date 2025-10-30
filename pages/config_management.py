import streamlit as st
from components.config_manager import config_manager
from components.vector_processor import VectorProcessor


def config_management_page():
    """配置管理页面"""
    st.markdown("## ⚙️ 系统配置管理")
    
    # 显示当前配置状态
    st.markdown("### 📊 当前配置状态")
    
    # 获取当前配置
    current_config = st.session_state.current_config
    milvus_config = current_config.get("milvus", {})
    mongodb_config = current_config.get("mongodb", {})
    model_config = current_config.get("model", {})
    # mongo_data = get_mongodb_data(mongodb_config)  # 如需实时拉取时再调用
    
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
        if st.button("📥 导入配置"):
            st.session_state.show_import = True
    
    with col2:
        if st.button("📤 导出配置"):
            st.session_state.show_export = True
    
    with col3:
        if st.button("🔄 重置配置"):
            st.session_state.show_reset = True
    
    with col4:
        if st.button("🔃 重新加载"):
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