import streamlit as st
from components.config_manager import config_manager
from components.milvus_mongo_insert import debug_collection_info, get_milvus_collection


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
    milvus_config = st.session_state.milvus_config
    # milvus_config = st.session_state.get("milvus_config", config_manager.get_milvus_config())

    
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
        connect_button = st.button("🔗 连接数据库", type="primary")
    
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
            if st.button("🆕 创建/连接集合"):
                if st.session_state.data_loaded:
                    with st.spinner("正在创建/连接集合..."):
                        dimension = st.session_state.vectors.shape[1]
                        success = st.session_state.components['milvus_manager'].create_collection(dimension)
                        if success:
                            st.rerun()
                else:
                    st.warning("⚠️ 请先上传并处理数据")
        
        with col2:
            if st.button("📤 插入数据到Milvus"):
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
                if st.button("🗑️ 删除集合", key="delete_collection_btn"):
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
            if st.button("🔍 调试集合信息"):
                debug_collection_info("text_vectors")
        with col2:
            if st.button("🧪 测试连接"):
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