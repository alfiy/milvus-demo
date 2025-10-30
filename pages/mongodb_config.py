import streamlit as st
from components.config_manager import config_manager
from components.utils import auto_connect_mongodb

def mongodb_config_page():
    st.markdown("## 🍃 MongoDB配置管理")

    # 配置信息初始化，从配置文件加载
    if "mongodb_config" not in st.session_state:
        saved_config = config_manager.get_mongodb_config()
        st.session_state["mongodb_config"] = {
            "host": saved_config.get("host", "localhost"),
            "port": saved_config.get("port", 27017),
            "username": saved_config.get("username", ""),
            "password": saved_config.get("password", ""),
            "db_name": saved_config.get("db_name", "textdb"),
            "col_name": saved_config.get("col_name", "metadata"),
            "auto_connect": saved_config.get("auto_connect", False)
        }
    # 自动连接（只需顶部运行一次即可）
    # auto_connect_mongodb()
    mongodb_config = st.session_state["mongodb_config"]

    # 显示当前连接状态
    st.markdown("### 📊 当前连接状态")

    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.get('mongodb_connected'):
            st.markdown(
            f"<div class='config-card'><h4>✅ MongoDB 已连接</h4>"
            f"<p><strong>主机:</strong> {mongodb_config.get('host', '')}:{mongodb_config.get('port', '')}</p>"
            f"<p><strong>数据库:</strong> {mongodb_config.get('db_name', '')}</p>"
            f"<p><strong>集合:</strong> {mongodb_config.get('col_name', '')}</p></div>",
            unsafe_allow_html=True
        )
        else:
            st.markdown(
                "<div class='config-card'><h4>❌ MongoDB 未连接</h4>"
                "<p>请配置连接信息并测试连接</p></div>",
                unsafe_allow_html=True
            )
            error_msg = st.session_state.get('mongodb_connect_error')
            if error_msg:
                st.error(error_msg)
    
    with col2:
        auto_connect_status = "✅ 开启" if mongodb_config.get("auto_connect", False) else "❌ 关闭"
        st.markdown(
            f"<div class='config-card'><h4>⚙️ 配置状态</h4>"
            f"<p><strong>自动连接:</strong> {auto_connect_status}</p>"
            f"<p><strong>配置保存:</strong> {'✅ 已保存' if mongodb_config.get('host') else '❌ 未保存'}</p></div>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("### 请输入MongoDB连接信息")
    st.markdown("### 🔧 连接配置")

    # 表单输入控件，使用临时变量避免直接修改session_state
    config_input = {k: v for k, v in mongodb_config.items()}  # 新建临时副本
              
    col1, col2 = st.columns(2)
    with col1:
        config_input["host"] = st.text_input("MongoDB主机地址", value= mongodb_config["host"])
        config_input["port"] = st.number_input("MongoDB端口", value= mongodb_config["port"], min_value=1, max_value=65535)
        config_input["db_name"] = st.text_input("数据库名", value= mongodb_config["db_name"])
        config_input["col_name"] = st.text_input("集合名", value= mongodb_config["col_name"])
    with col2:
        config_input["username"] = st.text_input("用户名", value= mongodb_config["username"], placeholder="可选")
        config_input["password"] = st.text_input("密码", value= mongodb_config["password"], type="password", placeholder="可选")
        config_input["auto_connect"] = st.checkbox("启动时自动连接", value= mongodb_config.get("auto_connect", False))

    col1, col2 = st.columns([3, 1])
    with col1:
        st.info("✅  测试连接成功后，配置将自动保存到配置文件，重启应用后可自动恢复连接")
    with col2:
        test_button = st.button("🧪 测试连接", type="primary")


    # 测试连接按钮逻辑
    if test_button:
        # 1. 保存临时输入到 session
        st.session_state["mongodb_config"] = config_input
        # 2. 用全局函数检测连接写入全局状态
        auto_connect_mongodb()
        # 3. 配置持久化
        config_manager.update_mongodb_config(**config_input)
        # 4. 刷新页面
        st.rerun()

    # 连接管理操作
    if st.session_state.get("mongodb_connected"):
        st.markdown("---")
        st.markdown("### 🛠️ 连接管理")
        co1, co2, co3 = st.columns(3)
        with co1:
            if st.button("🔄 重新连接"):
                auto_connect_mongodb()
                st.rerun()
        with co2:
            if st.button("🧪 测试数据库"):
                try:
                    client = st.session_state["mongodb_client"]
                    db = client[mongodb_config["db_name"]]
                    col = db[mongodb_config["col_name"]]
                    count = col.estimated_document_count()
                    st.success(f"✅ 数据库连接正常，集合中有 {count:,} 条记录")
                except Exception as e:
                    st.error(f"❌ 数据库测试失败: {e}")
        with co3:
            if st.button("🔌 断开连接"):
                st.session_state['mongodb_connected'] = False
                st.session_state['mongodb_client'] = None
                st.session_state['mongodb_connect_error'] = None
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

    # 数据管理
    if st.session_state.get('mongodb_connected'):
        client = st.session_state["mongodb_client"]
        db = client[mongodb_config["db_name"]]
        col = db[mongodb_config["col_name"]]
        st.markdown("---")
        st.markdown("### 🗄️ 数据管理")

        docs = list(col.find({}, {"_id": 1, "text": 1}).limit(10))
        if docs:
            st.markdown("**最近数据（仅显示前 10 条）**")
            for doc in docs:
                st.markdown(f"- <span style='font-size: 90%'>{str(doc.get('_id'))}: {doc.get('text', '')[:40]}</span>", unsafe_allow_html=True)

            st.markdown("#### 删除指定数据")
            doc_ids = [str(doc["_id"]) for doc in docs]
            del_id = st.selectbox("选择要删除的数据ID", options=doc_ids)
            if st.button("❌ 删除此数据", key="delete_one"):
                try:
                    result = col.delete_one({"_id": del_id})
                    if result.deleted_count:
                        st.success(f"✅ 数据 {del_id} 已删除")
                        st.rerun()
                    else:
                        st.error("❌ 删除失败或数据不存在")
                except Exception as e:
                    st.error(f"❌ 删除异常: {e}")

        st.markdown("#### 删除全部数据")
        if st.button("❌ 全部删除", key="delete_all"):
            try:
                result = col.delete_many({})
                st.success(f"✅ 已删除全部数据，共 {result.deleted_count} 条")
                st.rerun()
            except Exception as e:
                st.error(f"❌ 删除异常: {e}")