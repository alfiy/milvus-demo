import streamlit as st
from components.milvus_mongo_insert import milvus_mongo_upload
import pandas as pd

def data_upload_page():
    st.markdown("## 📊 数据上传与处理")

    # 模型配置安全获取
    raw_model_config = st.session_state.get("model_config", {})
    model_config = raw_model_config if isinstance(raw_model_config, dict) else {}
    current_model = model_config.get("last_used_model", "")

    if not current_model or not st.session_state.get("model_loaded", False):
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
            if st.button("🚀 前往模型管理", type="primary"):
                st.switch_page("🤖 嵌入模型管理")
                st.rerun()
        return

    # 显示当前使用的模型
    st.markdown("### 🤖 当前模型状态")
    col1, col2 = st.columns([3, 1])
    vp = st.session_state["components"]["vector_processor"]
    with col1:
        st.success(f"✅ 已加载模型: **{current_model}**")
        model_info = vp.get_model_info()
        if model_info:
            st.info(f"🔢 向量维度: {model_info.get('dimension', 'N/A')}")
    with col2:
        if st.button("🔄 切换模型"):
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
                file_content = uploaded_file.read().decode('utf-8')
                json_data = vp.parse_json_file(file_content)
                if not isinstance(json_data, list):
                    json_data = [json_data]
                st.success(f"✅ 成功加载 {len(json_data)} 条数据")
                file_size = uploaded_file.size / 1024 / 1024
                st.info(f"📁 文件大小: {file_size:.2f} MB")
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
                json_data = vp.parse_json_file(json_text)
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
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("数据条数", len(json_data))
        with col2:
            total_chars = sum(len(str(item)) for item in json_data)
            st.metric("总字符数", f"{total_chars:,}")
        with col3:
            avg_length = total_chars / len(json_data) if json_data else 0
            st.metric("平均长度", f"{avg_length:.1f}")
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
            process_button = st.button("🚀 开始向量化处理并持久化", type="primary")
        if process_button:
            progress_bar = st.progress(0)
            status_text = st.empty()
            try:
                status_text.text("📊 正在处理文本数据...")
                progress_bar.progress(30)
                texts, vectors, metadata = vp.process_json_data(json_data)
                embedding_dim = vectors.shape[1]
                progress_bar.progress(60)
                milvus_manager = st.session_state["components"]["milvus_manager"]
                collection = milvus_manager.collection

                # 检查集合维度逻辑自动重建
                need_rebuild = False
                if collection:
                    milvus_dim = None
                    for f in collection.schema.fields:
                        if 'dim' in f.params:
                            milvus_dim = int(f.params['dim'])
                            break
                    if milvus_dim is None:
                        st.error("❌ 当前集合schema未找到向量维度(dim)定义，请检查集合字段！")
                        progress_bar.empty()
                        status_text.empty()
                        return
                    if milvus_dim != embedding_dim:
                        status_text.text(f"❗ 检测到模型向量维度({embedding_dim})与Milvus集合({milvus_dim})不一致，自动重建集合...")
                        milvus_manager.delete_collection()
                        need_rebuild = True
                else:
                    need_rebuild = True

                if need_rebuild:
                    success = milvus_manager.create_collection(embedding_dim)
                    if not success:
                        st.error("❌ Milvus集合重建失败，请检查数据库连接和配置信息！")
                        progress_bar.empty()
                        status_text.empty()
                        return
                    status_text.text(f"✅ Milvus集合已重建，维度: {embedding_dim}")
                    progress_bar.progress(80)
                    # 保证collection对象为最新
                    milvus_manager.get_collection_object()

                # ==== 强制清洗文本，只保留string ====
                texts_clean = [t[0] if isinstance(t, list) and len(t) > 0 else t for t in texts]
                texts_clean = [str(t) for t in texts_clean if isinstance(t, str)]

                print("DEBUG texts_clean type前5:", [(t, type(t)) for t in texts_clean[:5]])
                print("DEBUG texts_clean结构:", texts_clean[:5])

                # 开始插入数据
                st.session_state.texts = texts
                st.session_state.vectors = vectors
                st.session_state.metadata = metadata
                st.session_state.data_loaded = True
                try:
                    inserted_ids = milvus_mongo_upload(texts, vectors, metadata, milvus_dim=embedding_dim)
                    progress_bar.progress(100)
                    status_text.text(f"✅ 向量化及持久化完成！已插入 {len(inserted_ids)} 条数据。")
                    st.success(f"💾 向量化和持久化完成！成功处理并写入 {len(inserted_ids)} 条文本数据。")
                except Exception as e:
                    progress_bar.progress(100)
                    status_text.text("⚠️ 向量化完成，但持久化失败")
                    st.warning(f"⚠️ 向量化完成，但数据持久化失败: {e}")
                    st.info("💾 数据已保存到内存中，可以进行搜索和聚类分析。要启用持久化，请检查Milvus和MongoDB连接。")

                # 搜索引擎、聚类分析同步
                st.session_state.components['search_engine'].load_data(vectors, texts, metadata)
                st.session_state.components['search_engine'].set_vector_processor(vp)
                st.session_state.components['clustering_analyzer'].load_data(vectors, texts, metadata)
                st.success(f"🎉 向量化完成！成功处理了 {len(texts)} 条文本")
                # 结果统计略
            except Exception as e:
                st.error(f"❌ 向量化处理失败: {e}")
                st.exception(e)
            finally:
                progress_bar.empty()
                status_text.empty()

