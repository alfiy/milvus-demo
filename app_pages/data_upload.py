# import streamlit as st
# from components.milvus_mongo_insert import milvus_mongo_upload
# import pandas as pd


# def data_upload_page():
#     st.markdown("## 📊 数据上传与处理")

#     # 模型配置安全获取
#     raw_model_config = st.session_state.get("model_config", {})
#     model_config = raw_model_config if isinstance(raw_model_config, dict) else {}
#     current_model = model_config.get("last_used_model", "")

#     if not current_model or not st.session_state.get("model_loaded", False):
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
#                 st.rerun()
#         return

#     # 显示当前使用的模型
#     st.markdown("### 🤖 当前模型状态")
#     col1, col2 = st.columns([3, 1])
#     vp = st.session_state["components"]["vector_processor"]
#     with col1:
#         st.success(f"✅ 已加载模型: **{current_model}**")
#         model_info = vp.get_model_info()
#         if model_info:
#             st.info(f"🔢 向量维度: {model_info.get('dimension', 'N/A')}")
#     with col2:
#         if st.button("🔄 切换模型"):
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
#             help="支持JSON、JSONL格式文件。JSON格式：[{\"text\":\"内容\"}]，JSONL格式：每行一个JSON对象"
#         )
#         if uploaded_file is not None:
#             try:
#                 file_content = uploaded_file.read().decode('utf-8')
#                 json_data = vp.parse_json_file(file_content)
#                 if not isinstance(json_data, list):
#                     json_data = [json_data]
#                 st.success(f"✅ 成功加载 {len(json_data)} 条数据")
#                 file_size = uploaded_file.size / 1024 / 1024
#                 st.info(f"📁 文件大小: {file_size:.2f} MB")
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
#                 1. **标准JSON数组**: `[{"text":"内容1"}, {"text":"内容2"}]`
#                 2. **JSONL格式**: 每行一个JSON对象
#                    ```
#                    {"text":"内容1"}
#                    {"text":"内容2"}
#                    ```
#                 3. **单个JSON对象**: `{"text":"内容"}`
#                 """)
#     elif upload_method == "✏️ 手动输入JSON数据":
#         json_text = st.text_area(
#             "输入JSON数据",
#             height=200,
#             placeholder='[{"text":"半生长以客为家，罢直初来瀚海槎。始信人间行不尽，天涯更复有天涯。"}]',
#             help="请输入有效的JSON格式数据"
#         )
#         if json_text.strip():
#             try:
#                 json_data = vp.parse_json_file(json_text)
#                 if not isinstance(json_data, list):
#                     json_data = [json_data]
#                 st.success(f"✅ 成功解析 {len(json_data)} 条数据")
#             except Exception as e:
#                 st.error(f"❌ JSON解析失败: {e}")
#     elif upload_method == "🎯 使用示例数据":
#         sample_data = [
#             {"text": "半生长以客为家，罢直初来瀚海槎。始信人间行不尽，天涯更复有天涯。"},
#             {"text": "春风得意马蹄疾，一日看尽长安花。"},
#             {"text": "山重水复疑无路，柳暗花明又一村。"},
#             {"text": "海内存知己，天涯若比邻。"},
#             {"text": "落红不是无情物，化作春泥更护花。"},
#             {"text": "会当凌绝顶，一览众山小。"},
#             {"text": "采菊东篱下，悠然见南山。"},
#             {"text": "明月几时有，把酒问青天。"}
#         ]
#         json_data = sample_data
#         st.info(f"🎯 使用示例数据，共 {len(json_data)} 条古诗词")

#     # 数据预览和处理
#     if json_data:
#         st.markdown("### 📋 数据预览")
#         col1, col2, col3 = st.columns(3)
#         with col1:
#             st.metric("数据条数", len(json_data))
#         with col2:
#             total_chars = sum(len(str(item)) for item in json_data)
#             st.metric("总字符数", f"{total_chars:,}")
#         with col3:
#             avg_length = total_chars / len(json_data) if json_data else 0
#             st.metric("平均长度", f"{avg_length:.1f}")
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
#                 texts, vectors, metadata = vp.process_json_data(json_data)
#                 embedding_dim = vectors.shape[1]
#                 progress_bar.progress(60)
#                 milvus_manager = st.session_state["components"]["milvus_manager"]
#                 collection = milvus_manager.collection

#                 # 检查集合维度逻辑自动重建
#                 need_rebuild = False
#                 if collection:
#                     milvus_dim = None
#                     for f in collection.schema.fields:
#                         if 'dim' in f.params:
#                             milvus_dim = int(f.params['dim'])
#                             break
#                     if milvus_dim is None:
#                         st.error("❌ 当前集合schema未找到向量维度(dim)定义，请检查集合字段！")
#                         progress_bar.empty()
#                         status_text.empty()
#                         return
#                     if milvus_dim != embedding_dim:
#                         status_text.text(f"❗ 检测到模型向量维度({embedding_dim})与Milvus集合({milvus_dim})不一致，自动重建集合...")
#                         milvus_manager.delete_collection()
#                         need_rebuild = True
#                 else:
#                     need_rebuild = True

#                 if need_rebuild:
#                     success = milvus_manager.create_collection(embedding_dim)
#                     if not success:
#                         st.error("❌ Milvus集合重建失败，请检查数据库连接和配置信息！")
#                         progress_bar.empty()
#                         status_text.empty()
#                         return
#                     status_text.text(f"✅ Milvus集合已重建，维度: {embedding_dim}")
#                     progress_bar.progress(80)
#                     # 保证collection对象为最新
#                     milvus_manager.get_collection_object()

#                 # ==== 强制清洗文本，只保留string ====
#                 texts_clean = [t[0] if isinstance(t, list) and len(t) > 0 else t for t in texts]
#                 texts_clean = [str(t) for t in texts_clean if isinstance(t, str)]

#                 # print("DEBUG texts_clean type前5:", [(t, type(t)) for t in texts_clean[:5]])
#                 # print("DEBUG texts_clean结构:", texts_clean[:5])

#                 # 开始插入数据
#                 st.session_state.texts = texts
#                 st.session_state.vectors = vectors
#                 st.session_state.metadata = metadata
#                 st.session_state.data_loaded = True
#                 try:
#                     inserted_ids = milvus_mongo_upload(texts, vectors, metadata, milvus_dim=embedding_dim)
#                     progress_bar.progress(100)
#                     status_text.text(f"✅ 向量化及持久化完成！已插入 {len(inserted_ids)} 条数据。")
#                     st.success(f"💾 向量化和持久化完成！成功处理并写入 {len(inserted_ids)} 条文本数据。")
#                 except Exception as e:
#                     progress_bar.progress(100)
#                     status_text.text("⚠️ 向量化完成，但持久化失败")
#                     st.warning(f"⚠️ 向量化完成，但数据持久化失败: {e}")
#                     st.info("💾 数据已保存到内存中，可以进行搜索和聚类分析。要启用持久化，请检查Milvus和MongoDB连接。")

#                 # 搜索引擎、聚类分析同步
#                 st.session_state.components['search_engine'].load_data(vectors, texts, metadata)
#                 st.session_state.components['search_engine'].set_vector_processor(vp)
#                 st.session_state.components['clustering_analyzer'].load_data(vectors, texts, metadata)
#                 st.success(f"🎉 向量化完成！成功处理了 {len(texts)} 条文本")
#                 # 结果统计略
#             except Exception as e:
#                 st.error(f"❌ 向量化处理失败: {e}")
#                 st.exception(e)
#             finally:
#                 progress_bar.empty()
#                 status_text.empty()

import streamlit as st
from components.milvus_mongo_insert import milvus_mongo_upload
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from collections import Counter


# ============================================
# 数据质量验证器（增强版）
# ============================================

@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    reason: str = ""
    quality_score: float = 0.0
    metrics: Dict[str, Any] = None


class TextQualityValidator:
    """文本质量验证器 - Streamlit优化版"""
    
    def __init__(
        self,
        min_length: int = 10,
        max_length: int = 10000,
        max_url_count: int = 3,
        max_special_char_ratio: float = 0.4,
        min_chinese_ratio: float = 0.05,
        enable_strict_mode: bool = False
    ):
        """
        初始化验证器
        
        Args:
            min_length: 最小文本长度
            max_length: 最大文本长度
            max_url_count: 允许的最大URL数量
            max_special_char_ratio: 特殊字符最大比例
            min_chinese_ratio: 中文字符最小比例
            enable_strict_mode: 是否启用严格模式（更严格的过滤规则）
        """
        self.min_length = min_length
        self.max_length = max_length
        self.max_url_count = max_url_count
        self.max_special_char_ratio = max_special_char_ratio
        self.min_chinese_ratio = min_chinese_ratio
        self.enable_strict_mode = enable_strict_mode
        
        # 编译正则表达式（提高性能）
        self.url_pattern = re.compile(
            r'https?://[^\s]+|www\.[^\s]+|\w+\.(com|cn|net|org|edu|gov|io|co)/[^\s]*',
            re.IGNORECASE
        )
        self.chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
        self.special_char_pattern = re.compile(r'[^\w\s\u4e00-\u9fff]')
        
        # 垃圾模式（根据你的实际数据调整）
        self.garbage_patterns = [
            (r'(\.Shtml\s*){3,}', "重复HTML后缀", 5),
            (r'(blog\.|5g\.|m\.|h5\.|www\.){8,}', "重复子域名", 5),
            (r'(\d{5,}\s*){8,}', "大量连续数字", 3),
            (r'(<[^>]+>\s*){5,}', "大量HTML标签", 4),
            (r'(FROM:|来源:|※|·){3,}', "重复元信息标记", 3),
            (r'(Article/\d+|details/\d+|blog/\d+){5,}', "URL路径模式", 4),
        ]
        
        # 统计计数器
        self.stats = {
            "total": 0,
            "accepted": 0,
            "rejected": 0,
            "reasons": Counter()
        }
    
    def validate(self, text: str) -> ValidationResult:
        """
        验证单个文本
        
        Args:
            text: 待验证文本
            
        Returns:
            ValidationResult: 验证结果对象
        """
        self.stats["total"] += 1
        
        # 基础检查
        if not text or not isinstance(text, str):
            self.stats["rejected"] += 1
            self.stats["reasons"]["空文本或类型错误"] += 1
            return ValidationResult(False, "空文本或类型错误")
        
        text = text.strip()
        text_length = len(text)
        
        # 长度检查
        if text_length < self.min_length:
            self.stats["rejected"] += 1
            self.stats["reasons"][f"文本过短(<{self.min_length})"] += 1
            return ValidationResult(
                False, 
                f"文本过短({text_length} < {self.min_length})"
            )
        
        if text_length > self.max_length:
            self.stats["rejected"] += 1
            self.stats["reasons"][f"文本过长(>{self.max_length})"] += 1
            return ValidationResult(
                False,
                f"文本过长({text_length} > {self.max_length})"
            )
        
        # URL检查
        urls = self.url_pattern.findall(text)
        url_count = len(urls)
        if url_count > self.max_url_count:
            self.stats["rejected"] += 1
            self.stats["reasons"][f"链接过多(>{self.max_url_count})"] += 1
            return ValidationResult(
                False,
                f"链接过多({url_count} > {self.max_url_count})"
            )
        
        # 特殊字符比例检查
        special_chars = len(self.special_char_pattern.findall(text))
        special_char_ratio = special_chars / text_length
        if special_char_ratio > self.max_special_char_ratio:
            self.stats["rejected"] += 1
            self.stats["reasons"]["特殊字符过多"] += 1
            return ValidationResult(
                False,
                f"特殊字符过多({special_char_ratio:.1%})"
            )
        
        # 中文字符检查
        chinese_chars = len(self.chinese_pattern.findall(text))
        chinese_char_ratio = chinese_chars / text_length
        if chinese_char_ratio < self.min_chinese_ratio:
            self.stats["rejected"] += 1
            self.stats["reasons"]["中文内容不足"] += 1
            return ValidationResult(
                False,
                f"中文内容不足({chinese_char_ratio:.1%})"
            )
        
        # 垃圾模式检测
        for pattern, reason, weight in self.garbage_patterns:
            match = re.search(pattern, text)
            if match:
                self.stats["rejected"] += 1
                self.stats["reasons"][reason] += 1
                return ValidationResult(False, f"检测到垃圾模式: {reason}")
        
        # 严格模式额外检查
        if self.enable_strict_mode:
            # 检查重复字符
            if re.search(r'(.)\1{10,}', text):
                self.stats["rejected"] += 1
                self.stats["reasons"]["重复字符"] += 1
                return ValidationResult(False, "包含过多重复字符")
            
            # 检查是否全是数字和符号
            if chinese_chars < 5 and text_length > 20:
                self.stats["rejected"] += 1
                self.stats["reasons"]["有效中文不足"] += 1
                return ValidationResult(False, "有效中文字符不足5个")
        
        # 计算质量分数 (0-100)
        quality_score = self._calculate_quality_score(
            text_length, url_count, special_char_ratio, chinese_char_ratio
        )
        
        # 通过验证
        self.stats["accepted"] += 1
        return ValidationResult(
            True,
            "通过验证",
            quality_score,
            {
                "length": text_length,
                "url_count": url_count,
                "special_ratio": special_char_ratio,
                "chinese_ratio": chinese_char_ratio
            }
        )
    
    def _calculate_quality_score(
        self, 
        length: int, 
        url_count: int, 
        special_ratio: float, 
        chinese_ratio: float
    ) -> float:
        """计算文本质量分数"""
        score = 100.0
        
        # 长度分数 (理想长度50-500字符)
        if length < 50:
            score -= (50 - length) * 0.2
        elif length > 500:
            score -= (length - 500) * 0.01
        
        # URL惩罚
        score -= url_count * 5
        
        # 特殊字符惩罚
        score -= special_ratio * 30
        
        # 中文比例奖励
        score += min(chinese_ratio * 20, 20)
        
        return max(0, min(100, score))
    
    def clean_text(self, text: str) -> str:
        """
        清洗文本
        
        Args:
            text: 原始文本
            
        Returns:
            清洗后的文本
        """
        if not isinstance(text, str):
            return ""
        
        # 移除多余空白
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # 可选：移除某些特定的垃圾字符
        text = re.sub(r'[※·→←↑↓]', '', text)
        
        return text
    
    def get_stats_summary(self) -> Dict[str, Any]:
        """获取统计摘要"""
        total = self.stats["total"]
        if total == 0:
            return {"message": "暂无数据"}
        
        acceptance_rate = (self.stats["accepted"] / total) * 100
        
        return {
            "总验证数": total,
            "通过数": self.stats["accepted"],
            "拒绝数": self.stats["rejected"],
            "通过率": f"{acceptance_rate:.1f}%",
            "主要拒绝原因": dict(self.stats["reasons"].most_common(3))
        }
    
    def reset_stats(self):
        """重置统计"""
        self.stats = {
            "total": 0,
            "accepted": 0,
            "rejected": 0,
            "reasons": Counter()
        }


# ============================================
# 批量数据处理函数
# ============================================

def batch_validate_and_clean(
    json_data: List[Dict],
    validator: TextQualityValidator,
    show_progress: bool = True
) -> Tuple[List[Dict], List[Dict], Dict[str, Any]]:
    """
    批量验证和清洗数据
    
    Args:
        json_data: 原始JSON数据列表
        validator: 验证器实例
        show_progress: 是否显示进度条
        
    Returns:
        (有效数据列表, 无效数据列表, 统计信息)
    """
    valid_data = []
    invalid_data = []
    
    if show_progress:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    total = len(json_data)
    
    for idx, item in enumerate(json_data):
        # 检查数据格式
        if not isinstance(item, dict):
            invalid_data.append({
                "item": item,
                "reason": "数据格式错误：不是字典类型"
            })
            continue
        
        # 尝试查找文本字段（支持 text, text1, content 等常见字段名）
        text = None
        text_field = None
        for field in ["text", "text1", "content", "body", "message"]:
            if field in item:
                text = item[field]
                text_field = field
                break
        
        if text is None:
            invalid_data.append({
                "item": item,
                "reason": f"缺少文本字段（检查了: text, text1, content等）。实际字段: {list(item.keys())}"
            })
            continue
        
        # 清洗文本
        text = validator.clean_text(str(text))
        
        # 验证
        result = validator.validate(text)
        
        if result.is_valid:
            # 更新文本为清洗后的版本，统一使用 "text" 字段
            item_copy = item.copy()
            item_copy["text"] = text  # 统一字段名
            if text_field != "text" and text_field in item_copy:
                del item_copy[text_field]  # 删除原字段名（如果不是text）
            item_copy["_quality_score"] = result.quality_score
            item_copy["_original_field"] = text_field  # 记录原始字段名
            valid_data.append(item_copy)
        else:
            invalid_data.append({
                "item": item,
                "reason": result.reason,
                "text_preview": text[:100] if text else "无文本"
            })
        
        # 更新进度
        if show_progress and (idx + 1) % max(1, total // 20) == 0:
            progress = (idx + 1) / total
            progress_bar.progress(progress)
            status_text.text(
                f"正在验证数据... {idx + 1}/{total} "
                f"(通过: {len(valid_data)}, 拒绝: {len(invalid_data)})"
            )
    
    if show_progress:
        progress_bar.empty()
        status_text.empty()
    
    stats = validator.get_stats_summary()
    
    return valid_data, invalid_data, stats


# ============================================
# Streamlit 主页面函数
# ============================================

def data_upload_page():
    st.markdown("## 📤 数据上传与处理")

    # ===== 初始化验证器 =====
    if "validator" not in st.session_state:
        st.session_state.validator = TextQualityValidator(
            min_length=10,
            max_url_count=3,
            min_chinese_ratio=0.05,
            enable_strict_mode=False
        )

    # 模型配置安全获取
    raw_model_config = st.session_state.get("model_config", {})
    model_config = raw_model_config if isinstance(raw_model_config, dict) else {}
    current_model = model_config.get("last_used_model", "")

    if not current_model or not st.session_state.get("model_loaded", False):
        st.warning("⚠️ 尚未加载嵌入模型！")
        st.info("📌 请先到 '🔥 嵌入模型管理' 页面加载模型，然后再回到此页面进行数据处理。")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("""
            **为什么需要先加载模型？**
            - 文本向量化需要使用嵌入模型
            - 模型加载后可以处理任何文本数据
            - 统一的模型管理确保配置一致性
            """)
        with col2:
            if st.button("🔧 前往模型管理", type="primary"):
                st.switch_page("🔥 嵌入模型管理")
                st.rerun()
        return

    # 显示当前使用的模型
    st.markdown("### 🔥 当前模型状态")
    col1, col2 = st.columns([3, 1])
    vp = st.session_state["components"]["vector_processor"]
    with col1:
        st.success(f"✅ 已加载模型: **{current_model}**")
        model_info = vp.get_model_info()
        if model_info:
            st.info(f"📊 向量维度: {model_info.get('dimension', 'N/A')}")
    with col2:
        if st.button("🔄 切换模型"):
            st.info("📌 请到 '🔥 嵌入模型管理' 页面切换模型")

    st.markdown("---")

    # ===== 数据质量设置（可折叠） =====
    with st.expander("⚙️ 数据质量设置（高级）", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_length = st.number_input(
                "最小文本长度",
                min_value=5,
                max_value=100,
                value=st.session_state.validator.min_length,
                help="短于此长度的文本将被过滤"
            )
            max_url_count = st.number_input(
                "最大URL数量",
                min_value=0,
                max_value=10,
                value=st.session_state.validator.max_url_count,
                help="包含超过此数量URL的文本将被过滤"
            )
        
        with col2:
            min_chinese_ratio = st.slider(
                "最小中文比例",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.validator.min_chinese_ratio,
                step=0.05,
                format="%.2f",
                help="中文字符占比低于此值的文本将被过滤"
            )
            max_special_ratio = st.slider(
                "最大特殊字符比例",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.validator.max_special_char_ratio,
                step=0.05,
                format="%.2f",
                help="特殊字符占比高于此值的文本将被过滤"
            )
        
        with col3:
            enable_strict = st.checkbox(
                "启用严格模式",
                value=st.session_state.validator.enable_strict_mode,
                help="启用更严格的过滤规则"
            )
        
        if st.button("💾 应用设置"):
            st.session_state.validator = TextQualityValidator(
                min_length=min_length,
                max_url_count=max_url_count,
                min_chinese_ratio=min_chinese_ratio,
                max_special_char_ratio=max_special_ratio,
                enable_strict_mode=enable_strict
            )
            st.success("✅ 设置已更新")
            st.rerun()

    st.markdown("---")

    # 数据上传选项
    upload_method = st.radio(
        "选择数据输入方式",
        ["📁 上传JSON文件", "✏️ 手动输入JSON数据", "📝 使用示例数据"],
        horizontal=True
    )

    json_data = None
    validator = st.session_state.validator
    
    # ===== 数据输入部分 =====
    if upload_method == "📁 上传JSON文件":
        uploaded_file = st.file_uploader(
            "选择JSON文件",
            type=['json', 'jsonl', 'txt'],
            help="支持JSON、JSONL格式文件。JSON格式：[{\"text\":\"内容\"}]，JSONL格式：每行一个JSON对象"
        )
        if uploaded_file is not None:
            try:
                file_content = uploaded_file.read().decode('utf-8')
                json_data = vp.parse_json_file(file_content)
                if not isinstance(json_data, list):
                    json_data = [json_data]
                
                st.success(f"✅ 成功加载 {len(json_data)} 条原始数据")
                file_size = uploaded_file.size / 1024 / 1024
                st.info(f"📦 文件大小: {file_size:.2f} MB")
                
            except Exception as e:
                st.error(f"❌ 文件加载失败: {e}")
                st.markdown("""
                **支持的文件格式：**
                1. **标准JSON数组**: `[{"text":"内容1"}, {"text":"内容2"}]`
                2. **JSONL格式**: 每行一个JSON对象
                3. **单个JSON对象**: `{"text":"内容"}`
                """)
                
    elif upload_method == "✏️ 手动输入JSON数据":
        json_text = st.text_area(
            "输入JSON数据",
            height=200,
            placeholder='[{"text":"半生长以客为家，罢直初来瀚海槎。始信人间行不尽，天涯更复有天涯。"}]',
            help="请输入有效的JSON格式数据"
        )
        if json_text.strip():
            try:
                json_data = vp.parse_json_file(json_text)
                if not isinstance(json_data, list):
                    json_data = [json_data]
                st.success(f"✅ 成功解析 {len(json_data)} 条原始数据")
            except Exception as e:
                st.error(f"❌ JSON解析失败: {e}")
                
    elif upload_method == "📝 使用示例数据":
        sample_data = [
            {"text": "半生长以客为家，罢直初来瀚海槎。始信人间行不尽，天涯更复有天涯。"},
            {"text": "春风得意马蹄疾，一日看尽长安花。"},
            {"text": "山重水复疑无路，柳暗花明又一村。"},
            {"text": "海内存知己，天涯若比邻。"},
            {"text": "落红不是无情物，化作春泥更护花。"},
            {"text": "会当凌绝顶，一览众山小。"},
            {"text": "采菊东篱下，悠然见南山。"},
            {"text": "明月几时有，把酒问青天。"}
        ]
        json_data = sample_data
        st.info(f"📝 使用示例数据，共 {len(json_data)} 条古诗词")

    # ===== 数据验证和预览 =====
    if json_data:
        st.markdown("### 🔍 数据质量检查")
        
        # 验证数据
        with st.spinner("正在验证数据质量..."):
            valid_data, invalid_data, stats = batch_validate_and_clean(
                json_data, 
                validator,
                show_progress=True
            )
        
        # 显示统计信息
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 原始数据", len(json_data))
        with col2:
            st.metric("✅ 有效数据", len(valid_data), 
                     delta=f"{(len(valid_data)/len(json_data)*100):.1f}%")
        with col3:
            st.metric("❌ 无效数据", len(invalid_data))
        with col4:
            if valid_data:
                avg_quality = np.mean([d.get("_quality_score", 0) for d in valid_data])
                st.metric("⭐ 平均质量分", f"{avg_quality:.1f}")
        
        # 显示详细统计
        if stats:
            st.info(f"📈 **验证统计**: {stats.get('通过率', 'N/A')} 通过率")
            if invalid_data:
                with st.expander("⚠️ 查看拒绝原因统计"):
                    reasons = stats.get("主要拒绝原因", {})
                    for reason, count in reasons.items():
                        st.write(f"- **{reason}**: {count} 条")
        
        # 显示无效数据样例
        if invalid_data:
            with st.expander(f"❌ 查看无效数据样例（共{len(invalid_data)}条）"):
                sample_invalid = invalid_data[:5]
                for idx, item in enumerate(sample_invalid, 1):
                    st.markdown(f"**样例 {idx}**: {item['reason']}")
                    text_preview = str(item['item'].get('text', ''))[:100]
                    st.code(text_preview + "..." if len(text_preview) == 100 else text_preview)
                    st.markdown("---")
        
        # 数据预览
        if valid_data:
            st.markdown("### 📋 有效数据预览")
            
            # 计算统计
            total_chars = sum(len(str(item.get('text', ''))) for item in valid_data)
            avg_length = total_chars / len(valid_data) if valid_data else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("数据条数", len(valid_data))
            with col2:
                st.metric("总字符数", f"{total_chars:,}")
            with col3:
                st.metric("平均长度", f"{avg_length:.1f}")
            
            # 数据表格预览
            df_preview = pd.DataFrame([
                {
                    "文本": item.get('text', '')[:50] + "..." if len(item.get('text', '')) > 50 else item.get('text', ''),
                    "质量分": f"{item.get('_quality_score', 0):.1f}",
                    "长度": len(item.get('text', ''))
                }
                for item in valid_data[:10]
            ])
            st.dataframe(df_preview, use_container_width=True)
            
            if len(valid_data) > 10:
                st.info(f"显示前10条数据，总共{len(valid_data)}条")
            
            # ===== 向量化处理 =====
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
                    status_text.text("🔄 正在处理文本数据...")
                    progress_bar.progress(30)
                    
                    # 提取文本（移除质量分数字段）
                    clean_data = [
                        {k: v for k, v in item.items() if k != "_quality_score"}
                        for item in valid_data
                    ]
                    
                    texts, vectors, metadata = vp.process_json_data(clean_data)
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
                            status_text.text(
                                f"❗ 检测到模型向量维度({embedding_dim})与Milvus集合({milvus_dim})不一致，自动重建集合..."
                            )
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
                        milvus_manager.get_collection_object()

                    # 数据已经过清洗，直接处理
                    texts_clean = [t[0] if isinstance(t, list) and len(t) > 0 else t for t in texts]
                    texts_clean = [str(t) for t in texts_clean]

                    # 存储到session state
                    st.session_state.texts = texts_clean
                    st.session_state.vectors = vectors
                    st.session_state.metadata = metadata
                    st.session_state.data_loaded = True
                    
                    try:
                        inserted_ids = milvus_mongo_upload(
                            texts_clean, vectors, metadata, milvus_dim=embedding_dim
                        )
                        progress_bar.progress(100)
                        status_text.text(f"✅ 向量化及持久化完成！已插入 {len(inserted_ids)} 条数据。")
                        st.success(f"🎉 向量化和持久化完成！成功处理并写入 {len(inserted_ids)} 条文本数据。")
                    except Exception as e:
                        progress_bar.progress(100)
                        status_text.text("⚠️ 向量化完成，但持久化失败")
                        st.warning(f"⚠️ 向量化完成，但数据持久化失败: {e}")
                        st.info("💡 数据已保存到内存中，可以进行搜索和聚类分析。要启用持久化，请检查Milvus和MongoDB连接。")

                    # 搜索引擎、聚类分析同步
                    st.session_state.components['search_engine'].load_data(vectors, texts_clean, metadata)
                    st.session_state.components['search_engine'].set_vector_processor(vp)
                    st.session_state.components['clustering_analyzer'].load_data(vectors, texts_clean, metadata)
                    st.success(f"✅ 向量化完成！成功处理了 {len(texts_clean)} 条文本")
                    
                except Exception as e:
                    st.error(f"❌ 向量化处理失败: {e}")
                    st.exception(e)
                finally:
                    progress_bar.empty()
                    status_text.empty()
        else:
            st.warning("⚠️ 没有有效数据可以处理，请检查数据质量设置或数据内容")


# ============================================
# 辅助函数：显示验证器状态
# ============================================

def show_validator_status():
    """在侧边栏显示验证器状态"""
    if "validator" in st.session_state:
        validator = st.session_state.validator
        with st.sidebar:
            st.markdown("### 🔍 当前质量设置")
            st.markdown(f"- **最小长度**: {validator.min_length}")
            st.markdown(f"- **最大URL数**: {validator.max_url_count}")
            st.markdown(f"- **最小中文比例**: {validator.min_chinese_ratio:.1%}")
            st.markdown(f"- **严格模式**: {'✅ 开启' if validator.enable_strict_mode else '❌ 关闭'}")
            
            if validator.stats["total"] > 0:
                st.markdown("---")
                st.markdown("### 📊 本次统计")
                st.markdown(f"- **总验证**: {validator.stats['total']}")
                st.markdown(f"- **通过**: {validator.stats['accepted']}")
                st.markdown(f"- **拒绝**: {validator.stats['rejected']}")


