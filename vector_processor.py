import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
import os
from typing import List, Dict, Any
import torch
import json
import subprocess


class VectorProcessor:
    def __init__(self, user_model_dir="./hf_embedding_cache"):
        """
        初始化向量处理器，支持用户动态添加本地嵌入模型，并自动扫描本地模型目录。
        """
        self.model = None
        self.model_name = None
        self.dimension = None
        self.user_model_dir = user_model_dir
        self.available_models = self.scan_local_models()
        self.model_select_idx = 0  # 默认选中第一个

    def scan_local_models(self):
        """
        扫描本地模型目录，返回所有模型名称列表
        """
        models = []
        if os.path.isdir(self.user_model_dir):
            for entry in os.listdir(self.user_model_dir):
                full_path = os.path.join(self.user_model_dir, entry)
                if os.path.isdir(full_path):
                    models.append(entry)
        return models

    def _ping_huggingface(self):
        """
        使用 curl 命令判断是否能连接 huggingface.co
        """
        try:
            # -I 只请求头，--connect-timeout 3 最多等3秒
            curl_cmd = ["curl", "-I", "--connect-timeout", "3", "https://huggingface.co"]
            result = subprocess.run(curl_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # 只要 exit code 为 0 且返回有 HTTP 状态行即认为可访问
            if result.returncode == 0 and b"HTTP" in result.stdout:
                return True
            else:
                return False
        except Exception:
            return False


    def _download_model(self, model_name):
        """
        使用 huggingface-cli 下载模型到本地 embedding cache 目录。
        支持用户直接输入完整模型名称（如 sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2、Qwen/Qwen3-Embedding-4B、BAAI/bge-m3 等）。
        """
        st.info(f"🌐 正在尝试下载模型: {model_name} 到 {self.user_model_dir}")
        os.makedirs(self.user_model_dir, exist_ok=True)
        try:
            # 用户输入的 model_name 应为完整的 huggingface 仓库名（如 Qwen/Qwen3-Embedding-4B）
            cmd = [
                "huggingface-cli", "download",
                model_name,
                "--cache-dir", self.user_model_dir
            ]
            st.info(f"执行命令: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                st.success("✅ 模型已成功下载到本地缓存目录")
                return True
            else:
                st.error(f"❌ 模型下载失败: {result.stderr}")
                return False
        except Exception as e:
            st.error(f"❌ 模型下载模型异常: {e}")
            return False


    def select_and_add_model_ui(self):
        """
        展示添加嵌入模型输入框和模型选择下拉框，返回用户选择的模型名
        """
        st.markdown("### 1. 添加或选择本地嵌入模型")
        st.info(f"请将 HuggingFace SentenceTransformer 格式的模型文件夹放置于 `{self.user_model_dir}/模型名`，或在下方输入模型名称自动下载。")

        # 输入框：允许用户添加新模型名
        new_model_name = st.text_input("添加或下载新模型（输入模型名称，如 paraphrase-multilingual-MiniLM-L12-v2 ）", value="")

        # 下载新模型逻辑
        if new_model_name:
            if new_model_name not in self.available_models:
                # 下载前先ping huggingface
                if self._ping_huggingface():
                    st.info("HuggingFace 官网可达，正在自动下载模型。")
                    download_success = self._download_model(new_model_name)
                    if download_success:
                        # 刷新模型列表，自动选中新模型
                        self.available_models = self.scan_local_models()
                        if new_model_name in self.available_models:
                            self.model_select_idx = self.available_models.index(new_model_name)
                            st.success(f"已添加新模型: {new_model_name}，无需再次下载。")
                else:
                    st.error("❌ 无法访问 huggingface.co，无法自动下载模型。请手动将模型文件夹放置到本地，并在下拉框中选择。")
                    st.info(f"你可以用命令/huggingface-cli离线下载模型，然后将其解压到 {self.user_model_dir}/{new_model_name}")
        # 下拉框：选择可用模型
        if self.available_models:
            selected = st.selectbox(
                "选择本地嵌入模型",
                options=self.available_models,
                index=self.model_select_idx,
                key="embedding_model_select"
            )
            self.model_name = selected
            st.info(f"当前选中模型: {selected}，路径: {os.path.join(self.user_model_dir, selected)}")
        else:
            st.warning(f"本地模型库 `{self.user_model_dir}` 为空，请添加模型或先下载！")
            self.model_name = None

    def load_model(self) -> bool:
        """
        加载实际 snapshot 子目录下的模型
        """
        if not self.model_name:
            st.error("❌ 未选中模型，请先添加/选择本地嵌入模型。")
            return False

        # 1. 自动找到 snapshot 目录
        base_dir = os.path.join(self.user_model_dir, self.model_name, "snapshots")
        if not os.path.isdir(base_dir):
            st.error(f"❌ 本地模型目录缺失的 snapshots 目录: {base_dir}")
            return False
        snapshot_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        if not snapshot_dirs:
            st.error(f"❌ 没有找到任何 snapshot 子目录，请检查模型下载是否完整！")
            return False
        model_path = os.path.join(base_dir, snapshot_dirs[0])
        st.info(f"正在加载模型目录: {model_path}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model = SentenceTransformer(model_path, device=device)
            test_vector = self.model.encode(["测试文本"])
            if len(test_vector.shape) == 2:
                self.dimension = test_vector.shape[1]
                st.success(f"✅ 模型加载成功，向量维度: {self.dimension}")
                return True
            else:
                st.error(f"❌ 模型输出维度异常: {test_vector.shape}")
                return False
        except Exception as e:
            st.error(f"❌ 模型加载失败: {e}")
            return False

    def preprocess_text(self, text: str) -> str:
        """
        文本预处理
        """
        if not text:
            return ""
        
        # 基本清理
        text = text.strip()
        # 可以根据需要添加更多预处理步骤
        return text
    
    def parse_json_file(self, file_content: str) -> List[Dict[str, Any]]:
        """
        解析JSON文件内容，支持多种格式
        """
        json_data = []
        
        try:
            # 尝试直接解析为JSON
            data = json.loads(file_content)
            if isinstance(data, list):
                json_data = data
            elif isinstance(data, dict):
                json_data = [data]
            else:
                st.error("JSON文件格式不正确，应为对象或对象数组")
                return []
        except json.JSONDecodeError as e:
            # 如果直接解析失败，尝试按行解析（JSONL格式）
            st.info("尝试按JSONL格式解析...")
            lines = file_content.strip().split('\n')
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    item = json.loads(line)
                    json_data.append(item)
                except json.JSONDecodeError as line_error:
                    st.warning(f"第{i+1}行JSON解析失败: {line_error}")
                    st.warning(f"问题行内容: {line[:100]}...")
                    continue
            
            if not json_data:
                st.error(f"无法解析JSON文件。原始错误: {e}")
                st.error("请确保文件是有效的JSON格式，例如：")
                st.code('[{"text":"文本内容1"}, {"text":"文本内容2"}]')
                st.error("或者JSONL格式（每行一个JSON对象）：")
                st.code('{"text":"文本内容1"}\n{"text":"文本内容2"}')
                return []
        
        return json_data
    
    def process_json_data(self, json_data: List[Dict[str, Any]]) -> tuple:
        """
        处理JSON格式的数据
        返回文本列表和对应的向量
        """
        texts = []
        metadata = []
        
        for i, item in enumerate(json_data):
            if isinstance(item, dict):
                for key, value in item.items():
                    if isinstance(value, str) and value.strip():
                        texts.append(value)
                        metadata.append({
                            'id': i,
                            'key': key,
                            'original_data': item
                        })
            elif isinstance(item, str) and item.strip():
                # 如果直接是字符串
                texts.append(item)
                metadata.append({
                    'id': i,
                    'key': 'text',
                    'original_data': {'text': item}
                })
        
        if not texts:
            return [], [], []
        
        # 编码文本
        vectors = self.encode_texts(texts)
        
        return texts, vectors, metadata
    
    def encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        将文本列表编码为向量
        """
        if not self.model:
            st.error("❌ 模型未加载")
            return np.array([])
        
        try:
            st.info(f"🔄 正在处理 {len(texts)} 条文本...")
            
            # 创建进度条
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            all_vectors = []
            total_batches = (len(texts) + batch_size - 1) // batch_size
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # 预处理文本
                processed_texts = [self.preprocess_text(text) for text in batch_texts]
                
                # 编码当前批次
                batch_vectors = self.model.encode(
                    processed_texts,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    batch_size=min(batch_size, len(batch_texts))
                )
                
                all_vectors.append(batch_vectors)
                
                # 更新进度
                current_batch = (i // batch_size) + 1
                progress = current_batch / total_batches
                progress_bar.progress(progress)
                status_text.text(f"✅ 已处理 {min(i + batch_size, len(texts))}/{len(texts)} 条文本")
            
            # 合并所有向量
            vectors = np.vstack(all_vectors)
            
            progress_bar.progress(1.0)
            status_text.text(f"🎉 文本向量化完成！生成 {vectors.shape[0]} 个 {vectors.shape[1]} 维向量")
            
            return vectors
            
        except Exception as e:
            st.error(f"❌ 文本编码失败: {e}")
            return np.array([])
        finally:
            # 清理进度显示
            if 'progress_bar' in locals():
                progress_bar.empty()
            if 'status_text' in locals():
                status_text.empty()
    
    def encode_single_text(self, text: str) -> np.ndarray:
        """
        编码单个文本为向量
        """
        if not self.model:
            st.error("❌ 模型未加载")
            return np.array([])
        
        try:
            processed_text = self.preprocess_text(text)
            vector = self.model.encode([processed_text], convert_to_numpy=True)
            return vector[0]  # 返回单个向量
        except Exception as e:
            st.error(f"❌ 单文本编码失败: {e}")
            return np.array([])
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        """
        if not self.model:
            return {}
        
        return {
            'model_name': self.model_name,
            'dimension': self.dimension,
            'device': str(self.model.device),
            'max_seq_length': getattr(self.model, 'max_seq_length', 'Unknown'),
            'cache_dir': self.user_model_dir,
            'embedding_dimension': self.model.get_sentence_embedding_dimension()
        }
    
    def encode(self, texts):
        """
        对输入文本列表进行向量化编码
        """
        if self.model is None:
            raise ValueError("模型未加载，请先加载嵌入模型。")
        return self.model.encode(texts)

    def calculate_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        计算两个向量的余弦相似度
        """
        try:
            # 计算余弦相似度
            dot_product = np.dot(vector1, vector2)
            norm1 = np.linalg.norm(vector1)
            norm2 = np.linalg.norm(vector2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            st.error(f"❌ 相似度计算失败: {e}")
            return 0.0
    

