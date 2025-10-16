import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
import os
from typing import List, Dict, Any
import torch
import json

class VectorProcessor:
    def __init__(self):
        """
        初始化向量处理器，使用本地缓存的模型
        """
        self.model = None
        self.model_name = "paraphrase-multilingual-MiniLM-L12-v2"
        self.dimension = 384  # 该模型的向量维度
        
        # 设置本地缓存路径
        self.cache_dir = os.path.expanduser("~/.cache/huggingface/transformers")


    def load_model(self) -> bool:
        """
        加载预训练的句子转换模型，自动优先使用本地缓存，无需手动检测缓存
        """
        try:
            st.info(" 正在加载文本向量化模型...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            st.info(f"️ 使用设备: {device}")

            # 直接加载模型，Huggingface会自动使用缓存
            self.model = SentenceTransformer(
                self.model_name,
                cache_folder=self.cache_dir,
                device=device
            )
            st.success("✅ 模型加载成功")

            # 验证模型
            test_text = "测试文本"
            test_vector = self.model.encode([test_text])
            if test_vector.shape[1] == self.dimension:
                st.success(f"📊 模型维度: {self.dimension} | 描述: 支持多语言，384维向量，平衡性能与质量")
                return True
            else:
                st.error(f"❌ 模型维度不匹配: 期望{self.dimension}，实际{test_vector.shape[1]}")
                return False

        except Exception as e:
            st.error(f"❌ 模型加载失败: {e}")
            # 尝试备用模型
            st.info(" 尝试使用备用模型...")
            try:
                backup_model = "all-MiniLM-L6-v2"
                self.model = SentenceTransformer(backup_model, device=device)
                self.dimension = 384
                st.warning(f"⚠️ 使用备用模型: {backup_model}")
                return True
            except Exception as backup_e:
                st.error(f"❌ 备用模型加载失败: {backup_e}")
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
            'cache_dir': self.cache_dir,
            'embedding_dimension': self.model.get_sentence_embedding_dimension()
        }
    
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
    

