import numpy as np
import os
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional, Callable
import torch
import json
import subprocess

class VectorProcessor:
    def __init__(self, user_model_dir="./hf_embedding_cache"):
        """
        向量处理器，不包含任何streamlit UI代码
        """
        self.model = None
        self.model_name = None
        self.dimension = None
        self.user_model_dir = user_model_dir
        self.available_models = self.scan_local_models()
        self.model_select_idx = 0

    def scan_local_models(self) -> List[str]:
        """扫描本地模型目录，返回模型列表"""
        models = []
        if os.path.isdir(self.user_model_dir):
            for entry in os.listdir(self.user_model_dir):
                if entry == ".locks":
                    continue
                full_path = os.path.join(self.user_model_dir, entry)
                if os.path.isdir(full_path):
                    models.append(entry)
        return models

    def _ping_huggingface(self) -> bool:
        """检查huggingface.co是否可访问"""
        try:
            curl_cmd = ["curl", "-I", "--connect-timeout", "3", "https://huggingface.co"]
            result = subprocess.run(curl_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return result.returncode == 0 and b"HTTP" in result.stdout
        except Exception:
            return False

    def download_model(
        self,
        model_name: str,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> (bool, str):
        """
        下载模型到本地缓存目录。如有log回调参数则每输出一行调用一次
        """
        os.makedirs(self.user_model_dir, exist_ok=True)
        try:
            cmd = [
                "huggingface-cli", "download",
                model_name,
                "--cache-dir", self.user_model_dir
            ]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            while True:
                line = proc.stdout.readline()
                if not line and proc.poll() is not None:
                    break
                if line and log_callback:
                    log_callback(line.rstrip())
            proc.wait()
            if proc.returncode == 0:
                return True, "模型已成功下载到本地缓存目录"
            else:
                return False, f"模型下载失败: 代码 {proc.returncode}"
        except Exception as e:
            return False, f"模型下载异常: {e}"

    def set_model_name(self, model_name: str):
        """设置要加载的模型名"""
        self.model_name = model_name

    def load_model(self) -> (bool, str):
        """
        加载 snapshot 目录下的模型，仅返回运行结果和提示信息
        """
        if not self.model_name:
            return False, "未选中模型"
        base_dir = os.path.join(self.user_model_dir, self.model_name, "snapshots")
        if not os.path.isdir(base_dir):
            return False, f"模型目录缺失: {base_dir}"
        snapshot_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        if not snapshot_dirs:
            return False, "没有snapshot子目录，请检查模型下载"
        model_path = os.path.join(base_dir, snapshot_dirs[0])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model = SentenceTransformer(model_path, device=device)
            test_vector = self.model.encode(["测试文本"])
            if len(test_vector.shape) == 2:
                self.dimension = test_vector.shape[1]
                return True, f"模型加载成功，维度: {self.dimension}"
            else:
                return False, f"模型输出形状异常: {test_vector.shape}"
        except Exception as e:
            return False, f"模型加载失败: {e}"

    def preprocess_text(self, text: str) -> str:
        """简单清理文本"""
        return text.strip() if text else ""

    def parse_json_file(self, file_content: str) -> List[Dict[str, Any]]:
        """
        解析JSON或JSONL格式，出错则返回空列表
        """
        json_data = []
        try:
            data = json.loads(file_content)
            if isinstance(data, list):
                json_data = data
            elif isinstance(data, dict):
                json_data = [data]
            else:
                return []
        except json.JSONDecodeError:
            lines = file_content.strip().split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    json_data.append(item)
                except Exception:
                    continue
        return json_data

    def process_json_data(self, json_data: List[Dict[str, Any]]) -> (List[str], np.ndarray, List[dict]):
        """
        处理json数据，返回：文本列表，向量数组，元数据列表
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
                texts.append(item)
                metadata.append({
                    'id': i,
                    'key': 'text',
                    'original_data': {'text': item}
                })
        if not texts:
            return [], np.array([]), []
        vectors = self.encode_texts(texts)
        return texts, vectors, metadata

    def encode_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> np.ndarray:
        """
        编码文本为向量。progress_callback(done_qty, total_qty)可用于进度条
        """
        if not self.model:
            return np.array([])
        try:
            all_vectors = []
            total_batches = (len(texts) + batch_size - 1) // batch_size
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                processed_texts = [self.preprocess_text(text) for text in batch_texts]
                batch_vectors = self.model.encode(
                    processed_texts,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    batch_size=min(batch_size, len(batch_texts))
                )
                all_vectors.append(batch_vectors)
                if progress_callback:
                    progress_callback(min(i + batch_size, len(texts)), len(texts))
            vectors = np.vstack(all_vectors)
            return vectors
        except Exception:
            return np.array([])

    def encode_single_text(self, text: str) -> np.ndarray:
        """编码单条文本"""
        if not self.model:
            return np.array([])
        processed_text = self.preprocess_text(text)
        try:
            vector = self.model.encode([processed_text], convert_to_numpy=True)
            return vector[0]
        except Exception:
            return np.array([])

    def get_model_info(self) -> Dict[str, Any]:
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

    def encode(self, texts: List[str]) -> np.ndarray:
        if self.model is None:
            raise ValueError("模型未加载")
        return self.model.encode(texts)

    def calculate_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        try:
            dot_product = np.dot(vector1, vector2)
            norm1 = np.linalg.norm(vector1)
            norm2 = np.linalg.norm(vector2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(dot_product / (norm1 * norm2))
        except Exception:
            return 0.0
