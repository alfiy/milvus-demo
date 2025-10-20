# milvus_manager_enhanced.py
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
import numpy as np
import streamlit as st
from typing import List, Dict, Any, Optional
import json
import math
import time
from config_manager import config_manager

class MilvusManager:
    def __init__(self, host: str = "localhost", port: str = "19530", user: str = "", password: str = ""):
        """
        初始化Milvus连接管理器
        """
        # 尝试从配置文件加载设置
        saved_config = config_manager.get_milvus_config()
        
        self.host = host if host != "localhost" else saved_config.get("host", "localhost")
        self.port = port if port != "19530" else saved_config.get("port", "19530")
        self.user = user if user else saved_config.get("user", "")
        self.password = password if password else saved_config.get("password", "")
        self.collection_name = saved_config.get("collection_name", "text_vectors")
        self.collection = None
        self.is_connected = False
        
        # 如果配置了自动连接，则尝试连接
        if saved_config.get("auto_connect", False):
            self.connect()
    
    def connect(self, save_config: bool = True) -> bool:
        """
        连接到Milvus服务器并检查现有集合
        
        Args:
            save_config: 是否保存连接配置
        """
        try:
            # 断开之前的连接（如果存在）
            try:
                connections.disconnect("default")
            except:
                pass
            
            # 准备连接参数
            connect_params = {
                "alias": "default",
                "host": self.host,
                "port": self.port
            }
            
            # 如果提供了用户名和密码，添加认证信息
            if self.user and self.password:
                connect_params.update({
                    "user": self.user,
                    "password": self.password
                })
            
            # 建立新连接
            connections.connect(**connect_params)
            self.is_connected = True
            
            # 保存连接配置
            if save_config:
                config_manager.update_milvus_config(
                    host=self.host,
                    port=self.port,
                    user=self.user,
                    password=self.password,
                    collection_name=self.collection_name,
                    auto_connect=True
                )
                st.success("✅ 连接配置已保存，下次启动将自动连接")
            
            # 检查是否存在现有集合
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
                # 加载集合到内存
                self.collection.load()
                
                # 获取集合统计信息
                stats = self.get_collection_stats()
                num_entities = stats.get('num_entities', 0)
                
                st.success(f"✅ 成功连接到Milvus服务器 {self.host}:{self.port}")
                if num_entities > 0:
                    st.info(f"🔍 发现现有集合 '{self.collection_name}'，包含 {num_entities:,} 条记录")
                else:
                    st.info(f"📝 发现现有集合 '{self.collection_name}'，但暂无数据")
            else:
                st.success(f"✅ 成功连接到Milvus服务器 {self.host}:{self.port}")
                st.info("📋 未发现现有集合，请创建新集合")
            
            return True
            
        except Exception as e:
            st.error(f"❌ 连接Milvus失败: {e}")
            self.is_connected = False
            return False
    
    def update_connection_params(self, host: str, port: str, user: str = "", password: str = "", 
                               collection_name: str = "text_vectors") -> None:
        """
        更新连接参数
        
        Args:
            host: 主机地址
            port: 端口
            user: 用户名
            password: 密码
            collection_name: 集合名称
        """
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.collection_name = collection_name
    
    def get_connection_info(self) -> Dict[str, Any]:
        """
        获取当前连接信息
        
        Returns:
            连接信息字典
        """
        return {
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "password": "***" if self.password else "",
            "collection_name": self.collection_name,
            "is_connected": self.is_connected,
            "has_collection": self.collection is not None
        }
    
    def create_collection(self, dimension: int = 384, description: str = "文本向量集合") -> bool:
        """
        创建向量集合（如果不存在）或连接到现有集合
        """
        try:
            # 检查集合是否已存在
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
                # 确保集合已加载
                self.collection.load()
                
                # 获取现有数据统计
                stats = self.get_collection_stats()
                num_entities = stats.get('num_entities', 0)
                
                st.info(f"📋 集合 '{self.collection_name}' 已存在，包含 {num_entities:,} 条记录")
                return True
            
            # 创建新集合
            st.info("🔨 正在创建新的向量集合...")
            
            # 定义字段模式
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension),
                FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535)
            ]
            
            # 创建集合模式
            schema = CollectionSchema(fields, description)
            
            # 创建集合
            self.collection = Collection(self.collection_name, schema)
            
            # 创建索引以提高搜索性能
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            self.collection.create_index("vector", index_params)
            
            # 加载集合到内存
            self.collection.load()
            
            st.success(f"✅ 成功创建集合 '{self.collection_name}'")
            return True
            
        except Exception as e:
            st.error(f"❌ 创建集合失败: {e}")
            return False
    
    def calculate_batch_size(self, texts: List[str], vectors: np.ndarray, metadata: List[Dict]) -> int:
        """
        根据数据大小计算合适的批次大小
        """
        # 估算单条记录的大小（字节）
        sample_size = min(100, len(texts))
        avg_text_size = sum(len(text.encode('utf-8')) for text in texts[:sample_size]) / sample_size
        vector_size = vectors[0].nbytes if len(vectors) > 0 else 0
        avg_metadata_size = sum(len(json.dumps(meta, ensure_ascii=False).encode('utf-8')) for meta in metadata[:sample_size]) / sample_size
        
        single_record_size = avg_text_size + vector_size + avg_metadata_size
        
        # gRPC默认最大消息大小为64MB，我们使用50MB作为安全边界
        max_message_size = 50 * 1024 * 1024  # 50MB
        
        # 计算批次大小
        batch_size = max(1, int(max_message_size / single_record_size))
        
        # 限制批次大小在合理范围内
        batch_size = min(batch_size, 1000)  # 最大1000条
        batch_size = max(batch_size, 10)    # 最小10条
        
        st.info(f"📊 估算单条记录大小: {single_record_size/1024:.2f} KB，批次大小: {batch_size}")
        
        return batch_size
    
    def insert_vectors(self, texts: List[str], vectors: np.ndarray, metadata: List[Dict]) -> bool:
        """
        分批插入向量数据，确保数据持久化
        """
        if not self.collection:
            st.error("❌ 集合未初始化")
            return False
        
        try:
            total_records = len(texts)
            
            # 检查是否有重复数据（简单检查）
            existing_stats = self.get_collection_stats()
            existing_count = existing_stats.get('num_entities', 0)
            
            if existing_count > 0:
                st.warning(f"⚠️ 集合中已存在 {existing_count:,} 条记录")
                
                # 询问用户是否继续
                if st.button("➕ 继续添加数据", key="continue_insert"):
                    pass  # 继续执行
                else:
                    st.info("💡 如需重新开始，请先删除现有集合")
                    return False
            
            # 计算合适的批次大小
            batch_size = self.calculate_batch_size(texts, vectors, metadata)
            
            # 计算总批次数
            total_batches = math.ceil(total_records / batch_size)
            
            st.info(f"🚀 开始分批插入数据，总共 {total_records:,} 条记录，分 {total_batches} 批处理")
            
            # 创建进度条
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            inserted_count = 0
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, total_records)
                
                # 准备当前批次的数据
                batch_texts = texts[start_idx:end_idx]
                batch_vectors = vectors[start_idx:end_idx]
                batch_metadata = metadata[start_idx:end_idx]
                
                # 准备插入数据
                data = [
                    batch_texts,
                    batch_vectors.tolist(),
                    [json.dumps(meta, ensure_ascii=False) for meta in batch_metadata]
                ]
                
                try:
                    # 插入当前批次
                    mr = self.collection.insert(data)
                    
                    # 强制刷新到磁盘，确保数据持久化
                    self.collection.flush()
                    
                    inserted_count += len(batch_texts)
                    
                    # 更新进度
                    progress = (batch_idx + 1) / total_batches
                    progress_bar.progress(progress)
                    status_text.text(f"✅ 已插入 {inserted_count:,}/{total_records:,} 条记录 (批次 {batch_idx + 1}/{total_batches})")
                    
                except Exception as batch_error:
                    st.error(f"❌ 批次 {batch_idx + 1} 插入失败: {batch_error}")
                    
                    # 如果批次仍然太大，尝试更小的批次
                    if "message larger than max" in str(batch_error):
                        st.warning(f"⚠️ 批次 {batch_idx + 1} 数据量过大，尝试拆分为更小批次...")
                        
                        # 将当前批次拆分为更小的子批次
                        sub_batch_size = max(1, len(batch_texts) // 4)
                        
                        for sub_start in range(0, len(batch_texts), sub_batch_size):
                            sub_end = min(sub_start + sub_batch_size, len(batch_texts))
                            
                            sub_data = [
                                batch_texts[sub_start:sub_end],
                                batch_vectors[sub_start:sub_end].tolist(),
                                [json.dumps(meta, ensure_ascii=False) for meta in batch_metadata[sub_start:sub_end]]
                            ]
                            
                            try:
                                mr = self.collection.insert(sub_data)
                                self.collection.flush()  # 确保每个子批次都持久化
                                inserted_count += (sub_end - sub_start)
                                
                                status_text.text(f"✅ 已插入 {inserted_count:,}/{total_records:,} 条记录 (子批次处理中...)")
                                
                            except Exception as sub_error:
                                st.error(f"❌ 子批次插入失败: {sub_error}")
                                continue
                    else:
                        # 如果不是大小问题，继续处理下一批次
                        continue
            
            # 最终刷新和压缩，确保所有数据都持久化
            self.collection.flush()
            
            # 等待索引构建完成
            if inserted_count > 0:
                st.info("🔧 正在构建索引，请稍候...")
                self.collection.load()  # 重新加载集合
            
            progress_bar.progress(1.0)
            status_text.text(f"🎉 数据插入完成！成功插入 {inserted_count:,} 条记录")
            
            if inserted_count > 0:
                # 验证数据是否真正插入
                final_stats = self.get_collection_stats()
                final_count = final_stats.get('num_entities', 0)
                
                st.success(f"✅ 数据已成功持久化到Milvus！")
                st.success(f"📊 集合中现有总记录数: {final_count:,}")
                
                return True
            else:
                st.error("❌ 没有成功插入任何记录")
                return False
                
        except Exception as e:
            st.error(f"❌ 插入数据失败: {e}")
            return False
        finally:
            # 清理进度显示
            if 'progress_bar' in locals():
                progress_bar.empty()
            if 'status_text' in locals():
                status_text.empty()
    
    def delete_records_by_ids(self, record_ids: List[int]) -> bool:
        """
        根据ID删除指定记录
        """
        if not self.collection:
            st.error("❌ 集合未初始化")
            return False
        
        try:
            st.info(f"🗑️ 正在删除 {len(record_ids)} 条记录...")
            
            # 创建进度条
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 构建删除表达式
            id_list_str = ",".join(map(str, record_ids))
            expr = f"id in [{id_list_str}]"
            
            status_text.text("🗑️ 执行删除操作...")
            progress_bar.progress(0.5)
            
            # 执行删除
            self.collection.delete(expr)
            
            status_text.text("💾 刷新数据到磁盘...")
            progress_bar.progress(0.8)
            
            # 刷新数据
            self.collection.flush()
            
            progress_bar.progress(1.0)
            status_text.text("✅ 删除完成")
            
            st.success(f"✅ 成功删除 {len(record_ids)} 条记录")
            return True
            
        except Exception as e:
            st.error(f"❌ 删除记录失败: {e}")
            return False
        finally:
            # 清理进度显示
            if 'progress_bar' in locals():
                progress_bar.empty()
            if 'status_text' in locals():
                status_text.empty()
    
    def delete_records_by_text_pattern(self, text_pattern: str, exact_match: bool = False) -> bool:
        """
        根据文本模式删除记录
        """
        if not self.collection:
            st.error("❌ 集合未初始化")
            return False
        
        try:
            st.info(f"🗑️ 正在删除包含 '{text_pattern}' 的记录...")
            
            # 创建进度条
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 构建删除表达式
            if exact_match:
                expr = f'text == "{text_pattern}"'
            else:
                expr = f'text like "%{text_pattern}%"'
            
            status_text.text("🔍 查找匹配记录...")
            progress_bar.progress(0.3)
            
            status_text.text("🗑️ 执行删除操作...")
            progress_bar.progress(0.6)
            
            # 执行删除
            self.collection.delete(expr)
            
            status_text.text("💾 刷新数据到磁盘...")
            progress_bar.progress(0.9)
            
            # 刷新数据
            self.collection.flush()
            
            progress_bar.progress(1.0)
            status_text.text("✅ 删除完成")
            
            st.success(f"✅ 成功删除包含 '{text_pattern}' 的记录")
            return True
            
        except Exception as e:
            st.error(f"❌ 删除记录失败: {e}")
            return False
        finally:
            # 清理进度显示
            if 'progress_bar' in locals():
                progress_bar.empty()
            if 'status_text' in locals():
                status_text.empty()
    
    def clear_all_data(self) -> bool:
        """
        清空集合中的所有数据，推荐通过删除集合后重新创建来保证数据彻底清空
        """
        if not self.collection:
            st.error("❌ 集合未初始化")
            return False

        try:
            st.info(f"🗑️ 正在删除集合 '{self.collection_name}'，以清空所有数据...")
            
            # 释放集合资源
            self.collection.release()
            
            # 删除集合
            utility.drop_collection(self.collection_name)
            
            st.success(f"✅ 集合 '{self.collection_name}' 已删除，数据清空成功")
            
            # 重新创建集合（保持之前的dimension和描述）
            success = self.create_collection(dimension=384, description="文本向量集合")
            if success:
                st.success("✅ 重新创建集合成功，已准备就绪")
                return True
            else:
                st.error("❌ 重新创建集合失败")
                return False

        except Exception as e:
            st.error(f"❌ 清空数据失败: {e}")
            return False

    def get_sample_records(self, limit: int = 10) -> List[Dict]:
        """
        获取样本记录用于预览和选择删除
        """
        if not self.collection:
            st.error("❌ 集合未初始化")
            return []
        
        try:
            # 确保集合已加载
            self.collection.load()
            
            # 查询前N条记录
            results = self.collection.query(
                expr="id >= 0",  # 查询所有记录
                output_fields=["id", "text", "metadata"],
                limit=limit
            )
            
            # 处理结果
            records = []
            for result in results:
                try:
                    metadata = json.loads(result.get('metadata', '{}'))
                except:
                    metadata = {}
                
                records.append({
                    'id': result['id'],
                    'text': result['text'][:100] + '...' if len(result['text']) > 100 else result['text'],
                    'full_text': result['text'],
                    'metadata': metadata
                })
            
            return records
            
        except Exception as e:
            st.error(f"❌ 获取样本记录失败: {e}")
            return []
    
    def search_records_by_text(self, search_text: str, limit: int = 50) -> List[Dict]:
        """
        根据文本内容搜索记录
        """
        if not self.collection:
            st.error("❌ 集合未初始化")
            return []
        
        try:
            # 确保集合已加载
            self.collection.load()
            
            # 构建搜索表达式（模糊匹配）
            expr = f'text like "%{search_text}%"'
            
            # 查询匹配的记录
            results = self.collection.query(
                expr=expr,
                output_fields=["id", "text", "metadata"],
                limit=limit
            )
            
            # 处理结果
            records = []
            for result in results:
                try:
                    metadata = json.loads(result.get('metadata', '{}'))
                except:
                    metadata = {}
                
                records.append({
                    'id': result['id'],
                    'text': result['text'],
                    'metadata': metadata
                })
            
            return records
            
        except Exception as e:
            st.error(f"❌ 搜索记录失败: {e}")
            return []
    
    def search_similar(self, query_vector: np.ndarray, top_k: int = 10) -> List[Dict]:
        """
        搜索相似向量
        """
        if not self.collection:
            st.error("❌ 集合未初始化")
            return []
        
        try:
            # 确保集合已加载到内存
            self.collection.load()
            
            # 搜索参数
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }
            
            # 执行搜索
            results = self.collection.search(
                data=[query_vector.tolist()],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                output_fields=["text", "metadata"]
            )
            
            # 处理结果
            similar_texts = []
            for hits in results:
                for hit in hits:
                    similar_texts.append({
                        'id': hit.id,
                        'text': hit.entity.get('text'),
                        'metadata': json.loads(hit.entity.get('metadata', '{}')),
                        'distance': hit.distance,
                        'score': float(hit.distance)
                    })
            
            return similar_texts
            
        except Exception as e:
            st.error(f"❌ 搜索失败: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        获取集合统计信息
        """
        if not self.collection:
            return {}
        
        try:
            # 确保集合已加载
            self.collection.load()
            
            # 刷新统计信息
            self.collection.flush()
            
            stats = {
                'name': self.collection.name,
                'num_entities': self.collection.num_entities,
                'description': self.collection.description,
                'is_loaded': True
            }
            return stats
        except Exception as e:
            st.error(f"❌ 获取统计信息失败: {e}")
            return {}
    
    def delete_collection(self) -> bool:
        """
        删除集合
        """
        try:
            if utility.has_collection(self.collection_name):
                # 先释放集合
                if self.collection:
                    self.collection.release()
                
                # 删除集合
                utility.drop_collection(self.collection_name)
                st.success(f"✅ 成功删除集合 '{self.collection_name}'")
                self.collection = None
                return True
            else:
                st.info("ℹ️ 集合不存在")
                return True
        except Exception as e:
            st.error(f"❌ 删除集合失败: {e}")
            return False
    
    def disconnect(self):
        """
        断开连接
        """
        try:
            if self.collection:
                self.collection.release()
            connections.disconnect("default")
            self.is_connected = False
            st.info("🔌 已断开Milvus连接")
        except Exception as e:
            st.error(f"❌ 断开连接失败: {e}")
    
    def verify_data_persistence(self) -> Dict[str, Any]:
        """
        验证数据持久化状态
        """
        if not self.is_connected:
            return {"status": "disconnected", "message": "未连接到Milvus"}
        
        try:
            if utility.has_collection(self.collection_name):
                collection = Collection(self.collection_name)
                collection.load()
                
                stats = {
                    "status": "success",
                    "collection_exists": True,
                    "num_entities": collection.num_entities,
                    "is_loaded": True,
                    "message": f"集合存在，包含 {collection.num_entities:,} 条记录"
                }
            else:
                stats = {
                    "status": "no_collection",
                    "collection_exists": False,
                    "num_entities": 0,
                    "is_loaded": False,
                    "message": "集合不存在"
                }
            
            return stats
            
        except Exception as e:
            return {
                "status": "error",
                "collection_exists": False,
                "num_entities": 0,
                "is_loaded": False,
                "message": f"验证失败: {e}"
            }
