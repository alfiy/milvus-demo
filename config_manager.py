# config_manager.py
import json
import os
import streamlit as st
from typing import Dict, Any, Optional

class ConfigManager:
    """配置管理器，用于保存和加载用户配置"""
    
    def __init__(self, config_file: str = "milvus_config.json"):
        """
        初始化配置管理器
        
        Args:
            config_file: 配置文件路径
        """
        self.config_file = config_file
        self.default_config = {
            "milvus": {
                "host": "localhost",
                "port": "19530",
                "user": "",
                "password": "",
                "collection_name": "text_vectors",
                "auto_connect": True
            },
            "mongodb": {
                "host": "localhost",
                "port": 27017,
                "username": "",
                "password": "",
                "db_name": "textdb",
                "col_name": "metadata",
                "auto_connect": False
            },
            "model": {
                "last_used_model": "",
                "auto_load": False
            }
        }
    
    def load_config(self) -> Dict[str, Any]:
        """
        加载配置文件
        
        Returns:
            配置字典
        """
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    # 合并默认配置，确保所有字段都存在
                    return self._merge_config(self.default_config, config)
            else:
                return self.default_config.copy()
        except Exception as e:
            st.error(f"❌ 加载配置文件失败: {e}")
            return self.default_config.copy()
    
    def save_config(self, config: Dict[str, Any]) -> bool:
        """
        保存配置到文件
        
        Args:
            config: 配置字典
            
        Returns:
            是否保存成功
        """
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            st.error(f"❌ 保存配置文件失败: {e}")
            return False
    
    def _merge_config(self, default: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
        """
        合并默认配置和用户配置
        
        Args:
            default: 默认配置
            user: 用户配置
            
        Returns:
            合并后的配置
        """
        result = default.copy()
        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_config(result[key], value)
            else:
                result[key] = value
        return result
    
    def update_milvus_config(self, host: str, port: str, user: str = "", password: str = "", 
                           collection_name: str = "text_vectors", auto_connect: bool = True) -> bool:
        """
        更新Milvus配置
        
        Args:
            host: 主机地址
            port: 端口
            user: 用户名
            password: 密码
            collection_name: 集合名称
            auto_connect: 是否自动连接
            
        Returns:
            是否更新成功
        """
        config = self.load_config()
        config["milvus"].update({
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "collection_name": collection_name,
            "auto_connect": auto_connect
        })
        return self.save_config(config)
    
    def update_mongodb_config(self, host: str, port: int, username: str = "", password: str = "",
                            db_name: str = "textdb", col_name: str = "metadata", 
                            auto_connect: bool = False) -> bool:
        """
        更新MongoDB配置
        
        Args:
            host: 主机地址
            port: 端口
            username: 用户名
            password: 密码
            db_name: 数据库名
            col_name: 集合名
            auto_connect: 是否自动连接
            
        Returns:
            是否更新成功
        """
        config = self.load_config()
        config["mongodb"].update({
            "host": host,
            "port": port,
            "username": username,
            "password": password,
            "db_name": db_name,
            "col_name": col_name,
            "auto_connect": auto_connect
        })
        return self.save_config(config)
    
    def update_model_config(self, last_used_model: str, auto_load: bool = False) -> bool:
        """
        更新模型配置
        
        Args:
            last_used_model: 最后使用的模型
            auto_load: 是否自动加载
            
        Returns:
            是否更新成功
        """
        config = self.load_config()
        config["model"].update({
            "last_used_model": last_used_model,
            "auto_load": auto_load
        })
        return self.save_config(config)
    
    def get_milvus_config(self) -> Dict[str, Any]:
        """获取Milvus配置"""
        return self.load_config().get("milvus", {})
    
    def get_mongodb_config(self) -> Dict[str, Any]:
        """获取MongoDB配置"""
        return self.load_config().get("mongodb", {})
    
    def get_model_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        return self.load_config().get("model", {})
    
    def reset_config(self) -> bool:
        """重置配置为默认值"""
        return self.save_config(self.default_config.copy())
    
    def export_config(self, export_path: str) -> bool:
        """
        导出配置到指定路径
        
        Args:
            export_path: 导出路径
            
        Returns:
            是否导出成功
        """
        try:
            config = self.load_config()
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            st.error(f"❌ 导出配置失败: {e}")
            return False
    
    def import_config(self, import_path: str) -> bool:
        """
        从指定路径导入配置
        
        Args:
            import_path: 导入路径
            
        Returns:
            是否导入成功
        """
        try:
            if os.path.exists(import_path):
                with open(import_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    merged_config = self._merge_config(self.default_config, config)
                    return self.save_config(merged_config)
            else:
                st.error(f"❌ 配置文件不存在: {import_path}")
                return False
        except Exception as e:
            st.error(f"❌ 导入配置失败: {e}")
            return False

# 全局配置管理器实例
config_manager = ConfigManager()
