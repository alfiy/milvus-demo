# milvus demo

## 🚀 本地运行步骤

### 1.环境准备

```bash
# 创建Python虚拟环境（推荐）
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 2.安装依赖

开发环境

```
sudo apt-get install qtbase5-dev qt5-qmake qtbase5-dev-tools
sudo apt-get install qtbase5-dev qtbase5-dev-tools
```

项目依赖

```bash
# 进入项目目录
cd streamlit_template

# 安装所有依赖包
pip install -r requirements.txt
```



如果使用国内源可能会出现`pymilvus`和`sentence-transformers`一些包无法安装的问题，使用下面的命令指定源安装。

```
# 在已激活的 .venv 里执行
pip install -i https://pypi.org/simple --upgrade pip setuptools wheel
pip install -i https://pypi.org/simple pymilvus
```

```
# 确保还在 .venv 内
pip install -i https://pypi.org/simple --upgrade pip setuptools wheel
pip install -i https://pypi.org/simple sentence-transformers
```



```
 pip install -i https://pypi.org/simple -r requirements.txt
```





### 3.启动应用

```bash
# 运行Streamlit应用
streamlit run app.py
```

### 4.访问应用

- 应用启动后，浏览器会自动打开 `http://localhost:8501`
- 如果没有自动打开，请手动访问该地址

 📋 使用流程

1. **首页概览** - 查看系统状态和功能介绍

2. **数据上传** - 上传您的JSON数据或使用示例数据进行测试

3. **向量化处理** - 点击"开始向量化处理"按钮

4. 功能体验

   ：

   - 🔍 **文本搜索** - 输入查询内容进行语义搜索
   - 🎯 **聚类分析** - 选择算法进行文本聚类
   - 🗄️ **Milvus管理** - 如果有Milvus服务器可以连接测试

 ⚠️ 注意事项

- **首次运行**会自动下载向量化模型（约200MB），请确保网络连接正常
- **Milvus数据库**是可选功能，没有也能正常使用本地搜索和聚类
- 如果遇到依赖安装问题，可以尝试升级pip：`pip install --upgrade pip`

 🔧 可能遇到的问题

1. **模型下载慢**：首次运行时会下载sentence-transformers模型
2. **内存不足**：处理大量数据时可能需要较多内存
3. **端口占用**：如果8501端口被占用，可以指定其他端口：`streamlit run app.py --server.port 8502`

项目已经完全可以独立运行，请按照上述步骤启动并测试各项功能！
