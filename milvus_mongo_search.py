def search(query_vector, top_k, collection, mongo_col):
    # Milvus ANN 检索
    results = collection.search(
        [query_vector],
        "embedding",
        params={"metric_type": "IP"},  # 余弦/内积
        limit=top_k
    )
    ids = [hit.id for hit in results[0]]
    scores = [hit.distance for hit in results[0]]
    # MongoDB 批量查元数据
    docs = list(mongo_col.find({"_id": {"$in": ids}}))
    # 按检索顺序排序
    id2doc = {doc["_id"]: doc for doc in docs}
    combined = [
        {
            "id": id,
            "score": score,
            "text": id2doc[id]["text"] if id in id2doc else "",
            "metadata": id2doc[id]["metadata"] if id in id2doc else {},
        }
        for id, score in zip(ids, scores)
    ]
    return combined