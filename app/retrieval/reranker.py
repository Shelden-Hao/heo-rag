from app.generation.llm import chat


# 精排，从已有检索结果中，再次挑选（LLM自行判断）最相关的文档
def rerank(query: str, results: list, top_k: int = 3) -> list:
    if not results or len(results) <= top_k:
        return results

    prompt = f"""你是一个文档排序专家。给定用户问题和多个候选文档片段，请判断每个片段与问题的相关性，只返回最相关的 {top_k} 个片段的序号。

用户问题：{query}

候选片段：
"""
    for i, doc in enumerate(results):
        content = doc.page_content if hasattr(doc, "page_content") else str(doc)
        heading = doc.metadata.get("heading", "")
        prompt += f"\n[{i}] 【标题：{heading}】\n{content}"

    prompt += f"\n\n请直接返回最相关的 {top_k} 个序号，用逗号分隔，例如：0,2,3"

    response = chat(prompt, model="deepseek-chat", system="你只返回序号，不返回其他内容。")
    try:
        indices = [int(x.strip()) for x in response.strip().split(",") if x.strip().isdigit()]
        return [results[i] for i in indices if i < len(results)]
    except (ValueError, IndexError):
        return results[:top_k]
