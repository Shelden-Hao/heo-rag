from langchain_core.prompts import ChatPromptTemplate, PromptTemplate


SYSTEM_PROMPT = """你是秀秀训练营的智能知识库助手。请基于提供的上下文内容，准确、简洁地回答用户问题。
要求：
1. 如果上下文中包含答案，请直接用中文回答
2. 如果上下文中没有足够的信息，请如实说不知道
3. 标注引用来源（文档名称）
4. 不要编造信息"""


RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", """请基于以下参考资料回答问题。

参考资料：
{context}

用户问题：{question}

请回答问题，并在答案末尾标注引用来源。"""),
])


QA_PROMPT = PromptTemplate(
    template="""请基于以下上下文回答问题。如果上下文中没有足够信息，请说"我不知道"。

上下文：
{context}

问题：{question}

回答：""",
    input_variables=["context", "question"],
)
