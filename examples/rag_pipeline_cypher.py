import os

from haystack import GeneratedAnswer, Pipeline
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.generators import HuggingFaceTGIGenerator

from neo4j_haystack import Neo4jClientConfig, Neo4jDynamicDocumentRetriever

# Load HF Token from environment variables.
HF_TOKEN = os.environ.get("HF_TOKEN")

# Make sure you have a running Neo4j database with indexed documents available (see `indexing_pipeline.py`),
# e.g. with Docker:
# docker run \
#     --restart always \
#     --publish=7474:7474 --publish=7687:7687 \
#     --env NEO4J_AUTH=neo4j/passw0rd \
#     neo4j:5.15.0

client_config = Neo4jClientConfig(
    url="bolt://localhost:7687",
    username="neo4j",
    password="passw0rd",
    database="neo4j",
)

cypher_query = """
            CALL db.index.vector.queryNodes($index, $top_k, $query_embedding)
            YIELD node as doc, score
            MATCH (doc)
            RETURN doc{.*, score}, score
            ORDER BY score DESC LIMIT $top_k
        """

# Build a RAG pipeline with a Retriever to get relevant documents to the query and a HuggingFaceTGIGenerator
# interacting with LLMs using a custom prompt.
prompt_template = """
Given these documents, answer the question.\nDocuments:
{% for doc in documents %}
    {{ doc.content }}
{% endfor %}

\nQuestion: {{question}}
\nAnswer:
"""
rag_pipeline = Pipeline()
rag_pipeline.add_component(
    "query_embedder",
    SentenceTransformersTextEmbedder(model_name_or_path="sentence-transformers/all-MiniLM-L6-v2", progress_bar=False),
)
rag_pipeline.add_component(
    "retriever",
    Neo4jDynamicDocumentRetriever(
        client_config=client_config,
        runtime_parameters=["query_embedding"],
        doc_node_name="doc",
        verify_connectivity=True,
    ),
)
rag_pipeline.add_component("prompt_builder", PromptBuilder(template=prompt_template))
rag_pipeline.add_component(
    "llm",
    HuggingFaceTGIGenerator(model="mistralai/Mistral-7B-v0.1", token=HF_TOKEN),
)
rag_pipeline.add_component("answer_builder", AnswerBuilder())

rag_pipeline.connect("query_embedder", "retriever.query_embedding")
rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder.prompt", "llm.prompt")
rag_pipeline.connect("llm.replies", "answer_builder.replies")
rag_pipeline.connect("llm.metadata", "answer_builder.metadata")
rag_pipeline.connect("retriever", "answer_builder.documents")

# Ask a question on the data you just added.
question = "Who created the Dothraki vocabulary?"
result = rag_pipeline.run(
    {
        "query_embedder": {"text": question},
        "retriever": {
            "query": cypher_query,
            "parameters": {"index": "document-embeddings", "top_k": 3},
        },
        "prompt_builder": {"question": question},
        "answer_builder": {"query": question},
    }
)

# For details, like which documents were used to generate the answer, look into the GeneratedAnswer object
answer: GeneratedAnswer = result["answer_builder"]["answers"][0]

# ruff: noqa: T201
print("Query: ", answer.query)
print("Answer: ", answer.data)
print("== Sources:")
for doc in answer.documents:
    print("-> ", doc.meta["file_path"])
