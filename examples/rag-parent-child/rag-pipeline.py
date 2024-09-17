from haystack import GeneratedAnswer, Pipeline
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.generators import HuggingFaceAPIGenerator
from haystack.utils.auth import Secret

from neo4j_haystack import Neo4jClientConfig, Neo4jDynamicDocumentRetriever

# Load HF Token from environment variables.
HF_TOKEN = Secret.from_env_var("HF_API_TOKEN")

# Make sure you have a running Neo4j database with indexed documents available (run `indexing_pipeline.py` first
# and keep docker running)
client_config = Neo4jClientConfig(
    url="bolt://localhost:7687",
    username="neo4j",
    password="passw0rd",
    database="neo4j",
)

cypher_query = """
    // Query Child documents by $query_embedding
    CALL db.index.vector.queryNodes($index, $top_k, $query_embedding)
    YIELD node as chunk, score

    // Find Parent document for previously retrieved child (e.g. extend RAG context)
    MATCH (chunk)-[:HAS_PARENT]->(parent:Document)
    WITH parent, max(score) AS score // deduplicate parents
    RETURN parent{.*, score}
"""

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
    SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2", progress_bar=False),
)
rag_pipeline.add_component(
    "retriever",
    Neo4jDynamicDocumentRetriever(
        client_config=client_config,
        runtime_parameters=["query_embedding"],
        doc_node_name="parent",
        verify_connectivity=True,
    ),
)
rag_pipeline.add_component("prompt_builder", PromptBuilder(template=prompt_template))
rag_pipeline.add_component(
    "llm",
    HuggingFaceAPIGenerator(
        api_type="serverless_inference_api",
        api_params={"model": "mistralai/Mistral-7B-Instruct-v0.3"},
        generation_kwargs={"max_new_tokens": 120, "stop_sequences": ["."]},
        token=HF_TOKEN,
    ),
)
rag_pipeline.add_component("answer_builder", AnswerBuilder())

rag_pipeline.connect("query_embedder", "retriever.query_embedding")
rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder.prompt", "llm.prompt")
rag_pipeline.connect("llm.replies", "answer_builder.replies")
rag_pipeline.connect("llm.meta", "answer_builder.meta")
rag_pipeline.connect("retriever", "answer_builder.documents")

question = "Why did author suppressed technology in the Dune universe?"
result = rag_pipeline.run(
    {
        "query_embedder": {"text": question},
        "retriever": {
            "query": cypher_query,
            "parameters": {"index": "chunk-embeddings", "top_k": 5},
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
    print("-> Score: ", doc.score, "Content: ", doc.content[:200] + "...")
