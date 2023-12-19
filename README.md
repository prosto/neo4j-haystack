<h1 align="center">neo4j-haystack</h1>

<p align="center">A <a href="https://docs.haystack.deepset.ai/docs/document_store"><i>Haystack</i></a> Document Store for <a href="https://neo4j.com/"><i>Neo4j</i></a>.</p>

<p align="center">
  <a href="https://github.com/prosto/neo4j-haystack/actions?query=workflow%3Aci">
    <img alt="ci" src="https://github.com/prosto/neo4j-haystack/workflows/ci/badge.svg" />
  </a>
  <a href="https://prosto.github.io/neo4j-haystack/">
    <img alt="documentation" src="https://img.shields.io/badge/docs-mkdocs%20material-blue.svg?style=flat" />
  </a>
  <a href="https://pypi.org/project/neo4j-haystack/">
    <img alt="pypi version" src="https://img.shields.io/pypi/v/neo4j-haystack.svg" />
  </a>
  <a href="https://img.shields.io/pypi/pyversions/neo4j-haystack.svg">
    <img alt="python version" src="https://img.shields.io/pypi/pyversions/neo4j-haystack.svg" />
  </a>
</p>

----

**Table of Contents**

- [Overview](#overview)
- [Usage](#usage)
- [Installation](#installation)
- [License](#license)

## Overview

An integration of [Neo4j](https://neo4j.com/) graph database with [Haystack](https://haystack.deepset.ai/)
by [deepset](https://www.deepset.ai). In Neo4j [Vector search index](https://neo4j.com/docs/cypher-manual/current/indexes-for-vector-search/)
is being used for storing document embeddings and dense retrials.

The library allows using Neo4j as a [DocumentStore](https://docs.haystack.deepset.ai/docs/document_store), and provides an in-place replacement
for any other vector embeddings store. Thus, you should expect any kind of application to be working
smoothly just by changing the provider to `Neo4jDocumentStore`.

The key difference between `Neo4jDocumentStore` and other types of stores is that Document properties are stored as Graph nodes. Embeddings are stored as properties of a Document node,
but indexing and querying of vector embeddings using approximate nearest neighbor search is managed by a dedicated Vector Index.

```text
                                   +-----------------------------+
                                   |       Neo4j Database        |
                                   +-----------------------------+
                                   |                             |
                                   |      +----------------+     |
                                   |      |    Document    |     |
                write_documents    |      +----------------+     |
          +------------------------+----->|   properties   |     |
          |                        |      |                |     |
+---------+----------+             |      |   embedding    |     |
|                    |             |      +--------+-------+     |
| Neo4jDocumentStore |             |               |             |
|                    |             |               |index/query  |
+---------+----------+             |               |             |
          |                        |      +--------+--------+    |
          |                        |      |  Vector Index   |    |
          +----------------------->|      |                 |    |
               query_embeddings    |      | (for embedding) |    |
                                   |      +-----------------+    |
                                   |                             |
                                   +-----------------------------+
```

In the above diagram:

- `Document` is a Neo4j node (with "Document" label)
- `properties` are Document [attributes](https://docs.haystack.deepset.ai/docs/documents_answers_labels#attributes) stored as part of the node.
- `embedding` is also a property of the Document node (just shown separately in the diagram for clarity) which is a vector of type `LIST[FLOAT]`
- `Vector Index` is where embeddings are getting indexed by Neo4j (as soon as those are updated in Document nodes)

The `neo4j-haystack` library uses [python-driver](https://neo4j.com/docs/api/python-driver/current/api.html#api-documentation) and
[Cypher Queries](https://neo4j.com/docs/cypher-manual/current/introduction/) to implement DocumentStore related API methods and hide all complexities under the hood.

## Installation

`neo4j-haystack` can be installed as any other Python library, using pip:

```bash
pip install --upgrade pip # optional
pip install neo4j-haystack
```

## Usage

Once installed, you can start using `Neo4jDocumentStore` as any other document stores that support embeddings.

```python
from neo4j_haystack import Neo4jDocumentStore

document_store = Neo4jDocumentStore(
    url="bolt://localhost:7687",
    username="neo4j",
    password="passw0rd",
    database="neo4j",
    embedding_dim=384,
    embedding_field="embedding",
    index="document-embeddings", # The name of the Vector Index in Neo4j
    node_label="Document", # Providing a label to Neo4j nodes which store Documents
)
```

The full list of parameters accepted by `Neo4jDocumentStore` can be found in
[API documentation](https://prosto.github.io/neo4j-haystack/reference/neo4j_store/#neo4j_haystack.document_stores.neo4j_store.Neo4jDocumentStore.__init__).

Please notice you will need to have a running instance of Neo4j database (in-memory version of Neo4j is not supported). There are several options available:

- [Docker](https://neo4j.com/docs/operations-manual/5/docker/), other options available in the same Operations Manual
- [AuraDB](https://neo4j.com/cloud/platform/aura-graph-database/) - a fully managed Cloud Instance of Neo4j
- [Neo4j Desktop](https://neo4j.com/docs/desktop-manual/current/) client application

The simplest way to start database locally will be with Docker container:

```bash
docker run \
    --restart always \
    --publish=7474:7474 --publish=7687:7687 \
    --env NEO4J_AUTH=neo4j/passw0rd \
    neo4j:5.15.0
```

## License

`neo4j-haystack` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
