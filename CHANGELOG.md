# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/) and this project adheres to
[Semantic Versioning](http://semver.org/spec/v2.0.0.html).

<!-- insertion marker -->

## [v2.2.1](https://github.com/prosto/neo4j-haystack/releases/tag/v2.2.1) - 2025-04-10

<small>[Compare with v2.2.0](https://github.com/prosto/neo4j-haystack/compare/v2.2.0...v2.2.1)</small>

### Dependencies

- For tests making sure numpy version is < 2 to avoid errors ([548cb95](https://github.com/prosto/neo4j-haystack/commit/548cb959d0ce099d75702143b10c37ada6db601b) by Sergey Bondarenco).

### Bug Fixes

- Keep "auth" as part of driver_config only for backward compatibility ([b08bb51](https://github.com/prosto/neo4j-haystack/commit/b08bb51ff9a34bbbf85f1fb241d28211ea7d9133) by Sergey Bondarenco).
- Unify auth container propagation logic ([55412aa](https://github.com/prosto/neo4j-haystack/commit/55412aaa38c1d99ed4f9760d6312c5185b878dfb) by Sergey Bondarenco). [Issue #10](https://github.com/prosto/neo4j-haystack/issues/6)
- Allow custom auth object propagation to database client ([9ac5028](https://github.com/prosto/neo4j-haystack/commit/9ac5028251085f65b45da77e358e080a8b1f2435) by Sergey Bondarenco). [Issue #10](https://github.com/prosto/neo4j-haystack/issues/6)
- Align ChatMessage serialization with latest Haystack version ([98e7e5d](https://github.com/prosto/neo4j-haystack/commit/98e7e5d26cd8e9baf31a123f89d1f0f11fb1521f) by Sergey Bondarenco).
- Base error does not take keyword arguments ([8421b4c](https://github.com/prosto/neo4j-haystack/commit/8421b4cf4dfa7ccc69e65448a61e96353e88ff32) by Sergey Bondarenco). [Issue #10](https://github.com/prosto/neo4j-haystack/issues/6)

### Code Refactoring

- Apply a more idiomatic way to convert a list to a dictionary of component parameters ([c259ab2](https://github.com/prosto/neo4j-haystack/commit/c259ab2195861f9d37c16f675633ed56972576a9) by Sergey Bondarenco).

## [v2.2.0](https://github.com/prosto/neo4j-haystack/releases/tag/v2.2.0) - 2024-12-03

<small>[Compare with v2.1.0](https://github.com/prosto/neo4j-haystack/compare/v2.1.0...v2.2.0)</small>

### Dependencies

- Update Haystack version to >=2.6.0 ([2b9f6be](https://github.com/prosto/neo4j-haystack/commit/2b9f6be6fee8d4cdff5318fb9dc145decd4c7706) by Sergey Bondarenco).

### Bug Fixes

- Remove those dynamic input slots in components which are declared in component's run method ([88e8a47](https://github.com/prosto/neo4j-haystack/commit/88e8a47279717d4fc433d5d32ae926af41c340a3) by Sergey Bondarenco).

## [v2.1.0](https://github.com/prosto/neo4j-haystack/releases/tag/v2.1.0) - 2024-09-17

<small>[Compare with v2.0.3](https://github.com/prosto/neo4j-haystack/compare/v2.0.3...v2.1.0)</small>

### Dependencies

- Update versions of dependencies ([b422aca](https://github.com/prosto/neo4j-haystack/commit/b422aca73c2a449213c1d87128418d0d77d37cb5) by Sergey Bondarenco).

### Features

- Allow Cypher query to be provided during creation of Neo4jDynamicDocumentRetriever component ([ddfbd2e](https://github.com/prosto/neo4j-haystack/commit/ddfbd2e277999d05c7f580c1cb61e0341b91783d) by Sergey Bondarenco).
- Allow a custom marshaller for converting Document to Neo4j record ([9ca023e](https://github.com/prosto/neo4j-haystack/commit/9ca023e059bfd6535eb0666e8c7518453f49ec46) by Sergey Bondarenco).
- Neo4jQueryReader component to run arbitrary Cypher queries to extract data from Neo4j ([bc23597](https://github.com/prosto/neo4j-haystack/commit/bc23597b66342e447a90fb12e9c8874894c9ccf0) by Sergey Bondarenco).
- A module responsible of serialization of complex types in Cypher query parameters ([99ff860](https://github.com/prosto/neo4j-haystack/commit/99ff86009f20adecab1bd38351632b47bf52a031) by Sergey Bondarenco).
- Example RAG pipeline for Parent-Child document setup ([525d166](https://github.com/prosto/neo4j-haystack/commit/525d1665ad43383d1abdea6d3395505f72d21153) by Sergey Bondarenco).
- Add raise_on_failure setting for Neo4jQueryWriter component for better error handling control ([51c819c](https://github.com/prosto/neo4j-haystack/commit/51c819c347d9633d59c404f63c04f5bdec74241e) by Sergey Bondarenco).
- Neo4jQueryWriter component to run Cypher queries which write data to Neo4j ([ceb569a](https://github.com/prosto/neo4j-haystack/commit/ceb569aded92e5657a054fa4fa0fa975ac9fa571) by Sergey Bondarenco).

### Bug Fixes

- Use HuggingFaceAPIGenerator in example scripts as HuggingFaceTGIGenerator is no longer available ([42ed60d](https://github.com/prosto/neo4j-haystack/commit/42ed60d3b873cb7306a4d0be9b5de682c533d8a0) by Sergey Bondarenco).

### Code Refactoring

- Adjust component outputs and parameter serialization for Neo4jQueryWriter ([0d93b21](https://github.com/prosto/neo4j-haystack/commit/0d93b2102c6b677739cb878316c711ddd4a890d2) by Sergey Bondarenco).

## [v2.0.3](https://github.com/prosto/neo4j-haystack/releases/tag/v2.0.3) - 2024-02-08

<small>[Compare with v2.0.2](https://github.com/prosto/neo4j-haystack/compare/v2.0.2...v2.0.3)</small>

### Build

- Update settings as per latest ruff requirements in pyproject.toml ([652d1f1](https://github.com/prosto/neo4j-haystack/commit/652d1f1ac6666d508edde825ed78c93d87ed6c4b) by Sergey Bondarenco).

### Features

- Introducing `execute_write` method in Neo4jClient to run arbitrary Cypher queries which modify data ([88e89bb](https://github.com/prosto/neo4j-haystack/commit/88e89bbe405a72e9185cf56de18aaabcebe71219) by Sergey Bondarenco).

### Bug Fixes

- Return number of written documents from `write_documents` as per Protocol for document stores ([f421ed5](https://github.com/prosto/neo4j-haystack/commit/f421ed54c671c14cabc0fb1a00d5b68c156dda6c) by Sergey Bondarenco).

### Code Refactoring

- Read token values using Secret util ([f75fa32](https://github.com/prosto/neo4j-haystack/commit/f75fa3258a6a53a610c7b7356a891a6ee63f2f08) by Sergey Bondarenco).

## [v2.0.2](https://github.com/prosto/neo4j-haystack/releases/tag/v2.0.2) - 2024-01-19

<small>[Compare with v2.0.1](https://github.com/prosto/neo4j-haystack/compare/v2.0.1...v2.0.2)</small>

### Bug Fixes

- Change imports for DuplicateDocumentError and DuplicatePolicy as per latest changes in haystack ([7a1f053](https://github.com/prosto/neo4j-haystack/commit/7a1f0535b143ef3b4a3e558174e369630079a824) by Sergey Bondarenco).
- Rename 'model_name_or_path' to 'model' as per latest changes in haystack ([0131059](https://github.com/prosto/neo4j-haystack/commit/0131059df8f9966568fea8716d3ba1910801542c) by Sergey Bondarenco).

## [v2.0.1](https://github.com/prosto/neo4j-haystack/releases/tag/v2.0.1) - 2024-01-15

<small>[Compare with v2.0.0](https://github.com/prosto/neo4j-haystack/compare/v2.0.0...v2.0.1)</small>

### Bug Fixes

- Rename metadata slot to meta in example pipelines ([1831d40](https://github.com/prosto/neo4j-haystack/commit/1831d4071bacd1cff4cd99f186cf7a7a1a4d1edc) by Sergey Bondarenco).

## [v2.0.0](https://github.com/prosto/neo4j-haystack/releases/tag/v2.0.0) - 2024-01-15

<small>[Compare with v1.0.0](https://github.com/prosto/neo4j-haystack/compare/v1.0.0...v2.0.0)</small>

### Build

- Update haystack dependency to 2.0 ([bd3e925](https://github.com/prosto/neo4j-haystack/commit/bd3e92543674ab4f3dd8f988a3bc882bbd00042a) by Sergey Bondarenco).

### Features

- Update examples based on haystack 2.0 pipelines ([a7e3bf1](https://github.com/prosto/neo4j-haystack/commit/a7e3bf1788ac9f6b87e82497740feea056386f87) by Sergey Bondarenco).
- Retriever component for documents stored in Neo4j ([b411ebc](https://github.com/prosto/neo4j-haystack/commit/b411ebc5f850272e0050307f03cc6157b7bc6e26) by Sergey Bondarenco).
- Update DocumentStore protocol implementation to match haystack 2.0 requirements ([9748e7d](https://github.com/prosto/neo4j-haystack/commit/9748e7d4f27087b80c8f028b8612f76ed1daf8a8) by Sergey Bondarenco).
- Update metadata filter parser for document store ([6ce780c](https://github.com/prosto/neo4j-haystack/commit/6ce780c846576d690b7216e37793532841a54dc3) by Sergey Bondarenco).

### Code Refactoring

- Organize modules into packages for better separation of concerns ([6a101e8](https://github.com/prosto/neo4j-haystack/commit/6a101e8047bcd2dac2b49598701f7233390bae88) by Sergey Bondarenco).
- Change name of retriever component as per documented naming convention ([f79a952](https://github.com/prosto/neo4j-haystack/commit/f79a952fbe59be0d1d5d13e03ae58401f6403ce9) by Sergey Bondarenco).

## [v1.0.0](https://github.com/prosto/neo4j-haystack/releases/tag/v1.0.0) - 2023-12-19

<small>[Compare with first commit](https://github.com/prosto/neo4j-haystack/compare/f801a10c8cf6eb7d784c77d8b72005cf5985dffc...v1.0.0)</small>

### Build

- Script to bump a new release by tagging commit and generating changelog ([84a923d](https://github.com/prosto/neo4j-haystack/commit/84a923dc5d8b1f5ff8602fbdf4f86ff5c682e565) by Sergey Bondarenco).

### Features

- Example of Neo4jDocumentStore used in a question answering pipeline ([e7628c6](https://github.com/prosto/neo4j-haystack/commit/e7628c672489f609c14d539859d110e8facda848) by Sergey Bondarenco).
- Sample script to download movie data from HF datasets ([ed44127](https://github.com/prosto/neo4j-haystack/commit/ed44127329454b555e906e1b5463fa8b9f4e8fe7) by Sergey Bondarenco).
- Project setup and initial implementation of Neo4jDataStore ([f801a10](https://github.com/prosto/neo4j-haystack/commit/f801a10c8cf6eb7d784c77d8b72005cf5985dffc) by Sergey Bondarenco).
