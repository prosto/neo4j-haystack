# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/) and this project adheres to
[Semantic Versioning](http://semver.org/spec/v2.0.0.html).

<!-- insertion marker -->
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
