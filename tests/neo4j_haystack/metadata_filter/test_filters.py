from contextlib import nullcontext as does_not_raise
from dataclasses import dataclass, field
from typing import Any, ContextManager, Dict, Literal

import pytest

from neo4j_haystack.errors import Neo4jFilterParserError
from neo4j_haystack.metadata_filter import FilterParser, FilterType, Neo4jQueryConverter


@dataclass
class TestCase:
    id: str
    filters: FilterType
    cypher_query: str = field(default="")
    params: Dict[str, Any] = field(default_factory=dict)
    expectation: ContextManager = field(default_factory=does_not_raise)

    def as_param(self):
        return pytest.param(self.filters, self.cypher_query, self.params, self.expectation, id=self.id)


def in_comparison_cypher(field_name: str, op: Literal["IN", "NOT IN"], field_param_name: str, prefix="doc"):
    """A helper function to generate "IN / NOT IN" Cypher statement"""

    not_expr = "NOT " if op == "NOT IN" else ""
    return (
        f"CASE valueType({prefix}.{field_name}) STARTS WITH 'LIST' "
        f"WHEN true THEN any(val IN {prefix}.{field_name} WHERE {not_expr}val IN {field_param_name}) "
        f"ELSE {not_expr}{prefix}.{field_name} IN {field_param_name} END"
    )


TEST_CASES = [
    TestCase(
        "== str",
        filters={"field": "name", "operator": "==", "value": "test"},
        cypher_query="doc.name = $fv_name",
        params={"fv_name": "test"},
    ),
    TestCase(
        "== num",
        filters={"field": "num", "operator": "==", "value": 11},
        cypher_query="doc.num = $fv_num",
        params={"fv_num": 11},
    ),
    TestCase(
        "== bool",
        filters={"field": "flag", "operator": "==", "value": True},
        cypher_query="doc.flag = $fv_flag",
        params={"fv_flag": True},
    ),
    TestCase(
        "!= num",
        filters={"field": "num", "operator": "!=", "value": 10},
        cypher_query="doc.num <> $fv_num",
        params={"fv_num": 10},
    ),
    TestCase(
        "!= str",
        filters={"field": "name", "operator": "!=", "value": "test"},
        cypher_query="doc.name <> $fv_name",
        params={"fv_name": "test"},
    ),
    TestCase(
        "!= bool",
        filters={"field": "flag", "operator": "!=", "value": False},
        cypher_query="doc.flag <> $fv_flag",
        params={"fv_flag": False},
    ),
    TestCase(
        "> num",
        filters={"field": "num", "operator": ">", "value": 10},
        cypher_query="doc.num > $fv_num",
        params={"fv_num": 10},
    ),
    TestCase(
        ">= num",
        filters={"field": "num", "operator": ">=", "value": 10},
        cypher_query="doc.num >= $fv_num",
        params={"fv_num": 10},
    ),
    TestCase(
        "< num",
        filters={"field": "num", "operator": "<", "value": 10},
        cypher_query="doc.num < $fv_num",
        params={"fv_num": 10},
    ),
    TestCase(
        "<= num",
        filters={"field": "num", "operator": "<=", "value": 10},
        cypher_query="doc.num <= $fv_num",
        params={"fv_num": 10},
    ),
    TestCase(
        "in str_list",
        filters={"field": "name", "operator": "in", "value": ["test1", "test2"]},
        cypher_query=in_comparison_cypher("name", "IN", "$fv_name"),
        params={"fv_name": ["test1", "test2"]},
    ),
    TestCase(
        "in number_list",
        filters={"field": "num", "operator": "in", "value": [1, 2]},
        cypher_query=in_comparison_cypher("num", "IN", "$fv_num"),
        params={"fv_num": [1, 2]},
    ),
    TestCase(
        "in number_set",
        filters={"field": "num", "operator": "in", "value": {2, 1}},
        cypher_query=in_comparison_cypher("num", "IN", "$fv_num"),
        params={"fv_num": [1, 2]},
    ),
    TestCase(
        "not in str_list",
        filters={"field": "name", "operator": "not in", "value": ["test1", "test2"]},
        cypher_query=in_comparison_cypher("name", "NOT IN", "$fv_name"),
        params={"fv_name": ["test1", "test2"]},
    ),
    TestCase(
        "not in num_set",
        filters={"field": "num", "operator": "not in", "value": {5, 2}},
        cypher_query=in_comparison_cypher("num", "NOT IN", "$fv_num"),
        params={"fv_num": [2, 5]},
    ),
    TestCase(
        "not in num_tuple",
        filters={"field": "num", "operator": "not in", "value": (1, 2)},
        cypher_query=in_comparison_cypher("num", "NOT IN", "$fv_num"),
        params={"fv_num": (1, 2)},
    ),
    TestCase(
        "exists true",
        filters={"field": "name", "operator": "exists", "value": True},
        cypher_query="doc.name IS NOT NULL",
    ),
    TestCase(
        "exists false", filters={"field": "name", "operator": "exists", "value": False}, cypher_query="doc.name IS NULL"
    ),
    TestCase(
        "AND many_operands",
        filters={
            "operator": "AND",
            "conditions": [
                {"field": "type", "operator": "==", "value": "article"},
                {"field": "rating", "operator": ">=", "value": 3},
            ],
        },
        cypher_query="(doc.type = $fv_type AND doc.rating >= $fv_rating)",
        params={"fv_type": "article", "fv_rating": 3},
    ),
    TestCase(
        "AND single_operand",
        filters={
            "operator": "AND",
            "conditions": [
                {"field": "rating", "operator": "==", "value": 3},
            ],
        },
        cypher_query="doc.rating = $fv_rating",
        params={"fv_rating": 3},
    ),
    TestCase(
        "OR many_operands",
        filters={
            "operator": "OR",
            "conditions": [
                {"field": "type", "operator": "==", "value": "article"},
                {"field": "rating", "operator": ">=", "value": 3},
            ],
        },
        cypher_query="(doc.type = $fv_type OR doc.rating >= $fv_rating)",
        params={"fv_type": "article", "fv_rating": 3},
    ),
    TestCase(
        "OR single_operand",
        filters={
            "operator": "OR",
            "conditions": [
                {"field": "type", "operator": "==", "value": "article"},
            ],
        },
        cypher_query="doc.type = $fv_type",
        params={"fv_type": "article"},
    ),
    TestCase(
        "flatten_field_name_with_underscores",
        filters={
            "operator": "AND",
            "conditions": [
                {"field": "meta.type", "operator": "==", "value": "article"},
                {"field": "meta.rating", "operator": ">=", "value": 3},
                {"field": "likes", "operator": ">", "value": 100},
            ],
        },
        cypher_query="(doc.meta_type = $fv_meta_type AND doc.meta_rating >= $fv_meta_rating AND doc.likes > $fv_likes)",
        params={"fv_meta_type": "article", "fv_meta_rating": 3, "fv_likes": 100},
    ),
    TestCase(
        "complex_query",
        filters={
            "operator": "AND",
            "conditions": [
                {"field": "type", "operator": "==", "value": "article"},
                {"field": "year", "operator": ">=", "value": 2015},
                {"field": "year", "operator": "<", "value": 2021},
                {"field": "rating", "operator": ">=", "value": 3},
                {
                    "operator": "OR",
                    "conditions": [
                        {"field": "genre", "operator": "in", "value": ["economy", "politics"]},
                        {"field": "publisher", "operator": "==", "value": "nytimes"},
                    ],
                },
            ],
        },
        cypher_query=(
            "(doc.type = $fv_type AND doc.year >= $fv_year AND doc.year < $fv_year_1 AND doc.rating >= $fv_rating AND"
            f" ({in_comparison_cypher('genre', 'IN', '$fv_genre')} OR doc.publisher = $fv_publisher))"
        ),
        params={
            "fv_type": "article",
            "fv_year": 2015,
            "fv_year_1": 2021,
            "fv_rating": 3,
            "fv_genre": ["economy", "politics"],
            "fv_publisher": "nytimes",
        },
    ),
    TestCase(
        "logical_operators_same_level",
        filters={
            "operator": "OR",
            "conditions": [
                {
                    "operator": "AND",
                    "conditions": [
                        {"field": "type", "operator": "==", "value": "news"},
                        {"field": "likes", "operator": "<", "value": 100},
                    ],
                },
                {
                    "operator": "AND",
                    "conditions": [
                        {"field": "type", "operator": "==", "value": "blog"},
                        {"field": "likes", "operator": ">=", "value": 500},
                    ],
                },
            ],
        },
        cypher_query=(
            "((doc.type = $fv_type AND doc.likes < $fv_likes) OR (doc.type = $fv_type_1 AND doc.likes >= $fv_likes_1))"
        ),
        params={"fv_type": "news", "fv_likes": 100, "fv_type_1": "blog", "fv_likes_1": 500},
    ),
    TestCase(
        "empty_filter",
        filters={},
        expectation=pytest.raises(Neo4jFilterParserError),
    ),
    TestCase(
        "unsupported_logical_operator",
        filters={
            "operator": "XOR",
            "conditions": [
                {"field": "type", "operator": "==", "value": "blog"},
                {"field": "likes", "operator": ">=", "value": 500},
            ],
        },
        expectation=pytest.raises(Neo4jFilterParserError),
    ),
    TestCase(
        "unsupported_comparison_operator",
        filters={
            "operator": "XOR",
            "conditions": [
                {"field": "type", "operator": "==", "value": "blog"},
                {"field": "likes", "operator": "mod", "value": 500},  # here
            ],
        },
        expectation=pytest.raises(Neo4jFilterParserError),
    ),
    TestCase(
        "empty_conditions",
        filters={"operator": "AND", "conditions": []},
        expectation=pytest.raises(Neo4jFilterParserError),
    ),
    TestCase(
        "wrong_conditions_type",
        filters={"operator": "AND", "conditions": 2},
        expectation=pytest.raises(Neo4jFilterParserError),
    ),
    TestCase(
        "no_logical_operator_provided",
        filters={"conditions": [{"field": "type", "operator": "==", "value": "blog"}]},
        expectation=pytest.raises(Neo4jFilterParserError),
    ),
    TestCase(
        "no_comparison_operator_provided",
        filters={"operator": "AND", "conditions": [{"field": "type", "value": "blog"}]},
        expectation=pytest.raises(Neo4jFilterParserError),
    ),
]


@pytest.mark.unit
@pytest.mark.parametrize("filters,expected_cypher,expected_params,expectation", [tc.as_param() for tc in TEST_CASES])
def test_filter_converter(filters, expected_cypher, expected_params, expectation):
    parser = FilterParser()
    converter = Neo4jQueryConverter("doc")

    with expectation:
        filter_ast = parser.parse(filters)
        cypher_query, params = converter.convert(filter_ast)

        assert cypher_query == expected_cypher
        assert params == expected_params
