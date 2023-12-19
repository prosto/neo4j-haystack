from contextlib import nullcontext as does_not_raise
from dataclasses import dataclass, field
from typing import Any, ContextManager, Dict, Literal

import pytest
from haystack.schema import FilterType

from neo4j_haystack.document_stores.errors import Neo4jFilterParserError
from neo4j_haystack.document_stores.filters import FilterParser, Neo4jFiltersConverter


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
        "eq_implicit_str", filters={"name": "test"}, cypher_query="doc.name = $fv_name", params={"fv_name": "test"}
    ),
    TestCase("eq_implicit_num", filters={"num": 10.25}, cypher_query="doc.num = $fv_num", params={"fv_num": 10.25}),
    TestCase(
        "eq_implicit_bool", filters={"flag": False}, cypher_query="doc.flag = $fv_flag", params={"fv_flag": False}
    ),
    TestCase(
        "eq_explicit_str",
        filters={"name": {"$eq": "test"}},
        cypher_query="doc.name = $fv_name",
        params={"fv_name": "test"},
    ),
    TestCase("eq_explicit_num", filters={"num": {"$eq": 11}}, cypher_query="doc.num = $fv_num", params={"fv_num": 11}),
    TestCase(
        "eq_explicit_bool",
        filters={"flag": {"$eq": True}},
        cypher_query="doc.flag = $fv_flag",
        params={"fv_flag": True},
    ),
    TestCase("ne_num", filters={"num": {"$ne": 10}}, cypher_query="doc.num <> $fv_num", params={"fv_num": 10}),
    TestCase(
        "ne_str", filters={"name": {"$ne": "test"}}, cypher_query="doc.name <> $fv_name", params={"fv_name": "test"}
    ),
    TestCase(
        "ne_bool", filters={"flag": {"$ne": False}}, cypher_query="doc.flag <> $fv_flag", params={"fv_flag": False}
    ),
    TestCase("gt_num", filters={"num": {"$gt": 10}}, cypher_query="doc.num > $fv_num", params={"fv_num": 10}),
    TestCase("gte_num", filters={"num": {"$gte": 10}}, cypher_query="doc.num >= $fv_num", params={"fv_num": 10}),
    TestCase("lt_num", filters={"num": {"$lt": 10}}, cypher_query="doc.num < $fv_num", params={"fv_num": 10}),
    TestCase("lte_num", filters={"num": {"$lte": 10}}, cypher_query="doc.num <= $fv_num", params={"fv_num": 10}),
    TestCase(
        "in_implicit_str_list",
        filters={"name": ["test1", "test2"]},
        cypher_query=in_comparison_cypher("name", "IN", "$fv_name"),
        params={"fv_name": ["test1", "test2"]},
    ),
    TestCase(
        "in_implicit_str_set",
        filters={"name": {"0_test", "1_test"}},
        cypher_query=in_comparison_cypher("name", "IN", "$fv_name"),
        params={"fv_name": ["0_test", "1_test"]},
    ),
    TestCase(
        "in_implicit_str_tuple",
        filters={"name": ("test1", "test2")},
        cypher_query=in_comparison_cypher("name", "IN", "$fv_name"),
        params={"fv_name": ("test1", "test2")},
    ),
    TestCase(
        "in_explicit_str_list",
        filters={"name": {"$in": ["test1", "test2"]}},
        cypher_query=in_comparison_cypher("name", "IN", "$fv_name"),
        params={"fv_name": ["test1", "test2"]},
    ),
    TestCase(
        "in_explicit_number_list",
        filters={"num": {"$in": [1, 2]}},
        cypher_query=in_comparison_cypher("num", "IN", "$fv_num"),
        params={"fv_num": [1, 2]},
    ),
    TestCase(
        "in_explicit_number_set",
        filters={"num": {"$in": {2, 1}}},
        cypher_query=in_comparison_cypher("num", "IN", "$fv_num"),
        params={"fv_num": [1, 2]},
    ),
    TestCase(
        "nin_str_list",
        filters={"name": {"$nin": ["test1", "test2"]}},
        cypher_query=in_comparison_cypher("name", "NOT IN", "$fv_name"),
        params={"fv_name": ["test1", "test2"]},
    ),
    TestCase(
        "nin_num_set",
        filters={"num": {"$nin": {5, 2}}},
        cypher_query=in_comparison_cypher("num", "NOT IN", "$fv_num"),
        params={"fv_num": [2, 5]},
    ),
    TestCase(
        "nin_num_tuple",
        filters={"num": {"$nin": (1, 2)}},
        cypher_query=in_comparison_cypher("num", "NOT IN", "$fv_num"),
        params={"fv_num": (1, 2)},
    ),
    TestCase("exists_true", filters={"field": {"$exists": True}}, cypher_query="doc.field IS NOT NULL"),
    TestCase("exists_false", filters={"field": {"$exists": False}}, cypher_query="doc.field IS NULL"),
    TestCase(
        "and_implicit",
        filters={"type": {"$eq": "article"}, "rating": {"$gte": 3}},
        cypher_query="(doc.type = $fv_type AND doc.rating >= $fv_rating)",
        params={"fv_type": "article", "fv_rating": 3},
    ),
    TestCase(
        "and_explicit",
        filters={"$and": {"type": {"$eq": "article"}, "rating": {"$gte": 3}}},
        cypher_query="(doc.type = $fv_type AND doc.rating >= $fv_rating)",
        params={"fv_type": "article", "fv_rating": 3},
    ),
    TestCase(
        "and_single_operand",
        filters={"$and": {"rating": 3}},
        cypher_query="doc.rating = $fv_rating",
        params={"fv_rating": 3},
    ),
    TestCase(
        "or_many_operands",
        filters={"$or": {"type": {"$eq": "article"}, "rating": {"$gte": 3}}},
        cypher_query="(doc.type = $fv_type OR doc.rating >= $fv_rating)",
        params={"fv_type": "article", "fv_rating": 3},
    ),
    TestCase(
        "or_single_operand",
        filters={"$or": {"type": {"$eq": "article"}}},
        cypher_query="doc.type = $fv_type",
        params={"fv_type": "article"},
    ),
    TestCase(
        "complex_query",
        filters={
            "$and": {
                "type": {"$eq": "article"},
                "year": {"$gte": 2015, "$lt": 2021},
                "rating": {"$gte": 3},
                "$or": {"genre": {"$in": ["economy", "politics"]}, "publisher": "nytimes"},
            }
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
            "$or": [
                {"$and": {"type": "news", "likes": {"$lt": 100}}},
                {"$and": {"type": "blog", "likes": {"$gte": 500}}},
            ]
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
        filters={"$xor": {"type": {"$eq": "article"}, "rating": {"$gte": 3}}},
        expectation=pytest.raises(Neo4jFilterParserError),
    ),
    TestCase(
        "unsupported_comparison_operator",
        filters={"rating": {"$mod": 3}},
        expectation=pytest.raises(Neo4jFilterParserError),
    ),
]


@pytest.mark.unit
@pytest.mark.parametrize("filters,expected_cypher,expected_params,expectation", [tc.as_param() for tc in TEST_CASES])
def test_filter_converter(filters, expected_cypher, expected_params, expectation):
    parser = FilterParser()
    converter = Neo4jFiltersConverter("doc")

    with expectation:
        filter_ast = parser.parse(filters)
        cypher_query, params = converter.convert(filter_ast)

        assert cypher_query == expected_cypher
        assert params == expected_params
