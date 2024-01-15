from collections import namedtuple
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from neo4j_haystack.errors import Neo4jFilterParserError

FilterType = Dict[str, Any]

FieldValuePrimitive = Union[int, float, str, bool]
FieldValueType = Union[FieldValuePrimitive, Iterable[FieldValuePrimitive]]

LogicalOpsTuple = namedtuple("LogicalOpsTuple", "OP_AND, OP_OR, OP_NOT")
LOGICAL_OPS = LogicalOpsTuple("AND", "OR", "NOT")

ComparisonOpsTuple = namedtuple(
    "ComparisonOpsTuple", "OP_EQ, OP_NEQ, OP_IN, OP_NIN, OP_GT, OP_GTE, OP_LT, OP_LTE, OP_EXISTS"
)
COMPARISON_OPS = ComparisonOpsTuple("==", "!=", "in", "not in", ">", ">=", "<", "<=", "exists")


class OpType(str, Enum):
    LOGICAL = "logical"
    COMPARISON = "comparison"
    UNKNOWN = "unknown"

    @staticmethod
    def from_op(op: Optional[str]) -> "OpType":
        if op in LOGICAL_OPS:
            return OpType.LOGICAL
        elif op in COMPARISON_OPS:
            return OpType.COMPARISON

        return OpType.UNKNOWN


class AST:
    """
    A base class for nodes of an abstract syntax tree for metadata filters. Its `descriptor` property provides a python
    friendly name (e.g. potentially used as a method name)
    """

    @property
    def descriptor(self) -> str:
        return type(self).__name__.lower()


class LogicalOp(AST):
    """
    `AST` node which represents a logical operator e.g. "AND", "OR", "NOT".
    Logical operator is comprised of an operator (e.g. "AND") and respective operands - expressions participating in
    the logical evaluation syntax. For example one could read it the following way: ``"operand1 AND operand2"``, where
    ``"AND"`` is the operator and ``"operand1", "operand2"`` are `AST` nodes which might evaluate into:

    * simple expressions like ``"field1 = 1 AND field2 >= 2"``
    * more complex expressions like ``"(field1 = 1 OR field2 < 4) AND field3 > 2"``

    Please notice the actual representation of expressions is managed by a separate component which knows how to parse
    the syntax tree and translate its nodes (`AST`) into DocumentStore's specific filtering syntax.
    """

    def __init__(self, operands: Sequence[AST], op: str):
        if OpType.from_op(op) != OpType.LOGICAL:
            raise Neo4jFilterParserError(
                f"The '{op}' logical operator is not supported. Consider using one of: {LOGICAL_OPS}."
            )

        self.operands = operands
        self.op = op

    @property
    def descriptor(self) -> str:
        return "logical_op"

    def __repr__(self):
        return f"<LogicalOp {self.op=}, {self.operands=}>"


class ComparisonOp(AST):
    """
    This `AST` node represents a comparison operator in filters syntax tree (e.g. "==", "in", "<=" etc).
    Comparison operator is comprised of an operator, a field name and field value. For example one could
    read it in the following way: ``"age == 20"``, where

    - ``"=="`` - is the operator
    - ``"age"`` - is a field name
    - "20" - is a comparison value.

    Please notice the actual representation of comparison expressions is managed by a separate component which knows how
    to translate the `ComparisonOp` into DocumentStore's specific filtering syntax.
    """

    def __init__(self, field_name: str, op: str, field_value: FieldValueType):
        if OpType.from_op(op) != OpType.COMPARISON:
            raise Neo4jFilterParserError(
                f"The '{op}' comparison operator is not supported. Consider using one of: {COMPARISON_OPS}."
            )

        self.field_name = field_name
        self.op = op
        self.field_value = field_value

    @property
    def descriptor(self) -> str:
        return "comparison_op"

    def __repr__(self):
        return f"<ComparisonOp {self.field_name=}, {self.op=}, {self.field_value=}>"


OperatorAST = Union[ComparisonOp, LogicalOp]


class FilterParser:
    """
    The implementation of metadata filter parser into an abstract syntax tree comprised of respective
    `AST` nodes. The tree structure has a single root node and, depending on actual filters provided, can result
    into a number of Logical nodes as well as Comparison (leaf) nodes. The parsing logic takes into consideration rules
    documented in the following document [Metadata Filtering](https://docs.haystack.deepset.ai/v2.0/docs/metadata-filtering).

    `FilterParser` does not depend on actual DocumentStore implementation. Its single purpose is to parse filters into a
    tree of `AST` nodes by applying metadata filtering rules (including default operators behavior).

    With a given example of a metadata filter parsing below:

    ```py
    filters = {
        "operator": "OR",
        "conditions": [
            {
                "operator": "AND",
                "conditions": [
                    {"field": "meta.type", "operator": "==", "value": "news"},
                    {"field": "meta.likes", "operator": "!=", "value": 100},
                ],
            },
            {
                "operator": "AND",
                "conditions": [
                    {"field": "meta.type", "operator": "==", "value": "blog"},
                    {"field": "meta.likes", "operator": ">=", "value": 500},
                ],
            },
        ],
    }
    op_tree = FilterParser().parse(filters)
    ```

    We should expect the following tree structure (`op_tree`) after parsing:

    ```console
                                            +-------------+
                                            | <LogicalOp> |
                                            |  op: "OR"   |
                                            +-------------+
                              +-------------+  operands   +----------------+
                              |             +-------------+                |
                              |                                            |
                        +-----+-------+                              +-----+-------+
                        | <LogicalOp> |                              | <LogicalOp> |
                        |  op: "AND"  |                              |  op: "AND"  |
                        +-------------+                              +-------------+
               +--------+  operands   +----+                 +-------+  operands   +-------+
               |        +-------------+    |                 |       +-------------+       |
               |                           |                 |                             |
    +----------+---------+  +--------------+------+   +------+-------------+   +-----------+---------+
    |   <ComparisonOp>   |  |   <ComparisonOp>    |   |   <ComparisonOp>   |   |   <ComparisonOp>    |
    |                    |  |                     |   |                    |   |                     |
    | field_name: "type" |  | field_name: "likes" |   | field_name: "type" |   | field_name: "likes" |
    | op: "=="           |  | op: ">="            |   | op: "=="           |   | op: "!="            |
    | field_value: "blog"|  | field_value: 500    |   | field_value: "news"|   | field_value: 100    |
    +--------------------+  +---------------------+   +--------------------+   +---------------------+
    ```

    Having such a tree DocumentStore should be able to traverse it and interpret into a Document Store specific syntax.
    """

    def __init__(self, flatten_field_name=True) -> None:
        self.flatten_field_name = flatten_field_name

    def comparison_op(self, field_name: str, op: str, field_value: FieldValueType) -> ComparisonOp:
        return ComparisonOp(field_name, op, field_value)

    def logical_op(self, op: str, operands: Sequence[AST]) -> LogicalOp:
        return LogicalOp(operands, op)

    def combine(self, *operands: Optional[OperatorAST], default_op: str = LOGICAL_OPS.OP_AND) -> Optional[OperatorAST]:
        """
        Combines several operands (standalone filter values) into a logical operator if number of operands is greater
        than one. Operands with `None` value are skipped (e.g. empty strings).

        Args:
            default_op: Default operator to be used to construct the `LogicalOp`, defaults to `OP_AND`

        Returns:
            Logical operator with `operands` or the operand itself if it is the only provided in arguments.
        """
        valid_operands = [op for op in operands if op]

        if len(valid_operands) == 0:
            return None

        return valid_operands[0] if len(valid_operands) == 1 else self.logical_op(default_op, valid_operands)

    def _parse_comparison_op(self, filters: FilterType) -> ComparisonOp:
        """
        Parsing a comparison operator dictionary.

        Args:
            filters: Comparison filter dictionary with `field`, `operator` and `value` keys expected.

        Raises:
            Neo4jFilterParserError: If required `field` or `value` comparison dictionary keys are missing.

        Returns:
           `AST` node representing the comparison expression in abstract syntax tree.
        """

        operator = filters["operator"]
        field_name = filters.get("field")
        filter_value = filters.get("value")

        if not field_name:
            raise Neo4jFilterParserError(f"`field` is mandatory in comparison filter dictionary: `{filters}`.")

        if filter_value is None:
            raise Neo4jFilterParserError(f"`value` is mandatory in comparison filter dictionary: `{filters}`.")

        return self.comparison_op(field_name, operator, filter_value)

    def _parse_logical_op(self, filters: FilterType) -> LogicalOp:
        """
        This method is responsible of parsing logical operators. It returns `LogicalOp` with specific operator
        (e.g. "OR") and its operands (list of `AST` nodes). Operands are parsed by calling `:::py self._parse_tree`
        which might result in further recursive parsing flow.

        Args:
            filters: Logical filter with conditions.

        Raises:
            Neo4jFilterParserError: If required `conditions` logic dictionary key is missing or is not a `list` with at
                least one item.

        Returns:
            The instance of `LogicalOp` with the list parsed operands.
        """

        op = filters["operator"]
        conditions = filters.get("conditions")

        if not conditions or not isinstance(conditions, list) or len(conditions) < 1:
            raise Neo4jFilterParserError("Can not parse logical operator with empty or absent conditions")

        operands: List[AST] = [self._parse_tree(condition) for condition in conditions]

        return self.logical_op(op, operands)

    def _parse_tree(self, filters: FilterType) -> OperatorAST:
        """
        This parses filters dictionary and identifies operations based on its "operator" type,
        e.g. "AND" would be resolved to a logical operator type (`OpType.LOGICAL`). Once recognized the parsing
        of the operator and its filter will be delegated to a respective method (e.g. `self._parse_logical_op` or
        `self._parse_comparison_op`).

        Args:
            filters: Metadata filters dictionary. Could be the full filter value or a smaller filters chunk.

        Raises:
            Neo4jFilterParserError: If required `operator` dictionary key is missing.
            Neo4jFilterParserError: If `filters` is not a dictionary.
            Neo4jFilterParserError: If operator value is unknown.

        Returns:
            A root `AST` node of parsed filter.
        """

        if not isinstance(filters, dict):
            raise Neo4jFilterParserError("Filter must be a dictionary.")

        if "operator" not in filters:
            raise Neo4jFilterParserError("`operator` must ne present in both comparison and logic dictionaries.")

        operator = filters["operator"]
        op_type = OpType.from_op(operator)

        if op_type == OpType.LOGICAL:
            return self._parse_logical_op(filters)
        elif op_type == OpType.COMPARISON:
            return self._parse_comparison_op(filters)
        else:
            raise Neo4jFilterParserError(
                f"Unknown operator({operator}) in filter dictionary. Should be either "
                f"comparison({COMPARISON_OPS}) or logical({LOGICAL_OPS})"
            )

    def parse(self, filters: FilterType) -> OperatorAST:
        """
        This is the entry point to parse a given metadata filter into an abstract syntax tree. The implementation
        delegates the parsing logic to the private `self._parse_tree` method.

        Args:
            filters: Metadata filters to be parsed.

        Raises:
            Neo4jFilterParserError: In case parsing results in empty `AST` tree or more than one root node has been
                created after parsing.

        Returns:
            Abstract syntax tree representing `filters`. You should expect a single root operator returned, which
            could used to traverse the whole tree.
        """

        return self._parse_tree(filters)
