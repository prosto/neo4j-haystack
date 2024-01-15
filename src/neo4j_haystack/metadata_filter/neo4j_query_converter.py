from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Iterable, Optional, Tuple, Union

from neo4j_haystack.metadata_filter.parser import (
    AST,
    COMPARISON_OPS,
    LOGICAL_OPS,
    ComparisonOp,
    LogicalOp,
)

OP_AND, OP_OR, OP_NOT = LOGICAL_OPS
OP_EQ, OP_NEQ, OP_IN, OP_NIN, OP_GT, OP_GTE, OP_LT, OP_LTE, OP_EXISTS = COMPARISON_OPS

FieldValuePrimitive = Union[int, float, str, bool]
FieldValueType = Union[FieldValuePrimitive, Iterable[FieldValuePrimitive]]


class NodeVisitor:
    """
    A base class for a converter which is responsible for converting filters abstract syntax tree into a
    particular DocumentStore specific syntax (e.g. SQL/Cypher queries). It provides a basic structure for a
    "Visitor Pattern" where each node of the tree could be "visited" (a method called with the node as a parameter).
    """

    PREFIX = "visit"

    def visit(self, node: AST) -> Any:
        """
        Resolves a method name (visitor) to be called with a particular AST `node`. The visitor is responsible for
        handling logic of the node (e.g. converting it to a database specific query). The method to be called is
        determined by a descriptor (python friendly name, e.g. ``logical_op`` for a node of type `LogicalOp`).

        If visitor method could not be found a generic exception is raised.

        Args:
            node: `AST` node to be visited.

        Returns:
            Any result implementing class is planning to return.
        """
        visitor = getattr(self, self._descriptor(node), self.generic_visit)
        return visitor(node)

    def generic_visit(self, node: AST):
        """
        A fallback visitor in case no specific ones have been found. Raises error by default.
        """
        raise Exception(f"No {self._descriptor(node)} method")

    def _descriptor(self, node: AST) -> str:
        """
        Composes a full name for the visitor method with the prefix, e.g. ``"visit_"`` + ``"logical_op"``.
        """
        return f"{self.PREFIX}_{node.descriptor}"


@dataclass
class CypherFieldParam:
    field_name: str
    field_param_name: str
    field_param_ref: str
    field_value: FieldValueType
    op: Optional[str]


@dataclass
class CypherQueryExpression:
    query: str
    params: Dict[str, Any] = field(default_factory=dict)


class Neo4jQueryConverter(NodeVisitor):
    """
    This class acts as a visitor for all nodes in an abstract syntax tree built by the `FilterParser`. Its job is
    to traverse the tree and "visit" (call respective method) nodes to accomplish the conversion between metadata
    filters into Neo4j Cypher expressions. Resulting Cypher query is then used as part of the ``WHERE`` Neo4j clause and
    is based on the following concepts:

    - [WHERE clause](https://neo4j.com/docs/cypher-manual/current/clauses/where/)
    - [Comparison operators](https://neo4j.com/docs/cypher-manual/current/syntax/operators/#query-operators-comparison)
    - [Working with null](https://neo4j.com/docs/cypher-manual/current/values-and-types/working-with-null/)

    below is an example usage of the converter:

    ```python
    parser = FilterParser()
    converter = Neo4jQueryConverter("doc")

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

    filter_ast = parser.parse(filters)
    cypher_query, params = converter.convert(filter_ast)
    ```

    The above code will produce the following Neo4j Cypher query (``cypher_query``):

    ```cypher
    ((doc.type = $fv_type AND doc.likes < $fv_likes) OR (doc.type = $fv_type_1 AND doc.likes >= $fv_likes_1))
    ```

    with parameters (``params``):

    ```python
    {"fv_type": "news", "fv_likes": 100, "fv_type_1": "blog", "fv_likes_1": 500}
    ```

    The reason Cypher query is accompanied with parameters is because **we delegate data type conversion of parameter
    values to Neo4j Python Driver instead of repeating the logic in this class**. See the full mapping of core and
    extended types in the [Data Types](https://neo4j.com/docs/api/python-driver/current/api.html#data-types) document.

    The conversion logic of this class starts with root node of the abstract syntax tree which is usually a logical
    operator (e.g. "$and") and calls `visit` method with that node. Depending on type of the node respective visitor
    method is called:

    - `visit_logical_op` if node represents a logical operator (`LogicalOp`)
    - `visit_comparison_op` if node represents a comparison operator (`LogicalOp`)

    Each visitor is responsible of producing a Cypher expression which is then combined with other expressions up in
    the tree. Parentheses are used to group expressions if required.
    """

    COMPARISON_MAPPING: ClassVar[Dict[str, str]] = {
        OP_EQ: "=",
        OP_NEQ: "<>",
        OP_GT: ">",
        OP_GTE: ">=",
        OP_LT: "<",
        OP_LTE: "<=",
    }

    LOGICAL_MAPPING: ClassVar[Dict[str, str]] = {
        OP_AND: "AND",
        OP_OR: "OR",
        OP_NOT: "NOT",
    }

    EMPTY_EXPR = ""

    def __init__(
        self,
        field_name_prefix: Optional[str] = None,
        include_null_values_when_not_equal: Optional[bool] = False,
        flatten_field_name: Optional[bool] = True,
    ):
        """
        Constructs a new `Neo4jQueryConverter` instance.

        Args:
            field_name_prefix: A prefix to be added to field names in Cypher queries (e.g. `:::cypher doc.age = 20`,
                where ``"doc"`` is the prefix).
            include_null_values_when_not_equal: When `True` will enable additional Cypher expressions for
                inequality operators "not in" and "!=" so that `null` values are considered as "not equal" instead of
                being skipped. **This is experimental and by default is disabled**.
            flatten_field_name: In case filed names are composite/nested like "meta.age" replace dot (".") with
                underscores ("_").
        """
        self._field_name_prefix = field_name_prefix
        self._include_null_values_when_not_equal = include_null_values_when_not_equal
        self._flatten_field_name = flatten_field_name
        self._params: Dict[str, Any] = {}

    def convert(self, op_tree: AST) -> Tuple[str, Dict[str, Any]]:
        """
        The method to be called for converting metadata filter AST into Cypher expression. It starts with calling
        `self.visit` for the root node of the tree.

        Examples:
            >>> op_tree = FilterParser().parse({ "age", 30 })
            >>> Neo4jQueryConverter("n").convert(op_tree)
            ('n.age = $fv_age', {'$fv_age': 30})

        Args:
            op_tree: The abstract syntax tree representing a parsed metadata filter. See `FilterParser` to learn
                about parsing logic.

        Returns:
            Cypher expression along with parameters used in the expression.
        """
        self._params = {}

        cypher_query = self.visit(op_tree)

        return cypher_query, self._params

    def visit_logical_op(self, node: LogicalOp) -> str:
        """
        Handles logical operators of type `LogicalOp` by visiting all its operands first. This might result in
        recursive calls as operands might be logical operators as well. Once all operands render its own Cypher
        expressions all of those expressions are combined in a single query using operator of the `node` (e.g. "AND").

        If there are more than one operand parentheses are used to group all expressions.

        Examples:
            >>> operand1 = ComparisonOp("age", "!=", 20)
            >>> operand2 = ComparisonOp("age", "<=", 30)
            >>> logical_op = LogicalOp([operand1, operand2], "OR")
            >>> self.visit_logical_op(operator)
            "(doc.age <> 20 OR doc.age <= 30"

        Args:
            node: The logical operator `AST` node to be converted.

        Returns:
            Cypher expression converted from the logical operator.
        """
        operands = [self.visit(operand) for operand in node.operands]

        if node.op == OP_NOT:
            return self._wrap_in_parentheses(*operands, join_operator="AND", prefix_expr="NOT ")

        return self._wrap_in_parentheses(*operands, join_operator=self.LOGICAL_MAPPING[node.op])

    def visit_comparison_op(self, node: ComparisonOp) -> str:
        """
        Handles comparison operators of type `ComparisonOp` by checking each operator type and delegating to
        appropriate method for translating the operator to a Cypher comparison expression. Each comparison expression
        might produce respective query parameters (and its values) which are added to the common parameters dictionary.
        Parameter names are unique in order to avoid clashes between potentially multiple comparison operators.

        Examples:
            >>> operator = ComparisonOp("age", "!=", 20)
            >>> self.visit_comparison_op(operator)
            "doc.age <> 20"

            >>> operator = ComparisonOp("age", "in", [10, 11])
            >>> self.visit_comparison_op(operator)
            "doc.age IN [10, 11]"

        Args:
            node: The comparison operator `AST` node to be converted.

        Returns:
            Cypher expression converted from a comparison operator.
        """
        field_param = self._field_param(node)

        # default comparison expression syntax (e.g. "field_name >= field_value")
        cypher_expr = self._op_default(field_param)

        if node.op == OP_EXISTS:
            cypher_expr = self._op_exists(field_param)
        elif node.op == OP_NEQ:
            cypher_expr = self._op_neq(field_param)
        elif node.op == OP_IN:
            cypher_expr = self._op_in(field_param)
        elif node.op == OP_NIN:
            cypher_expr = self._op_nin(field_param)

        self._update_query_parameters(cypher_expr.params)

        return cypher_expr.query

    def _field_is_null_expr(self, param: CypherFieldParam) -> str:
        """
        Constructs "IS NULL" Cypher operator for the provided field (e.g. `:::cypher doc.age IS NULL`).
        Non-empty expression will be returned only when `self._include_null_values_when_not_equal` is set to
        `True`.

        Note:
            This is experimental feature and is disabled as of now

        Additional "IS NULL" checks can be useful for non-equality checks (e.g. ``"doc.age <> 0"``) as by default Neo4j
        skips properties with null values, see [Operators Equality](https://neo4j.com/docs/cypher-manual/current/syntax/operators/#_equality)
        manual. The `:::cypher doc.age <> 0 OR doc.age IS NULL` expression will make sure all nodes are included in
        comparison.

        Args:
            param: Field parameter metadata to be use in he Cypher expression.

        Returns:
            Cypher "IS NULL" query clause for a given `param.field_name`. Empty `str` if logic is disabled by
                `self._include_null_values_when_not_equal`.
        """
        return f"{param.field_name} is NULL" if self._include_null_values_when_not_equal else self.EMPTY_EXPR

    def _op_exists(self, param: CypherFieldParam) -> CypherQueryExpression:
        """
        Translates ``"exists"`` metadata filter into ``"IS NULL / IS NOT NULL"`` Cypher expression. Useful for checking
        absent properties. See more details in Neo4j documentation:

        - [Filter on `null`](https://neo4j.com/docs/cypher-manual/current/clauses/where/#filter-on-null)
        - [Property existence checking](https://neo4j.com/docs/cypher-manual/current/clauses/where/#property-existence-checking)

        An example metadata filter would look as follows `:::py { "age": { "$exists": True } }`, which translates into
        `::cypher doc.age IS NOT NULL` Cypher expression. With `False` in the filter value expression would become
        `:::cypher doc.age IS NULL`.

        Args:
            param: Field parameter metadata to be use in he Cypher expression.

        Returns:
            Cypher expression to check if property is ``null`` (property existence in Neo4j)
        """
        return self._cypher_expression(f"{param.field_name} {'IS NOT' if param.field_value else 'IS'} NULL")

    def _op_neq(self, param: CypherFieldParam) -> CypherQueryExpression:
        """
        Translates ``"!="`` (not equal) metadata filter into ``"<>"`` Cypher expression.

        Args:
            param: Field parameter metadata to be use in he Cypher expression.

        Returns:
            Cypher expression using Cypher inequality operator, e.g. `:::cypher doc.age <> 18`.
        """
        return self._cypher_expression(
            self._wrap_in_parentheses(
                f"{param.field_name} {param.op} {param.field_param_ref}",
                self._field_is_null_expr(param),
                join_operator="OR",
            ),
            param,
        )

    def _op_in(self, param: CypherFieldParam) -> CypherQueryExpression:
        """
        Translates ``"in"`` (element exists in a list) metadata filter into ``"IN"`` Cypher expression.
        See more details in Neo4j documentation:

        - [IN operator](https://neo4j.com/docs/cypher-manual/current/clauses/where/#where-in-operator)
        - [Conditional expressions](https://neo4j.com/docs/cypher-manual/current/queries/case/)
        - [Function ``any()``](https://neo4j.com/docs/cypher-manual/current/functions/predicate/#functions-any)

        Please notice a combination of "CASE" expression and "IN" operator are being used to comply with
        all metadata filtering options. In simple cases we would expect the following Cypher expression to be built:
        `:::cypher "doc.age in [20, 30]"`,however, if the ``"age"`` property is a list the expression
        would not work in Neo4j. Thus ``CASE`` checks the type of the property and if its a list ``any()`` function
        evaluates every element in the list with "IN" operator.

        Args:
            param: Field parameter metadata to be use in he Cypher expression.

        Returns:
            Cypher expression using Cypher ``IN`` operator.
        """
        return self._cypher_expression(
            f"CASE valueType({param.field_name}) STARTS WITH 'LIST' "
            f"WHEN true THEN any(val IN {param.field_name} WHERE val IN {param.field_param_ref}) "
            f"ELSE {param.field_name} IN {param.field_param_ref} END",
            param,
        )

    def _op_nin(self, param: CypherFieldParam) -> CypherQueryExpression:
        """
        Translates ``"not in"`` (element **not** in a list) metadata filter into ``"NOT..IN"`` Cypher expression. See
        documentation of the [_op_in][neo4j_haystack.metadata_filter.neo4j_query_converter.Neo4jQueryConverter._op_in]
        method for more details.

        Additional "IS NULL" expression will be added if such configuration is enabled. See implementation of
            [_field_is_null_expr][neo4j_haystack.metadata_filter.neo4j_query_converter.Neo4jQueryConverter._field_is_null_expr].

        Args:
            param: Field parameter metadata to be use in he Cypher expression.

        Returns:
            Cypher expression using Cypher ``NOT IN`` operator, e.g. ``"NOT doc.age IN [20, 30]"``.
        """
        return self._cypher_expression(
            self._wrap_in_parentheses(
                (
                    f"CASE valueType({param.field_name}) STARTS WITH 'LIST' "
                    f"WHEN true THEN any(val IN {param.field_name} WHERE NOT val IN {param.field_param_ref}) "
                    f"ELSE NOT {param.field_name} IN {param.field_param_ref} END"
                ),
                self._field_is_null_expr(param),
                join_operator="OR",
            ),
            param,
        )

    def _op_default(self, param: CypherFieldParam) -> CypherQueryExpression:
        """
        Default method to translate comparison metadata filter into Cypher expression.
        The mapping between metadata filter operators and Neo4j operators is stored in `self.COMPARISON_MAPPING`.

        Args:
            param: Field parameter metadata to be use in he Cypher expression.

        Returns:
            Cypher expression using Cypher operator, e.g. `:::cypher doc.age > 18`
        """
        return self._cypher_expression(f"{param.field_name} {param.op} {param.field_param_ref}", param)

    def _cypher_expression(self, query: str, field_param: Optional[CypherFieldParam] = None) -> CypherQueryExpression:
        """
        A factory method to create `CypherQueryExpression` data structure comprised of a Cypher query and
        respective query parameters.

        Args:
            query: Cypher query (expression)
            field_param: Optional parameters to be added to the query execution.

        Returns:
            Data class with query and parameters if any.
        """
        params = {field_param.field_param_name: field_param.field_value} if field_param else {}
        return CypherQueryExpression(query, params)

    def _wrap_in_parentheses(
        self, *cypher_expressions: str, join_operator: str = "AND", prefix_expr: Optional[str] = None
    ) -> str:
        """
        Wraps given list of Cypher expressions in parentheses and combines them with a given operator which is
        "AND" by default. For example if `:::py cypher_expressions=("age > $fv_age", "height <> $fv_height")`
        the method will return `:::cypher ("age > $fv_age AND height <> $fv_height")`.

        For a single Cypher expression no parentheses are added and no operator is used.

        Args:
            join_operator: Logical operator to combine given expressions.
            prefix_expr: Additional operators to be added in-front of the final Cypher query. Could be ``"NOT "`` in
                order to add negation to the combined expressions.

        Returns:
            Cypher query expression, wrapped in parentheses if needed.
        """
        valid_cypher_expressions = [expr for expr in cypher_expressions if expr]

        # we expect at least one expression to be provided
        result = valid_cypher_expressions[0]

        if len(valid_cypher_expressions) > 1:
            logical_expression = f" {join_operator} ".join(valid_cypher_expressions)
            result = f"({logical_expression})"

        return f"{prefix_expr}{result}" if prefix_expr else result

    def _field_param(self, node: ComparisonOp) -> CypherFieldParam:
        """
        Constructs `CypherFieldParam` data class with aggregated information about comparison operation.
        Below is an example with resolved attribute values:

        ```python
        CypherFieldParam(
            field_name="doc.age",
            field_param_name="fv_age",
            field_param_ref="$fv_age", # to be referenced in Cypher query
            field_value=10,
            op="<>", # mapped from "$ne" to "<>"
        )
        ```

        In a simplest case information from `CypherFieldParam` could be converted into `:::cypher doc.age <> $fv_age`,
        ``params={"fv_age": 10}``

        Args:
            node: Comparison `AST` node.

        Returns:
            CypherFieldParam: data class with required field parameter metadata.
        """

        field_name = node.field_name
        if "." in field_name and self._flatten_field_name:
            field_name = field_name.replace(".", "_")

        field_param_name = self._generate_param_name(field_name)
        field_value = self._normalize_field_type(node.field_value)

        return CypherFieldParam(
            field_name=f"{self._field_name_prefix}.{field_name}" if self._field_name_prefix else field_name,
            field_param_name=field_param_name,
            field_param_ref=f"${field_param_name}",
            field_value=field_value,
            op=self.COMPARISON_MAPPING.get(node.op),
        )

    def _normalize_field_type(self, field_value: FieldValueType) -> FieldValueType:
        """
        Adjusts field value type if need. Generally we delegate the type conversion of python field values (data types)
        to `neo4j` python driver. This way we reduce amount of logic in the converter and rely on the driver to make
        sure field value types are properly mapped between Python and Neo4j. In some cases though we can handle
        additional conversions which are not supported by the driver. See example below:

        The following Metadata filter query `:::py { "age": {30, 20 ,40} }` should produce `IN` Cypher clause, e.g.
        `:::cypher doc.age IN [20, 30, 40]`. However neo4j python driver does not accept `set` python type, thus we
        convert it to `list` to make such filters possible.

        Args:
            field_value: Field value to be adjusted to be compatible with Neo4j types.

        Returns:
            Adjusted field value type if required.
        """
        if isinstance(field_value, set):
            return sorted(field_value)

        return field_value

    def _update_query_parameters(self, params: Dict[str, Any]):
        """
        Updates Cypher query parameters with a given parameter set in `params`.

        Args:
            params: Parameters to be added to the final set of parameters which will be returned along with
                the generated Cypher query.
        """
        self._params.update(params)

    def _generate_param_name(self, field_name: str) -> str:
        """
        Generates a new Cypher query parameter name ensuring it is unique in a given parameter dictionary
        `self._params`. For example if `self._params` is equal to ``{ "fv_age": 20 }`` a new parameter name called
        ``fv_age_1`` will be generated by adding appropriate incremented index.

        Args:
            field_name: The nme of the filed to be generated. ``"fv_"`` prefix is added to all fields to avoid
                collisions with parameters given during Cypher query execution.

        Returns:
            A new parameter name, with an unique index if needed.
        """
        i = 0
        field_param_name = f"fv_{field_name}"
        while field_param_name in self._params:
            i += 1
            field_param_name = f"fv_{field_name}_{i}"
        return field_param_name
