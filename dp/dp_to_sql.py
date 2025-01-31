import codecs
from pathlib import Path
import re
from typing import Optional


def read_plans(plan_file: Path) -> dict[Path, str]:
    queries = list(Path("queries/job").glob("*.sql"))
    query_dict = {q.name[:-4] + ".txt": q for q in queries}
    plan_dict = {}
    res = {}
    with open(plan_file, "r") as plans:
        while True:
            query_name = plans.readline()
            plan = plans.readline()
            if not query_name:
                break
            plan_dict[query_name[:-1]] = plan[:-1]
    for n in sorted(query_dict):
        q = query_dict[n]
        p = plan_dict[n]
        res[q] = p
    return res


class Plan:
    def __init__(self, name: str, left=None, right=None):
        self.name: str = name
        self.left: Optional[Plan] = left
        self.right: Optional[Plan] = right

    def get_relation_names(self) -> list[str]:
        if self.name != "":
            return [self.name]
        else:
            return self.left.get_relation_names() + self.right.get_relation_names()

    def is_leaf(self) -> bool:
        if self.name != "":
            assert self.left is None and self.right is None
        else:
            assert self.left is not None and self.right is not None
        return self.name != ""

    def is_bottom_join(self) -> bool:
        return not self.is_leaf() and self.left.is_leaf() and self.right.is_leaf()


def find_matching_parenthesis(string):
    count = 0
    for i in range(0, len(string)):
        if string[i] == "(":
            count += 1
        elif string[i] == ")":
            count -= 1
        if count == 0:
            return i


def parse_plan(plan_str) -> Plan:
    if plan_str.startswith("("):
        end_w1 = find_matching_parenthesis(plan_str)
        w1 = plan_str[1:end_w1]
        assert plan_str[end_w1 : end_w1 + 3] == ")⋈("
        w2 = plan_str[end_w1 + 3 : -1]
        return Plan("", parse_plan(w1), parse_plan(w2))
    else:
        return Plan(plan_str)


class Relation:
    def __init__(self, name: str, alias: str):
        self.name = name
        self.alias = alias

    def __repr__(self):
        return f"Relation(name={self.name}, alias={self.alias})"

    def get_numbered_name(self) -> str:
        pattern = r"\d+$"
        match = re.search(pattern, self.alias)
        if match:
            if match.group(0) != "1":
                return f"{self.name}_{match.group(0)}"
        return self.name


class Condition:
    def __init__(self, condition: str, is_join: bool, relations: list[Relation]):
        self.condition = condition
        self.is_join = is_join
        self.relations = relations

    def __repr__(self):
        return f"Condition(condition={self.condition}, is_join={self.is_join}, relations={self.relations})"


def is_join_condition(condition_str, relations):
    if " or " not in condition_str.lower() and "=" in condition_str:
        left, right = condition_str.split("=", 1)
        left, right = f" {left.strip()}", f" {right.strip()}"
        left_has_relation = any(f" {rel.alias}." in left for rel in relations.values())
        right_has_relation = any(f" {rel.alias}." in right for rel in relations.values())
        return left_has_relation and right_has_relation
    return False


class SQLQuery:
    def __init__(self, query: str):
        query = query.strip().rstrip(";")
        self.query = query
        self.select_clause = ""
        self.relations: dict[str, Relation] = {}
        self.relations_name_dict: dict[str, Relation] = {}
        self.conditions: list[Condition] = []
        self.joins: list[Condition] = []
        self.parse_query()

    def parse_query(self):
        self.extract_select_clause()
        self.extract_relations()
        self.extract_conditions_and_joins()

    def extract_select_clause(self):
        select_pattern = re.compile(r"SELECT\s+(.*?)\s+FROM\s+", re.DOTALL | re.IGNORECASE)
        select_match = select_pattern.search(self.query)
        if select_match:
            self.select_clause = select_match.group(1).strip()

    def extract_relations(self):
        from_pattern = re.compile(r"FROM\s+(.*?)\s+WHERE", re.DOTALL | re.IGNORECASE)
        from_match = from_pattern.search(self.query)
        if from_match:
            from_clause = from_match.group(1).strip()
            relations_pattern = re.compile(r"(\w+)\s+AS\s+(\w+)", re.IGNORECASE)
            relations_matches = relations_pattern.findall(from_clause)
            self.relations = {alias: Relation(name, alias) for name, alias in relations_matches}
        self.relations_name_dict = {r.get_numbered_name(): r for r in self.relations.values()}

    def extract_conditions_and_joins(self):
        where_pattern = re.compile(r"WHERE\s+(.*)", re.DOTALL | re.IGNORECASE)
        where_match = where_pattern.search(self.query)
        if where_match:
            where_clause = where_match.group(1).strip()
            conditions_pattern = re.compile(r"( AND | and )?\s*([\w\.]+.*?)(?=( AND | and |$))", re.IGNORECASE)
            # conditions_pattern = re.compile(r"\bAND\s*([\w\.]+.*?)(?=(AND|$))', re.IGNORECASE)
            conditions_matches = conditions_pattern.findall(where_clause)
            conditions_to_skip = []
            for i, (_, condition_str, _) in enumerate(conditions_matches):
                if i in conditions_to_skip:
                    continue
                if condition_str.lower().startswith("and"):
                    condition_str = condition_str[3:].strip()
                if "between" in condition_str.lower():
                    _, next_condition, _ = conditions_matches[i + 1]
                    if next_condition.lower().startswith("and"):
                        next_condition = next_condition[3:].strip()
                    condition_str = f"{condition_str.strip()} AND {next_condition.strip()}"
                    conditions_to_skip.append(i + 1)
                if " or " in condition_str.lower() and condition_str.count("(") > condition_str.count(")"):
                    _, next_condition, _ = conditions_matches[i + 1]
                    if condition_str.count("(") == next_condition.count(")"):
                        condition_str = f"{condition_str.strip()} AND {next_condition.strip()}"
                        conditions_to_skip.append(i + 1)

                condition_str = f" {condition_str.strip()} "
                is_join = is_join_condition(condition_str, self.relations)
                involved_relations = [rel for alias, rel in self.relations.items() if f" {alias}." in condition_str]
                condition = Condition(condition_str.strip(), is_join, involved_relations)
                if is_join:
                    self.joins.append(condition)
                else:
                    self.conditions.append(condition)

    def get_select_attributes(self) -> list:
        attributes = []
        for attr in self.select_clause.split(","):
            # Removing parentheses and functions
            matches = re.findall(r"\([^)]*\)", attr)
            if len(matches) > 0:
                assert len(matches) == 1
                attr = matches[0][1:-1]
            clean_attr = attr.split(" AS ")[0].strip()
            if clean_attr:  # Only add non-empty attributes
                attributes.append(clean_attr)
        return attributes

    def get_join_attributes(self) -> list:
        attributes = []
        for j in self.joins:
            l, r = j.condition.split("=")
            attributes += [x.strip() for x in (l, r)]
        return list(set(attributes))

    def get_join(self, r1: Relation, r2: Relation) -> Condition:
        for j in self.joins:
            if r1 in j.relations and r2 in j.relations:
                return j

    def get_relation_conditions(self, r: Relation):
        return [c for c in self.conditions if r in c.relations]


def read_file(p: Path) -> str:
    with open(p, "r") as file:
        file_content = file.read()
    return file_content


def get_identity_function_prefix():
    return codecs.encode("hzoen", "rot_13")


class Subtree:
    def __init__(self, relations, query, name):
        self.contained_relations: list[Relation] = relations
        self.query: str = query
        self.name: str = name

    def get_wrapped_query(self) -> str:
        return f"{get_identity_function_prefix()}.identity(table({self.query}), 'trace') as {self.name}"

    def rename_relation_attr(self, relation: Relation, attr: str) -> str:
        rel_name, attr_name = attr.split(".")
        assert rel_name == relation.alias
        if relation in self.contained_relations:
            return f"{self.name}.{relation.alias}_{attr_name}"
        else:
            return attr

    def get_final_query(self, q: SQLQuery) -> str:
        translated_select_clause = q.select_clause.replace(".", "_")
        return f"SELECT {translated_select_clause}\nFROM ({self.query})"


def get_join_conditions(rels2: list[Relation], rels1: list[Relation], q: SQLQuery) -> list[Condition]:
    result = []
    for r1 in rels1:
        for r2 in rels2:
            join_cond = q.get_join(r1, r2)
            if join_cond is not None:
                result.append(join_cond)
    return result


def get_select_str(
    involved_relations: list[Relation],
    required_attrs: dict[Relation, list[str]],
    new_relations: list[Relation],
    subtrees: list[Subtree],
):
    outputs = []
    output_names = []
    for r in involved_relations:
        if r in new_relations:
            for attr in set(required_attrs[r]):
                outputs.append(f"{r.alias}.{attr}")
                output_names.append(f"{r.alias}_{attr}")
        else:
            for attr in set(required_attrs[r]):
                combined_attr = f"{r.alias}.{attr}"
                combined_identifier = rename_attr(r, combined_attr, subtrees, True)
                outputs.append(combined_identifier)
                output_names.append(combined_identifier.split(".")[1])

    select_str = ", ".join(f"{output} AS {name}" for output, name in zip(outputs, output_names))
    return select_str


def get_from_str(involved_relations: list[Relation]):
    return ", ".join(f"{r.name} AS {r.alias}" for r in involved_relations)


def rename_attr(rel: Relation, rel_attr: str, subtrees: list[Subtree], must_find: bool = False) -> str:
    found = False
    result = ""
    for subtree in subtrees:
        if rel in subtree.contained_relations:
            assert not found
            result = subtree.rename_relation_attr(rel, rel_attr)
            found = True
    if not found:
        assert not must_find
        result = rel_attr
    return result


def rename_conditions(conditions: list[Condition], subtrees: list[Subtree]) -> list[str]:
    result = []
    for condition in conditions:
        if condition.is_join:
            assert len(condition.relations) == 2
            l, r = condition.relations
            l_a, r_a = condition.condition.split("=")
            l_a, r_a = l_a.strip(), r_a.strip()
            if l.alias == r_a.split(".")[0]:
                l, r = r, l
            condition = f"{rename_attr(l, l_a, subtrees)} = {rename_attr(r, r_a,subtrees)}"
            result.append(condition)
        else:
            result.append(condition.condition)
    return result


class Counter:
    def __init__(self):
        self.val = 0

    def draw(self) -> int:
        res = self.val
        self.val += 1
        return res


def plan_to_sql(
    plan: Plan,
    q: SQLQuery,
    possible_ghost_relations: dict[Relation, list[Relation]],
    required_attrs: dict[Relation, list[str]],
    relation_lookup: dict[str, Relation],
    subtree_count: Counter,
) -> Subtree:
    name = f"subtree{subtree_count.draw()}"
    if plan is None:
        assert False
    if plan.is_bottom_join():
        left_rel = relation_lookup[plan.left.name]
        right_rel = relation_lookup[plan.right.name]

        join_condition = q.get_join(left_rel, right_rel)
        assert join_condition is not None
        conditions = [join_condition]

        ghost_relations = [g for r in (left_rel, right_rel) for g in possible_ghost_relations[r]]
        new_relations = [left_rel, right_rel] + ghost_relations
        conditions += get_join_conditions([left_rel, right_rel], ghost_relations, q)

        for r in new_relations:
            conditions += q.get_relation_conditions(r)

        involved_relations = new_relations
        select_str = get_select_str(involved_relations, required_attrs, new_relations, [])

        from_str = get_from_str(involved_relations)
        condition_str = "\n AND ".join(c.condition for c in conditions)
    elif (plan.left.is_leaf() or plan.right.is_leaf()) and not (plan.left.is_leaf() and plan.right.is_leaf()):
        leaf, sub_plan = plan.right, plan.left
        if plan.left.is_leaf():
            leaf, sub_plan = sub_plan, leaf
        assert leaf.is_leaf() and not sub_plan.is_leaf()
        sub_plan = plan_to_sql(sub_plan, q, possible_ghost_relations, required_attrs, relation_lookup, subtree_count)
        leaf_rel = relation_lookup[leaf.name]
        ghost_relations = [g for g in possible_ghost_relations[leaf_rel]]
        new_relations = [leaf_rel] + ghost_relations
        involved_relations = sub_plan.contained_relations + new_relations

        conditions = get_join_conditions([leaf_rel], ghost_relations, q) + get_join_conditions(
            new_relations, sub_plan.contained_relations, q
        )
        for r in new_relations:
            conditions += q.get_relation_conditions(r)
        conditions = rename_conditions(conditions, [sub_plan])

        select_str = get_select_str(involved_relations, required_attrs, new_relations, [sub_plan])
        from_str = f"{get_from_str(new_relations)}, {sub_plan.get_wrapped_query()}"
        condition_str = "\n AND ".join(c for c in conditions)
    else:
        left = plan_to_sql(plan.left, q, possible_ghost_relations, required_attrs, relation_lookup, subtree_count)
        right = plan_to_sql(plan.right, q, possible_ghost_relations, required_attrs, relation_lookup, subtree_count)
        involved_relations = left.contained_relations + right.contained_relations

        conditions = get_join_conditions(left.contained_relations, right.contained_relations, q)
        conditions = rename_conditions(conditions, [left, right])
        select_str = get_select_str(involved_relations, required_attrs, [], [left, right])
        from_str = f"{left.get_wrapped_query()}, {right.get_wrapped_query()}"
        condition_str = "\n AND ".join(c for c in conditions)

    if len(condition_str) == 0:
        condition_str = "1 = 1"
    query = f"SELECT {select_str}\nFROM {from_str}\nWHERE {condition_str}"
    return Subtree(involved_relations, query, name)


def gen_query(q_file, p_text) -> str:
    q_text = read_file(q_file)
    plan = parse_plan(p_text[1:-1])
    q = SQLQuery(q_text)
    used_relations = [q.relations_name_dict[n] for n in plan.get_relation_names()]
    relation_lookup = {n: q.relations_name_dict[n] for n in plan.get_relation_names()}
    ghost_relations = [r for r in q.relations.values() if r not in used_relations]

    # attach the ghost relations to each relation that can be joined with it
    possible_ghost_relations = {r: [] for r in used_relations}
    for r, gs in possible_ghost_relations.items():
        joins = [j for j in q.joins if r in j.relations]
        for j in joins:
            for r2 in j.relations:
                if r2 in ghost_relations:
                    gs.append(r2)

    # find all required attributes
    select_attributes = q.get_select_attributes()
    join_attributes = q.get_join_attributes()
    required_attributes = {}
    for a in select_attributes + join_attributes:
        rel, attr = a.split(".")
        rel = q.relations[rel]
        if rel not in required_attributes:
            required_attributes[rel] = []
        required_attributes[rel].append(attr)

    plan = plan_to_sql(plan, q, possible_ghost_relations, required_attributes, relation_lookup, Counter())
    return " ".join(plan.get_final_query(q).splitlines())


def store_strings_to_file(strings: list[str], file: Path):
    with open(file, "w") as out:
        out.write("\n".join(strings))


def convert_all_dp_results_to_sql():
    cout_plans = read_plans(Path("dp/cout_plans.txt"))
    model_plans = read_plans(Path("dp/model_plans.txt"))
    cout_sql = []
    model_sql = []
    query_names = []

    # q = Path("queries/job/7a.sql")
    # p_cout = cout_plans[q]
    # p_model = cout_plans[q]
    # model_q = gen_query(q, p_model)
    # print(model_q)
    # model_c = gen_query(q, p_cout)
    # print(model_c)

    for (q, p_cout), (q2, p_model) in zip(cout_plans.items(), model_plans.items()):
        assert q == q2
        cout_query = gen_query(q, p_cout)
        model_query = gen_query(q, p_model)
        cout_sql.append(cout_query)
        model_sql.append(model_query)
        query_names.append(q.name)
        # print(q)
    store_strings_to_file(cout_sql, Path("dp/cout_plans.sql"))
    store_strings_to_file(model_sql, Path("dp/model_plans.sql"))
    store_strings_to_file(query_names, Path("dp/query_names.txt"))


def main():
    convert_all_dp_results_to_sql()


if __name__ == "__main__":
    main()
