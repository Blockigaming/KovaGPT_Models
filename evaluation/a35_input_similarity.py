"""Conservative lexical, Python-shape and graph leakage checks.

These checks catch substitutions and duplicated implementations; they cannot
prove statistical generalization or detect every semantic paraphrase. Shared
short JSON values and necessary policy facts are not answer contamination.
"""

import ast
from difflib import SequenceMatcher
from itertools import permutations
import re


def prompt_skeleton(text):
    text = re.sub(r"\[[^\n]*?\]", " DATA ", text)
    text = re.sub(r"(['\"])(.*?)\1", " QUOTED ", text)
    text = re.sub(r"\b[A-Za-z]\d+\b|\b[A-Z]\b", " ID ", text)
    text = re.sub(r"-?\d+(?:\.\d+)?", " NUMBER ", text)
    return re.findall(r"\w+",text.casefold())


def lexical_similarity(left,right):
    return SequenceMatcher(None,prompt_skeleton(left),prompt_skeleton(right),
                           autojunk=False).ratio()


class CodeShape(ast.NodeTransformer):
    def __init__(self):
        self.names = {}; self.strings = {}

    def visit_Name(self,node):
        builtins = {"range","list","dict","set","tuple","sum","len","zip",
                    "enumerate","sorted","int","str","bool","all","any",
                    "next","iter","isinstance"}
        if node.id not in builtins:
            node.id = self.names.setdefault(node.id,"v"+str(len(self.names)))
        return node

    def visit_Constant(self,node):
        if type(node.value) in (int,float): node.value = 0
        elif type(node.value) is str:
            node.value = self.strings.setdefault(node.value,"s"+str(len(self.strings)))
        return node

    def visit_List(self,node):
        if node.elts and all(isinstance(x,(ast.Constant,ast.UnaryOp)) for x in node.elts):
            node.elts = [ast.Constant(value="literal_vector")]
            return node
        return self.generic_visit(node)

    def visit_FunctionDef(self,node):
        node.name = self.names.setdefault(node.name,"v"+str(len(self.names)))
        return self.generic_visit(node)

    def visit_arg(self,node):
        node.arg = self.names.setdefault(node.arg,"v"+str(len(self.names)))
        return node


def code_shape(source, *, function_only=False):
    tree = ast.parse(source)
    if function_only:
        tree.body = [node for node in tree.body if isinstance(node,ast.FunctionDef)]
    elif len(tree.body)==1 and isinstance(tree.body[0],ast.Assign):
        tree = ast.Expression(body=tree.body[0].value)
    return ast.dump(CodeShape().visit(tree),include_attributes=False)


def benchmark_python(prompt):
    match = re.search(r"(?:evaluate|Python 3:)\s*(.*?)(?:\.\s*Return|$)",prompt,re.S)
    if not match: return None
    source = match.group(1).strip()
    try:
        expression = ast.parse(source,mode="eval")
        return "result="+ast.unparse(expression.body)
    except SyntaxError:
        try: ast.parse(source); return source
        except SyntaxError: return None


def graph_shape(tasks):
    """Exact name-independent isomorphism fingerprint for small authored graphs."""
    names = list(tasks)
    if not 0 < len(names) <= 7 or any(not set(deps) <= set(names) for deps in tasks.values()):
        raise ValueError("unsupported graph for structural check")
    # The degree sequence rejects most comparisons without interpreting labels.
    signatures = []
    for order in permutations(names):
        signatures.append(tuple(int(dependency in tasks[node]) for node in order for dependency in order))
    return (len(names),min(signatures))


def benchmark_graph(prompt):
    initial = re.search(r"Tasks (.*?) have no prerequisites\.",prompt)
    if not initial: return None
    split = lambda text: [word.strip() for word in re.split(r",|\band\b",text) if word.strip()]
    tasks = {name:[] for name in split(initial.group(1))}
    for name,deps in re.findall(r"(\w+) requires ([^.]+)\.",prompt): tasks[name] = split(deps)
    return tasks or None


def audit(rows,reviews,benchmarks,approved=()):
    by_id = {r["id"]:r for r in reviews}
    issues = []
    exclusions = [(r["id"],r["prompt"]) for r in benchmarks]
    exclusions += [(r["id"],r["messages"][0]["content"]) for r in approved]
    benchmark_code = [(r["id"],benchmark_python(r["prompt"])) for r in benchmarks]
    benchmark_code = [(identifier,code_shape(code)) for identifier,code in benchmark_code if code]
    benchmark_graphs = [(r["id"],benchmark_graph(r["prompt"])) for r in benchmarks]
    benchmark_graphs = [(identifier,graph_shape(graph)) for identifier,graph in benchmark_graphs if graph]
    fingerprints = {}; function_fingerprints = {}; lexical_max = 0.0
    for row in rows:
        identifier = row["id"]; prompt = row["messages"][0]["content"]
        review = by_id[identifier]; reference = review["reference"]
        for other,other_prompt in exclusions:
            ratio = lexical_similarity(prompt,other_prompt); lexical_max=max(lexical_max,ratio)
            if ratio >= .84: issues.append({"kind":"benchmark_or_approved_prompt_shape","id":identifier,"other":other})
        source = reference["inputs"]["source"] if reference and reference["operation"]=="python_trace" else None
        if source:
            fingerprints[identifier] = code_shape(source)
            for other,shape in benchmark_code:
                if fingerprints[identifier]==shape: issues.append({"kind":"benchmark_python_substitution","id":identifier,"other":other})
        fixture = review["code_fixture"]
        if fixture: function_fingerprints[identifier] = code_shape(fixture,function_only=True)
        if reference and reference["operation"] == "dependency_schedule":
            shape = graph_shape(reference["inputs"]["tasks"])
            for other,other_shape in benchmark_graphs:
                if shape==other_shape: issues.append({"kind":"benchmark_graph_isomorphism","id":identifier,"other":other})
    training = [r for r in rows if r["split"]=="train"]
    validation = [r for r in rows if r["split"]=="validation"]
    split_lexical_max = 0.0
    for left in training:
        for right in validation:
            a,b=left["id"],right["id"]
            ratio=lexical_similarity(left["messages"][0]["content"],right["messages"][0]["content"])
            split_lexical_max=max(split_lexical_max,ratio)
            if ratio >= .84: issues.append({"kind":"split_prompt_shape","id":a,"other":b})
            for shapes,kind in ((fingerprints,"split_python_shape"),(function_fingerprints,"split_function_shape")):
                if a in shapes and b in shapes and shapes[a]==shapes[b]: issues.append({"kind":kind,"id":a,"other":b})
            answer=left["messages"][1]["content"]
            if len(answer)>40 and answer==right["messages"][1]["content"]:
                issues.append({"kind":"split_long_answer_copy","id":a,"other":b})
            if by_id[a]["group"]==by_id[b]["group"] and by_id[a]["structure"]==by_id[b]["structure"]:
                issues.append({"kind":"split_scenario_reuse","id":a,"other":b})
    return {"issues":issues,"maximum_benchmark_or_approved_lexical_similarity":round(lexical_max,4),
            "maximum_split_lexical_similarity":round(split_lexical_max,4),
            "checks":["normalized_prompt","literal_identifier_normalized_lexical",
                      "python_ast_shape","graph_isomorphism","implementation_shape",
                      "long_answer_copy","declared_scenario_independence"],
            "semantic_generalization_proved":False}
