"""Restricted expression evaluation and native metrics for public SRSD tasks.

Equation canonicalization and node labels follow OMRON SINIC X's MIT-licensed
SRSD comparator at commit 7d00b45d56250717ac08a256dc9f79b836d61027.
License and original metric sources are retained under each task's source/.
The ordered tree-distance dynamic program below is locally implemented.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import sympy as sp
from sympy.utilities.misc import func_name

VARIABLES = ("x0", "x1", "x2")
FUNCTIONS = {"sin": (np.sin, sp.sin), "cos": (np.cos, sp.cos),
             "exp": (np.exp, sp.exp), "log": (np.log, sp.log),
             "sqrt": (np.sqrt, sp.sqrt)}
MAX_EXPRESSION_CHARS = 2000
MAX_EXPRESSION_NODES = 80
MAX_EXPRESSION_DEPTH = 16


def parse_expression(expression):
    """Accept arithmetic only; never evaluate candidate Python strings."""
    if type(expression) is not str or not 1 <= len(expression) <= MAX_EXPRESSION_CHARS:
        raise ValueError("expression must be a string of 1 to 2000 characters")
    root = ast.parse(expression, mode="eval").body
    count = 0

    def visit(node, depth):
        nonlocal count
        count += 1
        if count > MAX_EXPRESSION_NODES or depth > MAX_EXPRESSION_DEPTH:
            raise ValueError("Expression exceeds 80 nodes or depth 16")
        if isinstance(node, ast.Constant):
            value = node.value
            if type(value) not in (int, float) or abs(value) > 1e100:
                raise ValueError("Use finite numeric constants with magnitude <= 1e100")
            if not math.isfinite(value):
                raise ValueError("Nonfinite constants are invalid")
        elif isinstance(node, ast.Name):
            if node.id not in VARIABLES and node.id != "pi":
                raise ValueError("Unknown variable or constant")
        elif isinstance(node, ast.UnaryOp) and type(node.op) in (ast.UAdd, ast.USub):
            visit(node.operand, depth+1)
        elif isinstance(node, ast.BinOp) and type(node.op) in (
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow
        ):
            visit(node.left, depth+1)
            visit(node.right, depth+1)
            if isinstance(node.op, ast.Pow):
                exponent = node.right
                sign = 1
                if isinstance(exponent, ast.UnaryOp) and isinstance(exponent.op, ast.USub):
                    exponent, sign = exponent.operand, -1
                if not isinstance(exponent, ast.Constant) or type(exponent.value) not in (int, float):
                    raise ValueError("Powers require a numeric literal exponent")
                if not -8 <= sign*exponent.value <= 8:
                    raise ValueError("Exponent must lie in [-8,8]")
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
              and node.func.id in FUNCTIONS and len(node.args) == 1 and not node.keywords):
            visit(node.args[0], depth+1)
        else:
            raise ValueError("Only arithmetic, x0/x1/x2/pi and sin/cos/exp/log/sqrt are allowed")

    visit(root, 1)
    return root


def _interpret(node, variables, symbolic=False):
    if isinstance(node, ast.Constant):
        return sp.sympify(node.value) if symbolic else node.value
    if isinstance(node, ast.Name):
        if node.id == "pi":
            return sp.pi if symbolic else np.pi
        return variables[node.id]
    if isinstance(node, ast.UnaryOp):
        value = _interpret(node.operand, variables, symbolic)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.Call):
        function = FUNCTIONS[node.func.id][int(symbolic)]
        return function(_interpret(node.args[0], variables, symbolic))
    left = _interpret(node.left, variables, symbolic)
    right = _interpret(node.right, variables, symbolic)
    if isinstance(node.op, ast.Add):
        return left+right
    if isinstance(node.op, ast.Sub):
        return left-right
    if isinstance(node.op, ast.Mult):
        return left*right
    if isinstance(node.op, ast.Div):
        return left/right
    return left**right


def to_sympy(root):
    return _interpret(root, {name: sp.Symbol(name) for name in VARIABLES}, symbolic=True)


def canonicalize(expression):
    # Match upstream eq_comparator.load_eq_as_tree, after safe AST construction.
    expression = sp.sympify(str(expression))
    expression = expression.subs(sp.pi, sp.pi.evalf()).evalf().factor().simplify().subs(1.0, 1)
    return sp.sympify(str(expression))


def expression_tree(expression):
    if expression.is_number:
        label = "Const"
    elif isinstance(expression, sp.Symbol):
        label = str(expression)
    else:
        label = func_name(expression)
    return (label, tuple(expression_tree(arg) for arg in expression.args))


def _postorder(tree):
    labels, leftmost = [None], [0]

    def walk(node):
        child_roots = [walk(child) for child in node[1]]
        index = len(labels)
        labels.append(node[0])
        leftmost.append(leftmost[child_roots[0]] if child_roots else index)
        return index

    walk(tree)
    latest = {left: index for index, left in enumerate(leftmost) if index}
    return labels, leftmost, sorted(latest.values())


def ordered_tree_distance(first, second):
    """Exact Zhang-Shasha distance; insert/delete/unequal-label rename each cost 1."""
    labels_a, left_a, roots_a = _postorder(first)
    labels_b, left_b, roots_b = _postorder(second)
    na, nb = len(labels_a)-1, len(labels_b)-1
    if na > 200 or nb > 200:
        raise ValueError("Canonical expression tree exceeds 200 nodes")
    distance = np.zeros((na+1, nb+1), dtype=np.int64)
    for root_a in roots_a:
        for root_b in roots_b:
            start_a, start_b = left_a[root_a], left_b[root_b]
            rows, cols = root_a-start_a+2, root_b-start_b+2
            forest = np.zeros((rows, cols), dtype=np.int64)
            forest[:, 0] = np.arange(rows)
            forest[0, :] = np.arange(cols)
            for i in range(1, rows):
                node_a = start_a+i-1
                for j in range(1, cols):
                    node_b = start_b+j-1
                    deletion = forest[i-1, j]+1
                    insertion = forest[i, j-1]+1
                    if left_a[node_a] == start_a and left_b[node_b] == start_b:
                        rename = forest[i-1, j-1]+(labels_a[node_a] != labels_b[node_b])
                        forest[i, j] = min(deletion, insertion, rename)
                        distance[node_a, node_b] = forest[i, j]
                    else:
                        subtree = (forest[left_a[node_a]-start_a, left_b[node_b]-start_b]
                                   + distance[node_a, node_b])
                        forest[i, j] = min(deletion, insertion, subtree)
    return int(distance[na, nb])


def normalized_edit_distance(predicted, reference):
    first, second = expression_tree(canonicalize(predicted)), expression_tree(canonicalize(reference))
    nodes = len(_postorder(second)[0])-1
    distance = ordered_tree_distance(first, second)
    return min(distance, nodes)/nodes


def mean_squared_relative_error(prediction, target):
    if np.any(target == 0):
        raise ValueError("Native relative-error metric is undefined for exact zero targets")
    with np.errstate(all="raise"):
        value = float(np.mean(np.square((prediction-target)/target)))
    if not math.isfinite(value):
        raise ValueError("Nonfinite relative prediction error")
    return value


def solve(payload):
    """Trusted grading worker entrypoint, run in a second isolated process."""
    artifact = payload["artifact"]
    if type(artifact) is not dict or set(artifact) != {"expression"}:
        raise ValueError("Return exactly {'expression': '...'}")
    root = parse_expression(artifact["expression"])
    symbolic = to_sympy(root)
    if symbolic.is_number or not symbolic.free_symbols:
        raise ValueError("Pure constant models are excluded by native model selection")
    x, target = np.asarray(payload["x"], dtype=float), np.asarray(payload["y"], dtype=float)
    with np.errstate(all="raise"):
        prediction = np.asarray(_interpret(root, dict(zip(VARIABLES, x.T))))
        if np.iscomplexobj(prediction):
            raise ValueError("Complex predictions are invalid")
        prediction = prediction.astype(float)
    if prediction.shape != target.shape or not np.all(np.isfinite(prediction)):
        raise ValueError("Expression must give one finite real prediction per row")
    relative_error = mean_squared_relative_error(prediction, target)
    result = {"validity": 1.0, "combined_score": 1.0/(1.0+relative_error),
              "mean_squared_relative_error": relative_error, "n_observations": len(target),
              "split": payload["split"]}
    if payload["split"] == "test":
        reference = to_sympy(parse_expression(payload["reference_expression"]))
        result["normalized_edit_distance"] = normalized_edit_distance(symbolic, reference)
        result["canonical_expression"] = str(canonicalize(symbolic))
    return result


def load_split(task_dir, split):
    if split not in ("train", "val", "test"):
        raise ValueError("Unknown data split")
    task_dir = Path(task_dir)
    provenance = json.loads((task_dir/"data_provenance.json").read_text())
    filename = f"data/{split}.txt"
    entry = next(item for item in provenance["files"] if item["file"] == filename)
    raw = (task_dir/filename).read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise ValueError(f"Pinned {split} data checksum mismatch")
    data = np.loadtxt(task_dir/filename)
    expected_rows = 8000 if split == "train" else 1000
    if data.shape != (expected_rows, 4) or not np.all(np.isfinite(data)):
        raise ValueError("Invalid official data shape or values")
    return data[:, :3], data[:, 3]


def make_payload(task_dir, variable_descriptions):
    x, y = load_split(task_dir, "train")
    return {"x_train": x.tolist(), "y_train": y.tolist(),
            "variables": list(VARIABLES), "variable_descriptions": variable_descriptions}


def evaluate_task(program_path, task_dir, variable_descriptions, reference_expression, split="val"):
    from _public_optimization_runtime import run_candidate

    started = time.monotonic()
    try:
        artifact = run_candidate(program_path, make_payload(task_dir, variable_descriptions), timeout=30)
        x, y = load_split(task_dir, split)
        grading_payload = {"artifact": artifact, "x": x.tolist(), "y": y.tolist(), "split": split}
        if split == "test":
            grading_payload["reference_expression"] = reference_expression
        # Candidate never sees validation/test targets or the structural reference.
        result = run_candidate(__file__, grading_payload, timeout=30)
        result["evaluation_time"] = time.monotonic()-started
        return result
    except Exception as exc:
        return {"validity": 0.0, "combined_score": 0.0, "split": split,
                "error": f"{type(exc).__name__}: {exc}"[:1000],
                "evaluation_time": time.monotonic()-started}
