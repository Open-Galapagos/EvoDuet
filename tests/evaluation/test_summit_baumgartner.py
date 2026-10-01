"""Native-model parity and invalid-artifact tests for the Summit adapter."""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

TASK = Path(__file__).resolve().parents[2] / "benchmarks/summit_baumgartner"
spec = importlib.util.spec_from_file_location("summit_baumgartner_evaluator", TASK / "evaluator.py")
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)


def design(**updates):
    value = {"catalyst": "AlPhos", "base": "BTMG", "base_equivalents": 2.25,
             "temperature": 99.89, "t_res": 1763.66}
    value.update(updates)
    return value


def test_native_ann_and_sklearn_preprocessing_parity():
    """Execute the pinned upstream ANN class and sklearn transformations.

    This avoids installing Summit's obsolete training dependencies; the frozen
    inference path itself uses the released code, weights and scaler values.
    """
    torch = pytest.importorskip("torch")
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    import pandas as pd

    torch.set_num_threads(1)
    tree = ast.parse((TASK / "data/upstream_experimental_emulator.py").read_text())
    ann = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ANNRegressor")
    namespace = {"torch": torch, "F": torch.nn.functional}
    exec(compile(ast.Module(body=[ann], type_ignores=[]), "upstream_ANNRegressor", "exec"), namespace)
    config = json.loads((TASK / "data/model.json").read_text())
    domain = config["domain"]
    categorical = [v for v in domain if v["type"] == "CategoricalVariable"]
    numeric = [v for v in domain if v["type"] == "ContinuousVariable" and not v["is_objective"]]
    rng = np.random.default_rng(742)
    designs = [design()]
    for catalyst in domain[0]["levels"]:
        for base in domain[1]["levels"]:
            for _ in range(20):
                designs.append({"catalyst": catalyst, "base": base,
                                **{v["name"]: float(rng.uniform(*v["bounds"])) for v in numeric}})
    frame = pd.DataFrame(designs)
    predictions = []
    for index, parameter in enumerate(config["experiment_params"]["predictors"]):
        preprocessor = ColumnTransformer([
            ("num", StandardScaler(), [v["name"] for v in numeric]),
            ("cat", OneHotEncoder(categories=[v["levels"] for v in categorical]),
             [v["name"] for v in categorical]),
        ])
        preprocessor.fit(frame)
        scaler = preprocessor.named_transformers_["num"]
        for key, value in parameter["input_preprocessor"]["num"].items():
            setattr(scaler, key, np.asarray(value))
        x = torch.tensor(preprocessor.transform(frame)).float()
        model = namespace["ANNRegressor"](input_dim=10, output_dim=1)
        state = torch.load(TASK / f"data/baumgartner_aniline_cn_crosscoupling_predictor_{index}.pt",
                           map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            y = model(x).numpy()
        output_scaler = StandardScaler()
        for key, value in parameter["output_preprocessor"].items():
            setattr(output_scaler, key, np.asarray(value))
        predictions.append(output_scaler.inverse_transform(y)[:, 0])
    native = np.asarray(predictions).T
    adapted = np.asarray([evaluator.predict_members(value)[0] for value in designs])
    np.testing.assert_allclose(adapted, native, atol=2e-6, rtol=2e-6)
    actual_scores = np.asarray([evaluator.score_design(value)["combined_score"] for value in designs])
    np.testing.assert_allclose(actual_scores, np.clip(native, 0, 1).mean(axis=1), atol=2e-6, rtol=2e-6)


def test_clip_each_member_before_averaging():
    raw, clipped = evaluator.predict_members(design())
    assert np.any(raw > 1)
    assert evaluator.score_design(design())["combined_score"] == float(clipped.mean())
    assert abs(float(clipped.mean()) - float(np.clip(raw.mean(), 0, 1))) > 0.01


@pytest.mark.parametrize("value", [None, [], {"combined_score": 1},
    design(catalyst="unknown"), design(base=True), design(temperature=float("nan")),
    design(t_res=float("inf")), design(base_equivalents=True), design(t_res=59.9),
    design(temperature=100.0001), design(base_equivalents=2.5001), design(yld=1.0)])
def test_reject_invalid_conditions(value):
    with pytest.raises(ValueError):
        evaluator.score_design(value)


def test_bundled_asset_hashes():
    manifest = json.loads((TASK / "UPSTREAM_MANIFEST.json").read_text())
    for entry in [*manifest["files"], manifest["derived_weights"]]:
        assert hashlib.sha256((TASK / entry["retained_file"]).read_bytes()).hexdigest() == entry["sha256"]


def test_candidate_programs_and_failure(tmp_path):
    initial = evaluator.evaluate(TASK / "initial_program.py")
    reference = evaluator.evaluate(TASK / "oracle/best_program.py")
    assert initial["validity"] == reference["validity"] == 1
    assert 0 < initial["combined_score"] < reference["combined_score"] < 1
    bad = tmp_path / "bad.py"
    bad.write_text('def solve(payload):\n    raise RuntimeError("candidate failure")\n')
    assert evaluator.evaluate(bad)["validity"] == 0
    assert not any("model" in key or "yield" in key for key in evaluator.task_payload())
