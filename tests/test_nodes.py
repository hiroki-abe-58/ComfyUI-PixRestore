import json


def test_node_schemas(comfy_dir):
    from pixrestore_comfy import nodes, runtime

    ids = {}
    for cls in (nodes.PixRestoreLoader, nodes.PixRestoreRestore, nodes.PixRestoreUnload):
        schema = cls.define_schema()
        ids[schema.node_id] = schema
    assert set(ids) == {"PixRestoreLoader", "PixRestoreRestore", "PixRestoreUnload"}
    restore = ids["PixRestoreRestore"]
    names = [i.id for i in restore.inputs]
    assert names == ["model", "image", "seed", "preprocess"]
    pre = next(i for i in restore.inputs if i.id == "preprocess")
    assert pre.options == runtime.PREPROCESS_MODES and pre.default == runtime.PREPROCESS_EXACT
    assert [o.display_name for o in restore.outputs] == ["image", "report"]
    assert ids["PixRestoreUnload"].not_idempotent and ids["PixRestoreUnload"].is_output_node


def test_example_workflows_use_only_known_nodes():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "workflows" / "api"
    known = {"LoadImage", "SaveImage", "PreviewAny", "PixRestoreLoader", "PixRestoreRestore", "PixRestoreUnload"}
    files = sorted(root.glob("*.json"))
    assert files
    for f in files:
        wf = json.loads(f.read_text(encoding="utf-8"))
        assert {n["class_type"] for n in wf.values()} <= known, f.name
        restore = [n for n in wf.values() if n["class_type"] == "PixRestoreRestore"]
        assert restore and all(n["inputs"]["preprocess"].startswith("exact") or "center crop" in n["inputs"]["preprocess"] for n in restore)
