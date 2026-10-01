"""Vendored subset of the SimpleTES ``simpletes`` package.

The SimpleTES evaluators import ``simpletes.construction`` (the hook that snapshots
the evaluated construction when the engine asks for it through
``SIMPLETES_CAPTURE_CONSTRUCTION_PATH``); ``skydiscover_adapter.py`` puts this suite
root on ``PYTHONPATH`` exactly as SimpleTES puts its repo root, so the import and
the repo's ``sitecustomize.py`` behave as upstream.

``construction.py`` and ``utils/{__init__,text,log}.py`` are byte-identical to
upstream.  Upstream's ``simpletes/__init__.py`` imports the whole engine, so this
file replaces it; it is the only non-verbatim file in the package.
"""
