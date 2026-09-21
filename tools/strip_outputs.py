#!/usr/bin/env python3
"""git clean filter: strip outputs / execution counts from a notebook on stdin -> stdout.
Keeps the working copy (with figures) untouched; only the committed version is stripped."""
import json, sys
nb = json.load(sys.stdin)
for c in nb.get("cells", []):
    if c.get("cell_type") == "code":
        c["outputs"] = []; c["execution_count"] = None
    c.get("metadata", {}).pop("execution", None)
nb.get("metadata", {}).pop("widgets", None)
json.dump(nb, sys.stdout, indent=1, ensure_ascii=False); sys.stdout.write("\n")
