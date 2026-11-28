"""Compare two directories written by TestDumpUpstreamRequests (internal/proxy).

Usage: python compare_upstream_dumps.py <dir A> <dir B> [verbose]

For every recorded scenario: the status returned to the client, the number of
upstream calls, their method/path/query/headers must be equal, and each
upstream body must be either the same bytes or the same JSON value. Exits 1
on any difference.
"""
import json, os, sys, collections
a_dir, b_dir = sys.argv[1], sys.argv[2]
verbose = len(sys.argv) > 3
names = sorted(set(os.listdir(a_dir)) | set(os.listdir(b_dir)))
stats = collections.Counter(); per_route = collections.defaultdict(collections.Counter); problems = []

def first_diff(x, y, path="$"):
    if type(x) != type(y): return f"{path}: type {type(x).__name__} vs {type(y).__name__}"
    if isinstance(x, dict):
        for k in sorted(set(x) | set(y)):
            if k not in x: return f"{path}.{k}: only in B"
            if k not in y: return f"{path}.{k}: only in A"
            d = first_diff(x[k], y[k], f"{path}.{k}")
            if d: return d
        return None
    if isinstance(x, list):
        if len(x) != len(y): return f"{path}: len {len(x)} vs {len(y)}"
        for i, (p, q) in enumerate(zip(x, y)):
            d = first_diff(p, q, f"{path}[{i}]")
            if d: return d
        return None
    return None if x == y else f"{path}: {str(x)[:80]!r} vs {str(y)[:80]!r}"

for n in names:
    route = n.split("__")[0] + " <- " + n.split("__")[1].split("_")[0]
    pa, pb = os.path.join(a_dir, n), os.path.join(b_dir, n)
    if not (os.path.exists(pa) and os.path.exists(pb)):
        stats["missing"] += 1; problems.append((n, "missing on one side")); continue
    A, B = json.load(open(pa, encoding="utf-8")), json.load(open(pb, encoding="utf-8"))
    if A["client_status"] != B["client_status"]:
        stats["DIFFERENT"] += 1; problems.append((n, f"client status {A['client_status']} vs {B['client_status']}")); continue
    ua, ub = A["upstream"] or [], B["upstream"] or []
    if len(ua) != len(ub):
        stats["DIFFERENT"] += 1; problems.append((n, f"upstream calls {len(ua)} vs {len(ub)}")); continue
    if not ua:
        stats["no upstream call (same status)"] += 1; per_route[route]["no upstream call"] += 1; continue
    verdict = "identical bytes"
    for x, y in zip(ua, ub):
        meta_x = {k: v for k, v in x.items() if k != "body"}; meta_y = {k: v for k, v in y.items() if k != "body"}
        if meta_x != meta_y:
            verdict = "DIFFERENT"; problems.append((n, "meta: " + str(first_diff(meta_x, meta_y)))); break
        if x["body"] == y["body"]: continue
        try:
            jx, jy = json.loads(x["body"]), json.loads(y["body"])
        except Exception as e:
            verdict = "DIFFERENT"; problems.append((n, f"body not JSON and bytes differ: {e}")); break
        d = first_diff(jx, jy)
        if d:
            verdict = "DIFFERENT"; problems.append((n, d)); break
        verdict = "same JSON, different bytes"
    stats[verdict] += 1; per_route[route][verdict] += 1

print("TOTAL", dict(stats))
for route in sorted(per_route): print(f"  {route:28s} {dict(per_route[route])}")
for n, why in problems[: (200 if verbose else 25)]: print("  !!", n, "->", why[:300])
sys.exit(1 if stats["DIFFERENT"] or stats["missing"] else 0)
