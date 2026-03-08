import os
import json
import math
from typing import Dict, List, Tuple

import numpy as np

BASE_DIR = os.path.dirname(__file__)
TEMP_DIR = os.path.join(BASE_DIR, "temp")
MODEL_PATH = os.path.join(TEMP_DIR, "model.json")
RESULTS_PATH = os.path.join(TEMP_DIR, "results.json")


def ensure_temp_dir():
    os.makedirs(TEMP_DIR, exist_ok=True)


def read_model():
    with open(MODEL_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def assemble_system(model: Dict):
    nodes = model["nodes"]
    elements = model["elements"]
    supports_list = model.get("supports", [])
    loads_list = model.get("loads", [])

    node_ids = [n["id"] for n in nodes]
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    n_dof = 2 * len(nodes)

    K = np.zeros((n_dof, n_dof), dtype=float)
    R = np.zeros((n_dof,), dtype=float)

    def dof_index(nid: int) -> Tuple[int, int]:
        i = id_to_idx[nid]
        return 2 * i, 2 * i + 1

    for e in elements:
        n1 = next(n for n in nodes if n["id"] == e["n1"])
        n2 = next(n for n in nodes if n["id"] == e["n2"])
        x1, y1 = n1["x"], n1["y"]
        x2, y2 = n2["x"], n2["y"]
        dx, dy = x2 - x1, y2 - y1
        L = math.hypot(dx, dy)
        if L < 1e-12:
            continue
        c, s = dx / L, dy / L
        EA_L = e.get("E", model.get("defaults", {}).get("E", 210e9)) * e.get("A", model.get("defaults", {}).get("A", 1e-4)) / L
        ke = EA_L * np.array(
            [
                [ c*c,  c*s, -c*c, -c*s],
                [ c*s,  s*s, -c*s, -s*s],
                [-c*c, -c*s,  c*c,  c*s],
                [-c*s, -s*s,  c*s,  s*s],
            ],
            dtype=float,
        )
        dofs = (*dof_index(n1["id"]), *dof_index(n2["id"]))
        for i in range(4):
            for j in range(4):
                K[dofs[i], dofs[j]] += ke[i, j]

    for ld in loads_list:
        ux_i, uy_i = dof_index(ld["node"])  
        R[ux_i] += ld.get("fx", 0.0)
        R[uy_i] += ld.get("fy", 0.0)

    fixed_dofs = []
    for spt in supports_list:
        nid = spt["node"]
        ux_flag = bool(spt.get("ux", False))
        uy_flag = bool(spt.get("uy", False))
        dof_x, dof_y = dof_index(nid)
        if ux_flag:
            fixed_dofs.append(dof_x)
        if uy_flag:
            fixed_dofs.append(dof_y)

    all_dofs = np.arange(n_dof)
    fixed_dofs = np.array(sorted(set(fixed_dofs)), dtype=int)
    free_dofs = np.array([d for d in all_dofs if d not in set(fixed_dofs)], dtype=int)

    return K, R, free_dofs, fixed_dofs, id_to_idx


def solve_truss(model: Dict):
    K, R, free, fixed, id_to_idx = assemble_system(model)
    n_dof = K.shape[0]
    U = np.zeros(n_dof, dtype=float)

    Kff = K[np.ix_(free, free)]
    Rf = R[free]
    Uf = np.linalg.solve(Kff, Rf)
    U[free] = Uf

    reac_full = K @ U - R
    reactions = []
    for d in fixed:
        node_index = d // 2
        nid = next(k for k, v in id_to_idx.items() if v == node_index)
        if d % 2 == 0:
            reactions.append({"node": nid, "rx": float(reac_full[d]), "ry": 0.0})
        else:
            reactions.append({"node": nid, "rx": 0.0, "ry": float(reac_full[d])})

    reac_by_node: Dict[int, Tuple[float, float]] = {}
    for r in reactions:
        nid = r["node"]
        rx, ry = reac_by_node.get(nid, (0.0, 0.0))
        rx += r.get("rx", 0.0)
        ry += r.get("ry", 0.0)
        reac_by_node[nid] = (rx, ry)
    reactions_combined = [{"node": nid, "rx": float(rx), "ry": float(ry)} for nid, (rx, ry) in reac_by_node.items()]

    sum_rx = float(sum(r["rx"] for r in reactions_combined))
    sum_ry = float(sum(r["ry"] for r in reactions_combined))
    sum_R = float(math.hypot(sum_rx, sum_ry))


    node_ids = sorted(id_to_idx.keys(), key=lambda k: id_to_idx[k])
    node_results = []
    for nid in node_ids:
        i = id_to_idx[nid]
        node_results.append({"id": nid, "ux": float(U[2 * i]), "uy": float(U[2 * i + 1])})

    elem_results = []
    nodes = model["nodes"]
    elements = model["elements"]
    for e in elements:
        n1 = next(n for n in nodes if n["id"] == e["n1"])
        n2 = next(n for n in nodes if n["id"] == e["n2"])
        i1 = id_to_idx[n1["id"]]
        i2 = id_to_idx[n2["id"]]
        x1, y1 = n1["x"], n1["y"]
        x2, y2 = n2["x"], n2["y"]
        dx, dy = x2 - x1, y2 - y1
        L = math.hypot(dx, dy)
        if L < 1e-12:
            N = 0.0
            strain = 0.0
            stress = 0.0
        else:
            c, s = dx / L, dy / L
            ue = np.array([U[2 * i1], U[2 * i1 + 1], U[2 * i2], U[2 * i2 + 1]])
            delta = np.array([-c, -s, c, s]) @ ue
            E = e.get("E", model.get("defaults", {}).get("E", 210e9))
            A = e.get("A", model.get("defaults", {}).get("A", 1e-4))
            N = (E * A / L) * delta
            strain = delta / L
            stress = E * strain
        elem_results.append({
            "id": e["id"],
            "force": float(N),
            "strain": float(strain),
            "stress": float(stress),
        })

    results = {
        "nodes": node_results,
        "elements": elem_results,
        "reactions": reactions_combined,
        "reactions_sum": {"rx": sum_rx, "ry": sum_ry, "R": sum_R},
        "metadata": {
            "dof": int(n_dof),
            "free_dofs": int(getattr(free, 'size', len(free))),
            "fixed_dofs": int(getattr(fixed, 'size', len(fixed)))
        },
    }
    return results


def write_results(results: Dict):
    ensure_temp_dir()
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Ergebnisse gespeichert: {RESULTS_PATH}")


def run_solver():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modelldatei nicht gefunden: {MODEL_PATH}. Bitte zuerst den Preprocessor ausführen.")
    model = read_model()
    results = solve_truss(model)
    write_results(results)


if __name__ == "__main__":
    run_solver()

