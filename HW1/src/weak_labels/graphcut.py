# -*- coding: utf-8 -*-
"""超像素图上的多类图割（alpha-expansion，PyMaxflow 不可用时退化为 ICM）。"""
import numpy as np

try:
    import maxflow
except ImportError:  # pragma: no cover
    maxflow = None

BIG = 1e9


def _energy(f, D, edges, lam):
    e = float(D[np.arange(len(f)), f].sum())
    if len(edges) > 0:
        i = edges[:, 0].astype(np.int64)
        j = edges[:, 1].astype(np.int64)
        m = f[i] != f[j]
        e += float(lam * edges[m, 2].sum())
    return e


def _source_segment():
    """探测 PyMaxflow 中源点侧对应的 get_segment 取值（当前版本为 0）。"""
    g = maxflow.Graph[float]()
    n = g.add_nodes(1)
    g.add_tedge(n[0], 1.0, 0.0)
    g.maxflow()
    return int(g.get_segment(n[0]))


def _alpha_expansion(D, edges, hard, lam, n_labels, max_iter=8):
    src = _source_segment()
    f = hard.copy()
    f[f < 0] = 0
    best_e = _energy(f, D, edges, lam)

    for _ in range(max_iter):
        improved = False
        for a in range(n_labels):
            g = maxflow.Graph[float]()
            nodes = g.add_nodes(len(D))
            for i in range(len(D)):
                if hard[i] >= 0:
                    if hard[i] == a:
                        g.add_tedge(nodes[i], BIG, 0.0)
                    else:
                        g.add_tedge(nodes[i], 0.0, BIG)
                else:
                    others = np.delete(D[i], a)
                    g.add_tedge(nodes[i], float(others.min()), float(D[i, a]))
            for i, j, w in edges:
                cap = float(lam * w)
                g.add_edge(nodes[int(i)], nodes[int(j)], cap, cap)
            g.maxflow()

            y = np.array(
                [int(g.get_segment(nodes[i])) == src for i in range(len(D))], dtype=bool
            )
            cand = np.where(y, a, f)
            fixed = hard >= 0
            cand[fixed] = hard[fixed]
            e = _energy(cand, D, edges, lam)
            if e < best_e - 1e-6:
                f, best_e = cand, e
                improved = True
        if not improved:
            break
    return f, best_e


def _icm(D, edges, hard, lam, n_labels, max_iter=12):
    S = len(D)
    nbrs = [[] for _ in range(S)]
    for i, j, w in edges:
        nbrs[int(i)].append((int(j), float(w)))
        nbrs[int(j)].append((int(i), float(w)))

    f = hard.copy()
    f[f < 0] = 0
    for _ in range(max_iter):
        changed = 0
        for i in range(S):
            if hard[i] >= 0:
                continue
            cost = D[i].astype(np.float64).copy()
            for j, w in nbrs[i]:
                cost[f[j]] += lam * w
            best = int(np.argmin(cost))
            if best != f[i]:
                f[i] = best
                changed += 1
        if changed == 0:
            break
    return f, _energy(f, D, edges, lam)


def solve(D, edges, hard, lam=1.0, n_labels=None):
    """求解多类图割。

    D: [S, C] 数据项；edges: [E, 3]；hard: [S]，-1 无约束
    返回 (labels [S], energy, method)
    """
    n_labels = int(n_labels if n_labels is not None else D.shape[1])
    if maxflow is not None:
        try:
            f, e = _alpha_expansion(D, edges, hard, lam, n_labels)
            return f, e, "alpha-expansion"
        except Exception as exc:
            print(f"[warn] alpha-expansion 失败({exc})，改用 ICM")
    f, e = _icm(D, edges, hard, lam, n_labels)
    return f, e, "icm"
