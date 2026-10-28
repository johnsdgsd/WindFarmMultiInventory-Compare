from __future__ import annotations

import sys

sys.dont_write_bytecode = True

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(PROJECT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

from ss_static.policy import SSStaticPolicy  # noqa: E402

sys.stdout.reconfigure(errors="replace")

PASS = 0


def check(cond, msg):
    global PASS
    if not cond:
        raise AssertionError(msg)
    PASS += 1


def fake_prob():
    return SimpleNamespace(
        T=10, N_R=2, G=2, L=2,
        N=np.array([10.0, 10.0]),
        Qmax=np.array([3.0, 3.0]),
        Imax=np.array([6.0, 6.0]),
        Cs=20.0, h=1.0,
        tc_route=np.array([[0.0, 10.0], [10.0, 0.0]]),
        priority={1: [1, 2], 2: [2, 1]},
    )


OPTS = {
    "position": "net",
    "passive_transshipment": True,
    "cross_cost_rule": "all",
    "max_passive_cost": 20.0,
    "active_transfer": False,
    "fill": "local_then_priority",
}


def zeros(prob, P=1):
    I = np.zeros((P, prob.N_R))
    o = np.zeros((P, prob.L - 1, prob.N_R))
    n = np.tile(np.asarray(prob.N)[None, :, None],
                (P, 1, prob.G)).astype(float)
    q = np.zeros((P, prob.N_R))
    return I, o, n, q


def main():
    global PASS
    prob = fake_prob()

    # stage 0：严格 IP<s；s=0 不下单，s>0 下单
    I, o, n, q = zeros(prob)
    p0 = SSStaticPolicy(prob, [0, 0], [1, 1], OPTS)
    check(np.allclose(p0.order(0, I, o, n, q), 0.0), "stage0 s=0 不应下单")
    p1 = SSStaticPolicy(prob, [1, 1], [1, 1], OPTS)
    check(np.allclose(p1.order(0, I, o, n, q), 1.0), "stage0 s=1 应订 1")
    ph = SSStaticPolicy(prob, [0.5, 0.5], [1, 1], OPTS)
    check(np.allclose(ph.order(0, I, o, n, q), 1.0),
          "stage0 s=0.5,S=1 应补到 S=1")

    # 负 s 不是 docs/s,S.md 的策略空间
    try:
        SSStaticPolicy(prob, [-1, 0], [1, 1], OPTS)
    except ValueError:
        PASS += 1
    else:
        raise AssertionError("负 s 应被拒绝")

    # 严格触发：IP=s 不下单，IP<s 下单
    I1 = np.array([[1.0, 0.0]])
    qq = p1.order(1, I1, o, n, q)
    check(np.allclose(qq, [[0.0, 1.0]]), f"IP=s 不应下单: {qq}")
    I2 = np.array([[0.99, 0.0]])
    p2 = SSStaticPolicy(prob, [1, 1], [2, 2], OPTS)
    qq2 = p2.order(1, I2, o, n, q)
    check(np.allclose(qq2, [[1.01, 2.0]]), f"IP<s 应订: {qq2}")

    # Qmax 截断（MD 说 R 足够；本模型 q_out 还有 Qmax 变量上界）
    pq = SSStaticPolicy(prob, [0.5, 0.5], [6, 6], OPTS)
    check(np.allclose(pq.order(1, I, o, n, q), [[3.0, 3.0]]),
          "Qmax 截断失败")

    # 容量 R=Imax−gross：I=5,W=1 -> R=1；不应再加 out_known
    I3 = np.array([[5.0, 0.0]])
    n3 = np.array([[[9.0, 0.0], [10.0, 0.0]]])
    pcap = SSStaticPolicy(prob, [4.5, 0.5], [6, 6], OPTS)
    qc = pcap.order(1, I3, o, n3, q)
    check(np.allclose(qc, [[1.0, 3.0]]), f"容量 R 截断: {qc}")

    # 无停订偏移：t=T 仍按规则订货（q_T 免费，但规则照常执行）
    pT = SSStaticPolicy(prob, [1, 1], [1, 1], OPTS)
    check(np.allclose(pT.order(prob.T, I, o, n, q), 1.0), "t=T 应仍按规则")

    # 本地优先：后面的仓不会被前面的跨仓请求抢走本地库存
    y = p0.fulfill(1, np.array([[0.0, 2.0]]), np.array([[2.0, 2.0]]))
    check(np.allclose(y[0], [[0.0, 0.0], [0.0, 2.0]]),
          f"local-first y={y[0]}")

    # 全配对跨仓：dest 1 用 source 0；t=T 也允许（MD §3）
    y2 = p0.fulfill(1, np.array([[2.0, 0.0]]), np.array([[0.0, 2.0]]))
    check(np.allclose(y2[0, 1, 0], 2.0) and np.allclose(y2.sum(), 2.0),
          f"cross y={y2[0]}")
    yT = p0.fulfill(prob.T, np.array([[2.0, 0.0]]), np.array([[0.0, 2.0]]))
    check(np.allclose(yT[0, 1, 0], 2.0), f"t=T 跨仓应允许: {yT[0]}")
    yloc = p0.fulfill(prob.T, np.array([[2.0, 0.0]]),
                      np.array([[1.0, 0.0]]))
    check(np.allclose(yloc[0, 0, 0], 1.0), f"t=T 本地替换: {yloc[0]}")

    # fixed 阈值只作为消融；all 模式忽略 max_passive_cost
    opts_t = dict(OPTS, cross_cost_rule="fixed", max_passive_cost=5.0)
    pt = SSStaticPolicy(prob, [0, 0], [1, 1], opts_t)
    yt = pt.fulfill(1, np.array([[2.0, 0.0]]), np.array([[0.0, 2.0]]))
    check(np.allclose(yt, 0.0), f"fixed 阈值应禁止 10 的边: {yt[0]}")

    # JSON round-trip
    d = p1.params()
    pr = SSStaticPolicy.from_dict(prob, d)
    check(np.allclose(pr.s, p1.s) and np.allclose(pr.S, p1.S),
          "from_dict round-trip")

    print(f"_t_policy: {PASS} checks passed")


if __name__ == "__main__":
    main()
