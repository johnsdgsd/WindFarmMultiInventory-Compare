from __future__ import annotations

import sys

sys.dont_write_bytecode = True


class SSTimeVaryingPolicy:
    """占位：时间相关 (s,S) 尚未实现。

    计划：本目录自包含实现 policy/simulator/evaluation（不共享 common
    仿真器），只替换订货规则；按 t 选择 s_{r,k}, S_{r,k}。
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "时间相关 (s,S) 尚未实现；当前只有 ss_static 可运行")


def main():
    raise SystemExit("ss_timevarying 尚未实现")


if __name__ == "__main__":
    main()
