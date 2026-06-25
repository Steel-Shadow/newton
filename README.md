# Newton Rigid-Body Domino Experiment

这是一个基于 Newton 物理引擎的刚体碰撞仿真实验。场景包含地面、斜面、小球和多米诺骨牌：小球沿斜面滚下，撞击第一块多米诺骨牌，并触发后续骨牌的链式倒下。

当前实现使用 Newton 自带 viewer 进行可视化和调试，可作为后续接入 VR 显示、VR 交互或沉浸式实验界面的物理仿真核心。

## 场景组成

- 地面：静态平面，提供摩擦接触。
- 斜面：静态倾斜 box，角度默认 `18 deg`。
- 小球：动态刚体 sphere，带初始速度并受重力影响。
- 多米诺骨牌：动态刚体 box，默认 `12` 块，中心间距 `0.36 m`。
- 求解器：`newton.solvers.SolverXPBD`。
- 碰撞管线：默认 broad phase 为 `sap`。

## 运行方式

在仓库根目录执行：

```powershell
uv run --extra examples python src\main.py
```

如果只想无窗口验证物理过程和自动测试：

```powershell
uv run python src\main.py --viewer null --test --quiet
```

## 常用参数

```powershell
uv run --extra examples python src\main.py --domino-count 16 --ball-speed 2.8 --ramp-angle 20
```

可调参数：

- `--domino-count`：多米诺骨牌数量，默认 `12`。
- `--ball-speed`：小球沿斜面方向的初始速度 `[m/s]`，默认 `2.4`。
- `--ramp-angle`：斜面角度 `[deg]`，默认 `18.0`。
- `--iterations`：XPBD 每个 substep 的迭代次数，默认 `8`。
- `--num-frames`：仿真帧数，默认 `320`。
- `--viewer`：Newton viewer 类型，可用 `gl`、`null`、`usd` 等。

## 代码结构

- `Example.__init__()`：创建场景、模型、碰撞管线、求解器和 viewer。
- `_add_ramp()`：添加固定斜面。
- `_add_ball()`：添加小球并设置初始线速度和角速度。
- `_add_dominoes()`：添加一排多米诺骨牌。
- `simulate()`：执行碰撞检测和 XPBD 时间推进。
- `render()`：把刚体状态和接触点写入 viewer。
- `test_final()`：检查仿真状态有限，并确认有足够多的骨牌发生倾倒。

## 物理参数

主要几何常量位于 `src/main.py` 顶部：

```python
DOMINO_COUNT = 12
DOMINO_HALF_THICKNESS = 0.035
DOMINO_HALF_WIDTH = 0.14
DOMINO_HALF_HEIGHT = 0.40
DOMINO_SPACING = 0.36

BALL_RADIUS = 0.18
RAMP_LENGTH = 3.8
RAMP_WIDTH = 0.9
RAMP_THICKNESS = 0.12
```

如果多米诺太容易或太难连锁倒下，可以优先调整：

- `DOMINO_SPACING`
- `--ball-speed`
- `--ramp-angle`
- 多米诺密度和摩擦参数
- `--iterations`

## VR 集成方向

当前脚本负责物理仿真和 Newton viewer 可视化。后续接入 VR 时，可以保留 Newton 的模型构建和 `simulate()` 主循环，再把每帧的 `state_0.body_q` 刚体位姿同步到 VR 渲染层。

推荐的下一步：

- 将小球、斜面、骨牌的位姿导出到 VR 渲染对象。
- 用 VR 控制器添加重置、暂停、拖拽物体、改变斜面角度等交互。
- 如果需要实验记录，可记录每帧刚体位姿、接触事件和骨牌倒下时间。
