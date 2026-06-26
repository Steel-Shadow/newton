# Newton 刚体碰撞虚拟实验

这是一个基于 Newton 物理引擎的虚拟实验项目。当前场景包含地面、斜面、小球和大量多米诺骨牌：小球沿斜面滚下，撞击入口骨牌，并触发链式倒下。骨牌可以生成直线、圆形、螺旋、波浪和混合展示图案，用于提升课程展示中的视觉效果。

项目定位不是单纯播放一个预设动画，而是逐步做成一个可调参数、可交互、可记录实验结果的虚拟物理实验平台。最终展示可以是 Newton 自带交互 3D viewer、录屏视频，或进一步接入 VR 渲染环境。

## 当前功能

- 地面、斜面、小球、多米诺骨牌刚体场景。
- 默认 `96` 块骨牌，使用 `showcase` 混合布局：入口直线和带入口缺口的圆形环。
- 小球和入口骨牌之间加入软体缓冲块，用于展示不同柔软度和阻尼对碰撞传递的影响。
- 支持 `line`、`circle`、`spiral`、`wave`、`showcase` 多种骨牌图案。
- XPBD 刚体求解和 SAP 碰撞 broad phase。
- 小球初速度、斜面角度、骨牌数量、骨牌间距、图案尺度、软体材料参数等命令行参数。
- 自动测试入口和圆环链式反应是否发生。
- 暂停编辑模式：暂停时可以选择斜面、小球或任意骨牌，直接平移和旋转物体。
- 支持恢复单个物体或恢复整个场景到初始状态。

## 运行

在仓库根目录运行交互可视化：

```powershell
uv run --extra examples python src\main.py
```

无窗口验证：

```powershell
uv run python src\main.py --viewer null --test --quiet
```

## 图案示例

默认混合展示图案：

```powershell
uv run --extra examples python src\main.py --domino-pattern showcase --domino-count 120
```

圆形图案：

```powershell
uv run --extra examples python src\main.py --domino-pattern circle --domino-count 96
```

螺旋图案：

```powershell
uv run --extra examples python src\main.py --domino-pattern spiral --domino-count 120 --pattern-scale 1.2
```

波浪图案：

```powershell
uv run --extra examples python src\main.py --domino-pattern wave --domino-count 120
```

## 柔软度实验

默认场景会启用一个自由软体缓冲块。它位于小球和第一张骨牌之间，小球先压缩并推动缓冲块，再由缓冲块把动量传给骨牌。可以通过材料刚度和阻尼观察链式反应是否被削弱。

较硬的软橡胶：

```powershell
uv run --extra examples python src\main.py --soft-k-mu 30000 --soft-k-lambda 100000 --soft-damping 0.8
```

更软、更吸能的材料：

```powershell
uv run --extra examples python src\main.py --soft-k-mu 5000 --soft-k-lambda 25000 --soft-damping 6.0
```

关闭软体缓冲块，回到纯刚体链式碰撞：

```powershell
uv run --extra examples python src\main.py --disable-soft-buffer
```

## 交互编辑

启动后打开 Newton viewer，使用左侧面板：

1. 勾选 `Pause` 暂停仿真。
2. 在 `Example Options` 里的 `Paused Object Editor` 选择对象。
3. 调整 `X/Y/Z` 或 `Roll/Pitch/Yaw`。
4. 暂停状态下滑条会立即应用到仿真状态。
5. 可使用 `Restore Object` 恢复当前对象，或 `Restore Scene` 恢复整个实验。

如果希望应用位姿时清空该物体速度，可勾选 `Zero velocity on apply`。

## 常用参数

```powershell
uv run --extra examples python src\main.py --domino-count 160 --domino-spacing 0.34 --ball-speed 2.8 --ramp-angle 20
```

可调参数：

- `--domino-count`：骨牌总数量，默认 `96`。
- `--domino-spacing`：沿生成路径相邻骨牌中心间距 `[m]`，默认 `0.36`。
- `--domino-pattern`：骨牌图案，可选 `showcase`、`line`、`circle`、`spiral`、`wave`。
- `--pattern-scale`：圆形、螺旋、波浪等图案的尺度系数，默认 `1.0`。
- `--ball-speed`：小球沿斜面方向的初始速度 `[m/s]`，默认 `2.6`。
- `--ramp-angle`：斜面角度 `[deg]`，默认 `18.0`。
- `--iterations`：XPBD 每个 substep 的迭代次数，默认 `8`。
- `--sim-substeps`：每帧物理子步数，默认 `16`。提高该值会更稳定但降低 FPS。
- `--disable-soft-buffer`：关闭小球和入口骨牌之间的软体缓冲块。
- `--soft-density`：软体缓冲块密度 `[kg/m^3]`，默认 `100.0`。
- `--soft-k-mu`：软体剪切刚度，默认 `10000`。
- `--soft-k-lambda`：软体体积保持刚度，默认 `50000`。
- `--soft-damping`：软体材料阻尼，默认 `1.0`。
- `--soft-contact-margin`：刚体-软体接触生成距离 `[m]`，默认 `0.01`。
- `--num-frames`：仿真帧数，默认 `600`。
- `--viewer`：Newton viewer 类型，可用 `gl`、`null`、`usd` 等。

## 主要物理参数

几何常量位于 `src/main.py` 顶部：

```python
DOMINO_COUNT = 96
DOMINO_HALF_THICKNESS = 0.035
DOMINO_HALF_WIDTH = 0.14
DOMINO_HALF_HEIGHT = 0.40
DOMINO_SPACING = 0.36

BALL_RADIUS = 0.18
BALL_DENSITY = 350.0
RAMP_LENGTH = 3.8
RAMP_WIDTH = 0.9
RAMP_THICKNESS = 0.12

SOFT_BUFFER_DENSITY = 100.0
SOFT_BUFFER_K_MU = 1.0e4
SOFT_BUFFER_K_LAMBDA = 5.0e4
SOFT_BUFFER_K_DAMP = 1.0
SOFT_BUFFER_CONTACT_CLEARANCE = 0.03
```

如果链式反应太容易或太难触发，可以优先调整：

- `--domino-spacing`
- `--ball-speed`
- `--ramp-angle`
- `--domino-pattern`
- `--soft-k-mu`
- `--soft-k-lambda`
- `--soft-damping`
- 多米诺密度和摩擦参数
- `--iterations`

## 后续方向

推荐继续扩展为完整课程大作业：

- 增加实验数据记录，例如每块骨牌倒下时间、碰撞次数、实验是否成功。
- 增加批量实验，扫描不同速度、角度、间距、图案下的成功率。
- 输出 CSV/JSON 和统计图表。
- 支持添加或删除骨牌，并通过重建模型应用场景变化。
- 将每帧刚体位姿导出给 Three.js、Unity 或 VR 渲染层。
