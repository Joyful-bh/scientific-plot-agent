# Scientific Plot Agent

自然语言驱动的科研绘图 Agent 系统。用户用自然语言描述绘图需求，系统调用 1.5B 小模型将需求转化为结构化的 PlotSpec JSON，再由工具层渲染为 publication-ready 的科学图表，支持 Nature / IEEE / NeurIPS 等主流期刊风格。

---

## 目录结构

```
scientific-plot-agent/
├── schema.py              # 全局 Schema 定义（唯一共享契约）
├── model/
│   └── generator.py       # A线：generate_spec()，现为 Mock
├── tools/
│   ├── loader.py          # B线：DataLoader，CSV 解析 + 数据摘要
│   ├── themes.py          # B线：ThemeConfig + THEMES/PALETTES 注册表
│   └── renderer.py        # B线：PlotRenderer，图表渲染，现为 Mock
├── system/
│   ├── validator.py       # C线：PlotSpec 合法性校验
│   ├── merger.py          # C线：多轮 delta 合并 + 默认值填充
│   └── agent.py           # C线：PlotAgent 主循环
├── ui/
│   └── app.py             # C线：Gradio 界面
├── data/
│   ├── example_bar.csv    # 示例数据：6 模型 × 4 数据集准确率
│   ├── example_line.csv   # 示例数据：训练曲线
│   └── placeholder.png    # Mock 渲染占位图
├── tests/                 # 单元测试 + 集成测试
├── output/                # 生成图表（.gitignore 忽略）
└── requirements.txt
```

---

## 快速开始

```bash
pip install -r requirements.txt
python ui/app.py
```

浏览器打开 `http://localhost:7860`，上传 `data/example_bar.csv`，然后输入：

> 画一张柱状图，对比各模型在 SST-2 数据集上的准确率，使用 nature 风格

---

## 三条开发线接口说明

### A线（`model/generator.py`）

**目标**：接入 1.5B 本地模型，将用户意图转化为 PlotSpec JSON。

```python
def generate_spec(
    user_input: str,
    data_context: str,
    current_spec: dict | None = None,
) -> dict:
    ...
```

- 首轮（`current_spec=None`）：返回包含所有 `REQUIRED_FIELDS` 的完整 PlotSpec dict。
- 修改轮：返回仅含变更字段的 delta dict，由 `merger.py` 合并。
- **只替换函数体，不修改签名。**

### B线（`tools/renderer.py`）

**目标**：实现各图表类型的真实渲染逻辑。

```python
def render_plot(spec: dict, data_source: str) -> str:
    ...

def _render_bar(df: pd.DataFrame, spec: dict, theme: ThemeConfig) -> plt.Figure: ...
def _render_line(df: pd.DataFrame, spec: dict, theme: ThemeConfig) -> plt.Figure: ...
def _render_scatter(df: pd.DataFrame, spec: dict, theme: ThemeConfig) -> plt.Figure: ...
def _render_box(df: pd.DataFrame, spec: dict, theme: ThemeConfig) -> plt.Figure: ...
def _render_heatmap(df: pd.DataFrame, spec: dict, theme: ThemeConfig) -> plt.Figure: ...
```

通过 `tools/themes.py` 的 `apply_theme()` 获取 `ThemeConfig`，按 spec 参数渲染对应图表。

### C线（`system/agent.py` + `ui/app.py`）

**目标**：完善 Agent 状态管理和 UI 交互体验。

- `PlotAgent.process_input()` 已实现完整流水线。
- UI 层通过 `PlotAgent` 实例调用，不直接接触 model/ 或 tools/。

---

## PlotSpec 字段说明

### 必填字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `chart_type` | str | 图表类型：`bar` / `line` / `scatter` / `box` / `heatmap` |
| `data_source` | str | CSV 文件路径 |
| `data_x` | str | X 轴列名（字符串，非列的值） |
| `data_y` | str | Y 轴列名（字符串，非列的值） |
| `style_theme` | str | 风格主题：`nature` / `ieee` / `neurips` / `clean` / `morandi` |

### 可选字段（含默认值）

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `data_group_by` | `null` | 分组列名，用于多系列图表 |
| `data_error` | `null` | 误差棒列名 |
| `data_filter` | `null` | 行过滤条件（pandas query 语法） |
| `label_title` | `""` | 图表标题 |
| `label_x` | `""` | X 轴标签 |
| `label_y` | `""` | Y 轴标签 |
| `axes_y_min` / `axes_y_max` | `null` | Y 轴范围 |
| `axes_x_tick_rotation` | `0` | X 轴刻度旋转角度 |
| `style_palette_override` | `null` | 配色覆盖：`morandi` / `nature_d` / `tab10` / `coolwarm` |
| `params_orientation` | `"vertical"` | 柱状图方向（bar 专用） |
| `params_stacked` | `false` | 堆叠柱状图（bar 专用） |
| `params_markers` | `true` | 显示数据点标记（line 专用） |
| `params_smooth` | `false` | 平滑曲线（line 专用） |
| `params_alpha` | `0.8` | 透明度（scatter 专用） |
| `params_show_regression` | `false` | 显示回归线（scatter 专用） |
| `params_annot` | `true` | 显示数值标注（heatmap 专用） |
| `params_fmt` | `".2f"` | 数值格式（heatmap 专用） |

---

## 运行测试

```bash
pytest tests/ -v
```
