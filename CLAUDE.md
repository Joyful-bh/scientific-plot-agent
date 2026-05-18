本文件是给 Claude Code 的全局约束说明。在本项目的任何任务中，
必须始终遵守以下规范，优先级高于任何局部指令。

---

## 项目背景与架构概述

这是一个自然语言驱动的科研绘图 Agent 系统，分三条开发线：
- A线（model/）：1.5B 小模型推理，输入用户意图+数据摘要，输出 PlotSpec JSON
- B线（tools/）：数据加载、图表渲染、审美配置
- C线（system/ + ui/）：Agent 主循环、Schema 校验、对话状态、Gradio UI

三条线通过两个接口解耦：
- A→C：generate_spec(user_input, data_context, current_spec) -> dict
- B→C：render_plot(spec, data_source) -> str

**schema.py 是唯一的共享契约**，所有枚举值、字段名、默认值只从这里读取，
任何文件都不允许硬编码这些值。

---

## 一、模块边界约束

### 严格禁止的跨模块行为

- model/ 不得 import tools/ 或 system/ 的任何内容
- tools/ 不得 import model/ 的任何内容
- ui/ 只允许 import system/agent.py，不得直接调用 model/ 或 tools/
- 任何模块都可以 import schema.py

### Mock 实现保护

model/generator.py 和 tools/renderer.py 目前是 Mock 实现。
修改这两个文件时：
- 只允许替换函数体内部实现
- 不允许修改函数签名（参数名、类型、返回值）
- 不允许在函数签名上新增必填参数
- Mock 注释"# Mock实现，X线替换"在真实实现完成前必须保留

---

## 二、schema.py 的使用规范

所有枚举值必须从 schema.py 引用，禁止在其他文件里出现魔法字符串。

```python
# ❌ 禁止
if spec["chart_type"] == "bar":
if spec["style_theme"] not in ["nature", "ieee", "neurips"]:

# ✅ 正确
from schema import CHART_TYPES, STYLE_THEMES
if spec["chart_type"] in CHART_TYPES:
if spec["style_theme"] not in STYLE_THEMES:
```

修改 schema.py 是高风险操作。如果任务需要新增或修改枚举值、字段名、
默认值，必须在操作前输出一条警告说明影响范围，不能静默修改。

---

## 三、PlotSpec 的核心设计约束

这是本项目最重要的数据结构，以下规则不得违反：

**data_x 和 data_y 存的是列名字符串，不是列的值。**

```python
# ❌ 错误（列的值）
"data_x": ["SST-2", "MR", "CR", "CoLA"]
"data_y": [93.5, 94.8, 94.2, 92.1]

# ✅ 正确（列名）
"data_x": "dataset"
"data_y": "accuracy"
```

这个约束在 validator.py 里有硬性检查。任何生成、构造、Mock PlotSpec
的代码都必须遵守这个规则，包括测试数据和示例数据。

PlotSpec 是 flat 结构，禁止出现嵌套 dict 作为字段值：

```python
# ❌ 禁止嵌套
{"data": {"x": "method", "y": "accuracy"}}

# ✅ 正确 flat
{"data_x": "method", "data_y": "accuracy"}
```

---

## 四、编码规范

### Python 版本与类型

- 使用 Python 3.10+
- 所有函数必须有类型注解
- 返回值是复合结构时，必须用 dataclass 定义，不允许返回裸 dict 或 tuple
  （例外：PlotSpec 本身用 dict，因为需要 JSON 序列化）

```python
# ❌ 禁止
def validate(spec: dict) -> tuple:
    return (False, ["data_x"], "请补充 data_x 字段")

# ✅ 正确
@dataclass
class ValidationResult:
    ok: bool
    missing_required: list[str]
    type_errors: list[str]
    prompt: str

def validate(spec: dict) -> ValidationResult:
    ...
```

### 文件路径

所有文件路径操作使用 pathlib.Path，禁止字符串拼接路径：

```python
# ❌ 禁止
path = "output/" + filename + ".png"
path = os.path.join("output", filename)

# ✅ 正确
from pathlib import Path
path = Path("output") / f"{filename}.png"
```

### 导入顺序

每个文件的 import 按以下顺序排列，组间空一行：
1. 标准库
2. 第三方库
3. 本项目模块（from schema import ... 等）

### 常量命名

模块级常量全大写加下划线：CHART_TYPES、OPTIONAL_DEFAULTS
不允许在函数内部定义常量，应提升到模块级或 schema.py。

---

## 五、错误处理规范

所有面向外部输入的函数（接收用户数据、文件路径、PlotSpec 的函数）
必须显式处理异常，不允许让异常直接传播到 UI 层。

```python
# ❌ 禁止
def load_data(source: str):
    df = pd.read_csv(source)  # 文件不存在会直接崩溃
    ...

# ✅ 正确
def load_data(source: str) -> tuple[str, str]:
    try:
        df = pd.read_csv(source)
    except FileNotFoundError:
        raise DataLoadError(f"文件不存在：{source}")
    except pd.errors.ParserError:
        raise DataLoadError(f"文件格式无法解析，请确认是合法的 CSV：{source}")
    ...
```

自定义异常类定义在各自模块内，命名规则：模块功能+Error：
- tools/loader.py → DataLoadError
- tools/renderer.py → RenderError
- system/validator.py → ValidationError（但 validate() 本身不抛出，
  返回 ValidationResult，由 agent.py 决定如何处理）

---

## 六、测试规范

- 每个新增的公开函数必须在对应的 tests/ 文件里有至少一个测试用例
- 测试函数命名：test_[函数名]_[场景描述]()
- 测试不允许依赖外部文件，使用 data/ 目录下的示例 CSV 或 tmp 文件
- Mock 实现的测试只验证接口契约（返回类型、必要字段），不验证业务逻辑

```python
# Mock 阶段的测试示例
def test_generate_spec_first_round_contract():
    """验证首轮调用返回包含所有 REQUIRED_FIELDS 的 dict"""
    from schema import REQUIRED_FIELDS
    result = generate_spec("画一张柱状图", "数据摘要...")
    assert isinstance(result, dict)
    for field in REQUIRED_FIELDS:
        assert field in result, f"缺少必要字段：{field}"
```

---

## 七、Gradio UI 约束

- 不使用 gr.Blocks 以外的布局方式
- 所有回调函数在 app.py 内定义，不允许在回调里直接调用 model/ 或 tools/，
  必须通过 PlotAgent 实例
- PlotAgent 实例在 app.py 模块级初始化，所有回调共享同一个实例
- 不允许在 UI 层做任何数据处理或业务逻辑

```python
# ❌ 禁止在回调里直接处理业务
def on_submit(user_input):
    spec = generate_spec(user_input, ...)
    result = render_plot(spec, ...)
    return result

# ✅ 正确，通过 agent 封装
agent = PlotAgent()

def on_submit(user_input):
    response = agent.process_input(user_input)
    if response.status == "ok":
        return response.image_path, ...
```

---

## 八、输出目录管理

- 渲染生成的图表保存到 output/ 目录
- output/ 目录由 renderer.py 在首次调用时自动创建，不预先存在
- 文件名格式：plot_{timestamp}_{chart_type}.png，timestamp 精确到秒
- output/ 已加入 .gitignore，不提交生成的图表文件
- data/ 目录下的示例文件和 placeholder.png 必须提交

---

## 九、禁止事项清单

在本项目的任何任务中，以下操作一律禁止，无论理由多充分：

| 禁止行为 | 原因 |
|----------|------|
| 在 schema.py 以外的文件硬编码枚举值 | 破坏单一数据源 |
| 修改 generate_spec / render_plot 的函数签名 | 破坏 A/B/C 线解耦 |
| 在 PlotSpec 中使用嵌套 dict | 违反 flat 设计原则 |
| data_x / data_y 填列的值而非列名 | 核心设计错误 |
| 在 ui/ 层直接调用 model/ 或 tools/ | 破坏分层架构 |
| 使用字符串拼接构造文件路径 | 跨平台兼容性问题 |
| 函数返回裸 tuple 表示复合结果 | 可读性和类型安全 |
| 让异常从工具函数直接传播到 UI 层 | 用户体验崩溃 |
| 在测试中访问外部网络或真实模型 | 测试稳定性 |
| 提交 output/ 目录下的生成文件 | 仓库污染 |

---

## 十、当任务描述与本文件冲突时

本文件的约束优先级高于对话中的临时指令。

如果某个任务要求违反上述规范（例如"直接在 UI 里调用 generate_spec"），
应当：
1. 指出该要求与 CLAUDE.md 的哪条规范冲突
2. 给出符合规范的替代实现方案
3. 等待确认后再执行

不允许静默地绕过约束执行任务。