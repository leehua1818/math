# OCR Spike 样本集标注规范（PRD §20.4 阶段 0 / AC-128）

## 1. 样本要求

- **60–100 张**去标识化图片（JPG / JPEG / PNG / WebP），放入 `images/`。
- 覆盖：手写中文、英文、数字、数学公式，以及**三者混排**；含清晰、轻微倾斜、
  低对比度、涂改、上下标密集、多行推导、多图同题等情况。
- 拍摄/导出前请去掉姓名、学号、学校、面部等个人信息（PRD §25.2-7）。
- 按 `group` 分 `clean`（清晰）与 `hard`（困难）两组，两组都要有样本。
- ⚠️ `images/` 与 `eval/ocr_predictions.jsonl`（含识别出的题面文本）属学生数据，
  已加入 `.gitignore`；**不要提交 Git、不要放进部署包**，按保留期清理（附录 A-14）。

## 2. `ground_truth.jsonl` 字段（每行一个 JSON 对象）

| 字段 | 必填 | 说明 |
|---|---|---|
| `sample_id` | ✅ | 唯一，如 `s001` |
| `image` | ✅ | 相对本目录的图片路径，如 `images/s001.jpg` |
| `transcription_markdown` | ✅ | 人工逐字/逐公式核对的完整转写；行内公式 `$...$`，独立 `$$...$$` |
| `group` | | `clean` \| `hard`，默认 `clean` |
| `content_tags` | | `zh` / `en` / `math` / `mixed` 数组，用于分组统计 |
| `document_type` | | `question` / `student_work` / `mixed` / `unknown` |
| `formulas` | | 按阅读顺序的公式 LaTeX 数组（**不含定界符**），用于 Exact Match |
| `critical_symbols` | | 本样本的关键符号（如 `\lim`、`\frac`、`dx`、`^`），用于专项错误统计 |
| `notes` | | 标注备注 |

**Ground Truth 必须由教师/标注人员人工核对**：上下标、括号、分数线、根号范围、
积分上下限、极限方向、矩阵行列位置。不得用任何 OCR 的输出当 GT（会自我验证）。

## 3. 运行

```bash
# 零成本自检（推荐先做）
python scripts/ocr_benchmark.py selftest
python scripts/ocr_benchmark.py run --providers mock

# 真实对比
python scripts/ocr_benchmark.py run --providers vision,mathpix,baidu,cmd

# 多视觉模型对比（同批样本，一次出横向对比表）
OCR_VISION_MODELS="Qwen/Qwen3-VL-32B-Instruct,Qwen/Qwen3-VL-8B-Instruct" \
  python scripts/ocr_benchmark.py run --providers vision,cmd
```

> `OCR_VISION_MODELS` 逗号分隔；`--providers vision` 会展开成 `vision:<model>` 虚拟 provider，
> 每个模型独立计时/计费/判门槛。也可直接写 `--providers vision:模型A,vision:模型B`。
> 未配置该变量时回退到单模型 `OCR_VISION_MODEL`，行为与旧版一致。

## 4. Provider 环境变量（只存服务器 `.env`，不要写进 Git）

```
OCR_VISION_BASE=...            # OpenAI 兼容 /chat/completions 基址
OCR_VISION_KEY=...             # 密钥
OCR_VISION_MODELS=...          # 逗号分隔多模型（优先），如 Qwen/Qwen3-VL-32B-Instruct,Qwen/Qwen3-VL-8B-Instruct
OCR_VISION_MODEL=...           # 单模型回退；未配 MODELS 时使用（向后兼容）
OCR_VISION_COST_PER_IMAGE=...  # 元/张，人工按实际价目表填；0 表示未配置
OCR_IMAGE_DETAIL=high          # 硅基流动仅支持 low|high|auto；OCR 用 high，设 none 则不传

MATHPIX_APP_ID=... MATHPIX_APP_KEY=... MATHPIX_COST_PER_IMAGE=...
BAIDU_OCR_API_KEY=... BAIDU_OCR_SECRET_KEY=... BAIDU_OCR_COST_PER_IMAGE=...

OCR_LOCAL_CMD=...              # 本地开源库适配命令，见下
OCR_PROMPT_FILE=...            # 可选，自定义 OCR-only 系统提示词
OCR_TIMEOUT_SECONDS=45
```

> 未配置的 provider 会自动跳过并在报告中标注，**不会静默失败**。
> 单价默认 0，报告会提示「未配置，请人工补」，不臆造价格（PRD §27.2）。

## 5. 接入本地开源库（路径 E）

`OCR_LOCAL_CMD` 必须含且仅含一个 `{image}`；`{image}` 作为独立参数传入，
**不经 shell**（Windows 请用正斜杠路径，如 `C:/tools/adapter.py`，或成对引号包裹）。
命令 stdout 只需输出一个统一 JSON 对象：

```json
{"transcription_markdown": "求极限：\n\n$$\\lim_{x\\to 0} \\frac{\\sin x}{x}$$",
 "formulas": ["\\lim_{x\\to 0} \\frac{\\sin x}{x}"]}
```

适配器由你按所选库自行写一层薄封装（脚本不内置任何 OCR SDK），例如：

```python
# tools/p2t_adapter.py（示意，Pix2Text）
import json, sys
from pix2text import Pix2Text
md = Pix2Text().recognize_text(sys.argv[1])
print(json.dumps({"transcription_markdown": md, "formulas": [...]}, ensure_ascii=False))
```

```bash
OCR_LOCAL_CMD='python tools/p2t_adapter.py {image}' \
  python scripts/ocr_benchmark.py run --providers cmd
```

PaddleOCR-VL / TrOCR / Qwen3-VL 自托管同理，只需输出上述 JSON。
这样同一批样本可与路径 A/B/C 用**完全相同的指标**对比，且图片不出网。

## 6. 报告口径

- 文本 CER：剥离公式 → NFKC 全半角统一 → 去空白 →（可选）去标点，字符级编辑距离/参考字符数。
- 公式 Exact Match：去首尾定界符 + 去空白后**完全相等**，按阅读顺序配对；数量不等计未命中。
- 整题可用率：文本 CER ≤ `--usable-cer`(默认 0.10) 且所有公式命中 且关键符号无缺失。
- AC-128 门槛：清晰文本 CER ≤5%、全样本 ≤10%、清晰公式 EM ≥90%、全部公式 EM ≥80%。
- 纯公式样本（GT 剥离公式后无文本）**不计入文本 CER 加权**，由公式 EM 单独反映。
- **样本量门槛（三态判定：通过 / 未通过 / 样本不足）**：
  - 总有效样本 ≥ `--min-samples`（默认 60，依据 PRD AC-128 的 60–100 张）。
  - 四个子项分母 ≥ `--min-per-check`（默认 20；clean 子项取其一半，即 10）。
    分母依次为：清晰文本 CER = clean 组有文本样本数、全样本 CER = 有文本样本数、
    清晰公式 EM = clean 组公式区域数、全部公式 EM = 全部公式区域数。
  - 分母不足的子项记「样本不足」，**不判通过也不判未通过**；整体与「指标未达标」区分，
    避免 3 张样本 CER=0 就宣称达标。报告明细表逐项列出「分母 / 最少」。
  - 小样本试跑可临时调低：`--min-samples 3 --min-per-check 1 --allow-fail`，
    但**试跑结论不得用于服务商选型或对外宣称达标**。
- `--aggressive-latex` 会额外统一 `\dfrac/\tfrac`、去 `\left/\right`；默认**关闭**以避免刷分。
