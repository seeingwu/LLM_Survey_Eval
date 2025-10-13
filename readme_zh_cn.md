# llm_survey_eval（中文）

[![English](https://img.shields.io/badge/lang-English-blue)](readme.md) [![简体中文](https://img.shields.io/badge/语言-简体中文-red)](README.zh-CN.md)

`llm_survey_eval` 是一个用于评估 **LLM 生成问卷数据** 与 **人类问卷数据** 的四层（Tier‑1→4）方法工具包，当前以 **GitHub 源码安装** 为主（可通过克隆或 ZIP 解压安装）。当 Tier‑4 稳定后将考虑发布 PyPI 版本。

---

## 安装（两种方式）

### 方式 A：克隆仓库
```bash
git clone https://github.com/<your-username>/llm_survey_eval.git
cd llm_survey_eval
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e .
```

### 方式 B：下载 ZIP（无需 Git）
1. 在 GitHub 页点击 **Code → Download ZIP**（或使用我们提供的打包 ZIP）。
2. 解压后进入目录：`cd llm_survey_eval`
3. 创建并激活虚拟环境：
   - Windows PowerShell：
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - macOS/Linux：
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
4. 安装本地源码（可编辑模式）：
   ```bash
   pip install --upgrade pip
   pip install -e .
   ```
5. （可选）安装 Jupyter 并运行演示：
   ```bash
   pip install jupyter
   jupyter notebook examples/demo_full_pipeline.ipynb
   ```

> Python ≥ 3.9；依赖：NumPy、pandas、SciPy、scikit‑learn、statsmodels（详见 `pyproject.toml`）。

---

## 数据要求
为了保证方法可比性，请准备两份 CSV：**Human** 与 **LLM**。两者需满足：

1. **列名与类型一致**：
   - 两个文件使用**完全一致**的特征名与类型（例如 `shopping_frequency`=有序、`shopping_mode`=名义）。
   - 在代码中通过 `ordered_features`、`nominal_features`，或 Tier‑4 的 `feature_schema` 显式声明类型。

2. **已数值化（非字符串）**：
   - Ordinal：整数编码（1–5、1–6 等）。
   - Nominal：整数 ID（1–7 等）；内部基于**固定类别**进行 one‑hot 对齐。
   - Continuous：浮点；Binary：0/1（若不是 0/1，会按中位数阈值转布尔）。

3. **可选的 ID 对齐**：
   - 若包含公共主键 `agent_id`，Tier‑1/2 采用内连接对齐样本；否则各自计算边际与结构。

4. **固定类别/层级（强烈建议）**：
   - 对名义变量与名义结局建议显式提供 `categories/levels`（如 `[1,2,3,4,5,6,7]`），以保证 one‑hot 维度与基准一致；否则可能出现系数无法一一匹配的情况。

> 若原始数据是文本标签，请先完成编码映射；Human 与 LLM 应使用**同一份映射字典**。

---

## 四层方法概览
1. **Tier‑1：描述性相似度**（边际分布）  
   Nominal/Ordinal：TV、JS、χ²、G‑test、Cramér’s V；Ordinal 另含 W₁、均值/方差比。Continuous：KS、W₁、均值/方差。
2. **Tier‑2：结构一致性**（成对关联）  
   Ordered–ordered：Spearman ρ；Nominal–nominal：Cramér’s V；Ordered–nominal：η。输出 Human/LLM/Diff 矩阵与 MAE/RMSE/|max| 汇总。
3. **Tier‑3：联合形状**  
   混合嵌入（有序→[0,1]；名义→固定类别 one‑hot），计算能量距离（√ED²）、高斯核 MMD（中位数带宽启发）与 C2ST AUC。
4. **Tier‑4：推断等价性**  
   有序结局→**Ordered Logit**；名义结局→**Multinomial Logit**。以 Human 为参照，计算 DCR（方向一致率）与 SMR（显著性匹配率），并输出系数级对齐明细。

---

## 快速上手
参见 [(examples_demo_full_pipeline.ipynb)]（覆盖 **Tiers 1–4**）。Notebook 构造可复现的玩具 Human/LLM 数据，演示：
- 有序重叠直方图、名义占比柱状图；
- Tier‑2 三联热图与 |Δ| 热图；
- Tier‑3 的 PCA 投影；
- Tier‑4 的 DCR/SMR 汇总与系数级明细。

---

## 当前局限
- 目前主打 **GitHub 源码安装**；PyPI 发布待 Tier‑4 稳定后进行。
- 可视化为研究版示例，将逐步引入**更丰富的图形表达**（雷达图、森林图、交互矩阵等）。
- 指标体系计划补充**更多可自定义的 metrics**（能量距离口径、正则/稳健回归、特征级 DCR/SMR 聚合等）。
- 对极端稀疏类别与小样本的鲁棒性仍需扩展测试，建议先做类别合并或样本充足性检查。

---

## 许可与引用
MIT Licence（见 `LICENSE`）。

> Wu, S. (2025). *Evaluating LLM‑Generated Survey Data: A Four‑Tier Framework for Behavioural Equivalence.* University of Leeds.  
> Software: `llm_survey_eval`, v0.1.0.

