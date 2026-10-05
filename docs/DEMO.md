# SalesOps Agent — 5分钟演示

## 准备（演示前）

在仓库根目录配置本地 `.env`；模型Key只在服务端。按README安装既有依赖，明确执行合成数据Seed及政策索引构建。Seed `--reset` 会清空配置数据库里的本地演示任务/告警，先确认无需保留。索引构建和Live调查会调用配置的Provider。

```powershell
.venv\Scripts\python.exe -m backend.app.data.seed --reset
.venv\Scripts\python.exe -m backend.app.rag.index
.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

另开终端：

```powershell
cd frontend
# 首次 npm ci；复制 .env.example 为 .env.local，公开API地址指向127.0.0.1:8000
npm run build
npm start
```

打开 `http://localhost:3000/business`，确认指标可读。Backend health：`http://127.0.0.1:8000/health`。不要点击Run预热模型后又重复请求；若想展示已有结果，应明确标注为历史记录，不能冒充当前运行。

## 演示顺序（不超过5分钟）

**0:00–0:45 /business**：介绍4,005条合成订单、真实SQLite查询。展示8月收入、利润、利润率和667订单。说明前一等长周期不是同比，利润下降不等于负利润。

**0:45–3:30 /agent**：点击一个推荐问题并仅运行一次：

> 分析2026年8月为什么销售额增长但利润下降，找出最主要的三个原因，并给出可以执行的建议。结合销售政策，如有合适动作则提出需人工审批的Action，但不要直接执行。

说明Understanding → Planning → Investigation → 可选Policy → Synthesis → 可选Action Proposal。阶段反映真实执行，不展示思维链。展开Business与Policy两组Evidence，说明它们支持不同类型的主张。若出现Pending Action，指出它尚未写入；此页面不持有Token、不支持Approve/Reject按钮。

真实Provider可能超过演示窗口。3:30时即使仍运行，也可转到 `/eval` 介绍已有验收；切换页面不保证取消进行中的Provider调用。若连接失败或Validator拒绝，展示状态/错误码，不连续重跑、不把失败包装成成功。

**3:30–4:30 /eval**：Offline 24/25=96%，两次一致；Small Live Selector 4/5=80%，n=5。强调脚本回归不代表模型准确率，工具名F1不代表参数正确性。

**4:30–5:00 边界与工程取舍**：展示N4/L3/L4。说明有限Retry、循环预算、Evidence可见性、服务端审批、只读/写入分离；当前是Portfolio Demo，非分布式生产系统。

## 另一条演示问题

> 分析2026年8月华东区域销售风险，结合销售政策提出一个需要人工审批的后续动作，但不要直接执行。

每次点击Run会产生真实请求与费用。当前最终验收只安排一次Golden E2E，不为了展示而重跑或更改冻结逻辑。

## 审批与交付

现有 `/api/actions/{id}/approve`、`/reject` 仍要求服务端生成的能力Token。Demo页面不提供获取Token的端点，也不自动调用审批。可信应用集成需在Python应用层通过既有token sink安全交付Token，再明确决定批准或拒绝；不要把密钥、Token或其hash放进浏览器、日志或Git。

验收后检查 `git status --short` 与暂存范围。运行数据、真实Key、数据库和报告均不提交；创建GitHub仓库及push由项目所有者另行授权。
