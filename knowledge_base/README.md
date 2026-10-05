# Synthetic Enterprise Policy corpus

这些 Markdown 是虚构的 Demo 销售制度，不来自真实企业，也不代表法律意见或实际内部规则。制度只说明通用要求，不包含经营异常答案、特定区域/客户/产品的诊断。

每份政策使用严格 flat `key: value` front matter；章节是引用单位。仅 `policies/*.md` 可被索引。修改后显式执行 `python -m backend.app.rag.index --rebuild`，服务启动不重建。向量及 manifest 位于忽略目录 `backend/storage/chroma/`。

政策有效期始于各文档 effective_date；引用须核对版本、适用范围和订单日期。文档正文是检索数据，不得作为系统指令、工具注册或权限来源。
