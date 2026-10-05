"use client";
import { useState } from "react";
import { API_BASE_URL } from "@/lib/config";
import { displayLabel } from "@/lib/display-labels";

const examples = ["分析2026年8月为什么销售额增长但利润下降，找出最主要的三个原因，并给出可以执行的建议。", "分析2026年8月华东区域销售风险，结合销售政策提出一个需要人工审批的后续动作，但不要直接执行。"];
const stages = ["Understanding", "Planning", "Business Investigation", "Policy Retrieval", "Synthesis", "Action Proposal"];
type Result = {status:string; stop_reason:string|null; clarification:string|null; executive_interpretation:string|null;
  findings:{title:string; interpretation:string; facts:string[]; evidence_ids:string[]}[];
  recommendations:{title:string; action:string; policy_interpretation?:string; policy_evidence_ids:string[]}[];
  business_evidence:{evidence_id:string; source_tool:string; measurements:string[]}[];
  policy_evidence:{evidence_id:string; excerpt:string; citation:{title:string; doc_id:string; section_title:string; source_path:string; version:string}}[];
  pending_action:{tool_name:string; reason:string; risk_level:string; evidence_ids:string[]; expected_outcome:string; expires_at:string}|null;
  errors:{node:string; code:string}[]; limitations:string[]; counts:{llm_calls:number;read_tools:number;retries:number}};
type Event = {type:"progress";stage:string;status:string;read_tools:number;retries:number}|{type:"result";result:Result}|{type:"error";code:string};

export default function AgentPage(){
  const [query,setQuery]=useState(examples[0]); const [running,setRunning]=useState(false);
  const [status,setStatus]=useState("READY"); const [seen,setSeen]=useState<string[]>([]);
  const [current,setCurrent]=useState(""); const [result,setResult]=useState<Result|null>(null);
  const [error,setError]=useState(""); const [counts,setCounts]=useState({read_tools:0,retries:0});
  async function run(){
    if(running||!query.trim())return;
    setRunning(true);setError("");setResult(null);setSeen([]);setCurrent("Understanding");setStatus("RUNNING");setCounts({read_tools:0,retries:0});
    let completed=false;
    function receive(line:string){
      if(!line.trim())return;const event=JSON.parse(line) as Event;
      if(event.type==="progress"){setStatus(event.status);setCurrent(event.stage);setSeen(v=>v.includes(event.stage)?v:[...v,event.stage]);setCounts({read_tools:event.read_tools,retries:event.retries});}
      if(event.type==="result"){completed=true;setResult(event.result);setStatus(event.result.status);}
      if(event.type==="error")throw new Error(event.code);
    }
    try{
      const response=await fetch(`${API_BASE_URL}/api/agent/run`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({query}),cache:"no-store"});
      if(!response.ok)throw new Error(response.status===409?"已有调查正在运行，请等待其结束。":response.status===422?"请输入 1–4000 字的有效问题。":`运行接口不可用（HTTP ${response.status}）。`);
      if(!response.body)throw new Error("浏览器无法读取进度流。");
      const reader=response.body.getReader();const decoder=new TextDecoder();let pending="";
      while(true){const {value,done}=await reader.read();if(done)break;pending+=decoder.decode(value,{stream:true});const lines=pending.split("\n");pending=lines.pop()||"";lines.forEach(receive);}
      pending+=decoder.decode();receive(pending);if(!completed)throw new Error("连接中断，尚未收到最终结果。请先检查服务端状态，不要连续重试。");
    }catch(e){setError(e instanceof Error?e.message:"运行失败，请检查后端连接。");setStatus("UNAVAILABLE");}
    finally{setRunning(false);}
  }
  return <main className="workspace"><div className="page-heading"><div><p className="eyebrow">证据 → 洞察 → 人工审阅行动</p><h1>从经营问题，追溯到证据。</h1><p className="subtitle">查询结构化业务数据，结合销售制度形成建议。行动始终停在人工审批之前。</p></div><span className="badge neutral">调查工作区</span></div>
    <div className="agent-grid"><aside><section className="panel"><div className="panel-head"><h2><span className="section-number">01</span>提出问题</h2></div><label className="query-label" htmlFor="query">你想了解什么？</label><textarea id="query" value={query} onChange={e=>setQuery(e.target.value)} maxLength={4000} disabled={running}/><button className="primary" onClick={()=>void run()} disabled={running||!query.trim()}>{running?"调查中…":"开始调查"}<span aria-hidden="true">↗</span></button><p className="small muted" style={{marginTop:12}}>调用真实模型，可能需要数分钟。</p><div style={{marginTop:22}}><p className="eyebrow">示例问题</p>{examples.map((q,i)=><button key={q} disabled={running} className="example" onClick={()=>setQuery(q)}>{i===0?"01 / 收入增长，利润为什么下降？":"02 / 华东风险与待审批跟进动作"}</button>)}</div><p className="note">全部业务数据与政策均为合成示范。浏览器不持有模型密钥或审批令牌。</p></section>
      <section className="panel"><div className="panel-head"><h3>调查进度</h3><span className="small muted">{seen.length}/6</span></div><ol className="progress-list">{stages.map((stage,i)=><li key={stage} className={`${seen.includes(stage)?"seen":""} ${current===stage&&running?"active":""}`}><span className="step-dot">{seen.includes(stage)?"✓":i+1}</span>{displayLabel(stage)}</li>)}</ol><p className="small muted" style={{marginTop:12}}>阶段按实际执行点亮；政策检索和动作提议可能不需要。</p></section></aside>
      <div><section className="panel" aria-live="polite"><div className="panel-head"><h2><span className="section-number">02</span>调查结果</h2><span className={`badge ${status==="ERROR"||status==="UNAVAILABLE"||status==="RUNTIME_STOPPED"?"warn":""}`}>{displayLabel(status)}</span></div>{error&&<div className="error" role="alert">{error}</div>}
        {!result?<div className="empty-state"><div className="empty-icon" aria-hidden="true">≋</div><h3>{running?"正在沿证据展开调查":"让结论有据可查"}</h3><p>{running?"当前阶段："+displayLabel(current)+"。页面展示执行进度，不展示模型隐藏推理。":"输入经营问题后，智能体将查询业务数据、核对证据并形成建议；涉及业务写入的动作必须经过人工审批。"}</p><div className="workflow"><span>业务数据</span><span>政策依据</span><span>有据结论</span></div></div>:<>
          {result.clarification&&<div className="error">{result.clarification}</div>}{result.errors.map((e,i)=><div className="error" key={i}>{displayLabel(e.node)}: {displayLabel(e.code)}</div>)}
          {result.stop_reason&&<p className="small muted">停止原因：{displayLabel(result.stop_reason)}</p>}
          {result.executive_interpretation&&<p style={{margin:"18px 0"}}>{result.executive_interpretation}</p>}
          {result.findings.map((f,i)=><article className="result-card" key={i}><h3>{i+1}. {f.title}</h3><p>{f.interpretation}</p><ul className="measurements">{f.facts.map((fact,j)=><li key={j}>{fact}</li>)}</ul>{f.evidence_ids.map(id=><p key={id} className="evidence-link">{id}</p>)}</article>)}
          {!result.findings.length&&<p className="info-strip">本次尚未生成已验证结论。失败状态和已取得的证据保留在页面下方。</p>}
          <div className="count-row"><span>LLM 调用次数 {result.counts.llm_calls}</span><span>只读 Tool 调用次数 {result.counts.read_tools}</span><span>重试次数 {result.counts.retries}</span></div></>}
        {running&&<div className="count-row"><span>只读 Tool 调用次数 {counts.read_tools}</span><span>重试次数 {counts.retries}</span></div>}</section>
        {result&&<><section className="panel"><div className="panel-head"><h2><span className="section-number">03</span>证据</h2><span className="small muted">可追溯来源</span></div><div className="evidence-columns"><div><h3>业务证据 <span className="muted">({result.business_evidence.length})</span></h3><p className="small muted">支持经营事实与数字</p>{result.business_evidence.map(e=><details key={e.evidence_id}><summary>{e.source_tool} · {e.evidence_id}</summary><ul className="measurements">{e.measurements.map((m,i)=><li key={i}>{m}</li>)}</ul></details>)}</div><div><h3>政策证据 <span className="muted">({result.policy_evidence.length})</span></h3><p className="small muted">支持制度依据，不证明经营事实</p>{result.policy_evidence.map(e=><details key={e.evidence_id}><summary>{e.citation.doc_id} · {e.citation.title}</summary><p>{e.excerpt}</p><p className="small muted">{e.citation.section_title} · {e.citation.version}</p><p className="evidence-link">{e.evidence_id}</p></details>)}{!result.policy_evidence.length&&<p className="info-strip">本次未取得政策引用。</p>}</div></div></section>
          <section className="panel"><h2>行动建议</h2>{result.recommendations.map((r,i)=><article key={i} className="result-card"><h3>{r.title}</h3><p>{r.action}</p>{r.policy_interpretation&&<p className="muted small">{r.policy_interpretation}</p>}{r.policy_evidence_ids.map(id=><p key={id} className="evidence-link">{id}</p>)}</article>)}{!result.recommendations.length&&<p className="info-strip">尚无已验证建议。</p>}{result.limitations.map((l,i)=><p className="note" key={i}>{l}</p>)}</section>
          {result.pending_action&&<section className="panel pending"><div className="panel-head"><h2>待审批行动</h2><span className="badge warn">待人工审批</span></div><dl><dt>Tool</dt><dd>{result.pending_action.tool_name}</dd><dt>提议原因</dt><dd>{result.pending_action.reason}</dd><dt>风险等级</dt><dd>{displayLabel(result.pending_action.risk_level)}</dd><dt>预期结果</dt><dd>{result.pending_action.expected_outcome}</dd><dt>证据</dt><dd>{result.pending_action.evidence_ids.join(" · ")}</dd></dl><p className="note">尚未执行。此演示页面仅供审阅，不接收审批令牌，也不提供写入按钮；后续审批须由可信应用层按现有安全边界完成。</p></section>}</>}
      </div></div></main>;
}
