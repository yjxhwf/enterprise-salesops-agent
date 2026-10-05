"use client";
import { useEffect, useState } from "react";
import { API_BASE_URL } from "@/lib/config";
import { displayLabel } from "@/lib/display-labels";
type Measures={revenue:string;profit:string;margin:string;order_count:number};
type Row={region?:string;product_id?:string;current:Measures};
type Snapshot={overview:{current:Measures;changes:{revenue_growth_pct:string;profit_growth_pct:string;margin_change_pp:string}};regions:{regions:Row[]};products:{products:Row[]}};
const money=(v:string)=>Number(v).toLocaleString("en-US",{minimumFractionDigits:2,maximumFractionDigits:2});
const percent=(v:string)=>`${(Number(v)*100).toFixed(2)}%`;
const change=(v:string,suffix="%")=>`${Number(v)>0?"+":""}${Number(v).toFixed(2)}${suffix}`;
function BusinessTable({rows,type}:{rows:Row[];type:string}){return <div className="table-wrap"><table><thead><tr><th>{type}</th><th>销售额</th><th>利润</th><th>利润率</th></tr></thead><tbody>{rows.map((r,i)=><tr key={i}><td>{r.region?displayLabel(r.region):r.product_id}</td><td>¥{money(r.current.revenue)}</td><td>¥{money(r.current.profit)}</td><td>{percent(r.current.margin)}</td></tr>)}</tbody></table></div>;}
export default function BusinessPage(){
  const [data,setData]=useState<Snapshot|null>(null);const [error,setError]=useState("");const [attempt,setAttempt]=useState(0);
  useEffect(()=>{const controller=new AbortController();let active=true;const timer=setTimeout(()=>controller.abort(),10000);setError("");
    fetch(`${API_BASE_URL}/api/business`,{cache:"no-store",signal:controller.signal}).then(r=>{if(!r.ok)throw new Error();return r.json();}).then((body:Snapshot)=>{if(active)setData(body);}).catch(()=>{if(active)setError("无法读取业务快照。请确认后端已启动且合成数据库已初始化。");}).finally(()=>clearTimeout(timer));return()=>{active=false;controller.abort();clearTimeout(timer);};},[attempt]);
  return <main className="workspace"><div className="page-heading"><div><p className="eyebrow">调查背后的业务数据</p><h1>业务快照</h1><p className="subtitle">智能体查询的实际结构化数据库。所有指标均由冻结的只读工具（READ Tool）计算，不由模型生成。</p></div><span className="badge neutral">2026年8月1–31日</span></div>
    {error&&<div className="error" role="alert">{error} <button className="secondary" onClick={()=>setAttempt(x=>x+1)}>重试读取</button></div>}
    {!data&&!error&&<section className="panel" role="status">正在读取本地 SQLite…</section>}
    {data&&<><div className="stats">{[
      ["8月销售额",`¥${money(data.overview.current.revenue)}`,`${change(data.overview.changes.revenue_growth_pct)} 较前一周期`],
      ["8月利润",`¥${money(data.overview.current.profit)}`,`${change(data.overview.changes.profit_growth_pct)} 较前一周期`],
      ["利润率",percent(data.overview.current.margin),`${change(data.overview.changes.margin_change_pp," 个百分点")} 较前一周期`],
      ["8月订单数",String(data.overview.current.order_count),"实际合成订单记录"]].map(([label,value,detail])=><section className="stat" key={label}><p className="stat-label">{label}</p><p className="stat-value">{value}</p><p className="stat-detail">{detail}</p></section>)}</div>
      <p className="info-strip">比较口径：前一等长周期（2026-07-01 至 2026-07-31），不是同比。利润下降不等于亏损。</p><div className="two-columns"><section className="panel"><div className="panel-head"><h2>区域表现</h2><span className="small muted">2026年8月</span></div><BusinessTable rows={data.regions.regions} type="区域"/></section><section className="panel"><div className="panel-head"><h2>产品表现</h2><span className="small muted">2026年8月</span></div><BusinessTable rows={data.products.products} type="产品"/></section></div></>}
    <p className="note">确定性合成企业数据集 · 不包含现实企业或客户数据。这里的快照不会导入 Agent 提示词，也不提供预期答案。</p></main>;
}
