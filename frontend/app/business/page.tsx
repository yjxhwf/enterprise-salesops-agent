"use client";
import { useEffect, useState } from "react";
import { API_BASE_URL } from "@/lib/config";
type Measures={revenue:string;profit:string;margin:string;order_count:number};
type Row={region?:string;product_id?:string;current:Measures};
type Snapshot={overview:{current:Measures;changes:{revenue_growth_pct:string;profit_growth_pct:string;margin_change_pp:string}};regions:{regions:Row[]};products:{products:Row[]}};
const money=(v:string)=>Number(v).toLocaleString("en-US",{minimumFractionDigits:2,maximumFractionDigits:2});
const percent=(v:string)=>`${(Number(v)*100).toFixed(2)}%`;
const change=(v:string,suffix="%")=>`${Number(v)>0?"+":""}${Number(v).toFixed(2)}${suffix}`;
function BusinessTable({rows,type}:{rows:Row[];type:string}){return <div className="table-wrap"><table><thead><tr><th>{type}</th><th>REVENUE</th><th>PROFIT</th><th>MARGIN</th></tr></thead><tbody>{rows.map((r,i)=><tr key={i}><td>{r.region||r.product_id}</td><td>¥{money(r.current.revenue)}</td><td>¥{money(r.current.profit)}</td><td>{percent(r.current.margin)}</td></tr>)}</tbody></table></div>;}
export default function BusinessPage(){
  const [data,setData]=useState<Snapshot|null>(null);const [error,setError]=useState("");const [attempt,setAttempt]=useState(0);
  useEffect(()=>{const controller=new AbortController();let active=true;const timer=setTimeout(()=>controller.abort(),10000);setError("");
    fetch(`${API_BASE_URL}/api/business`,{cache:"no-store",signal:controller.signal}).then(r=>{if(!r.ok)throw new Error();return r.json();}).then((body:Snapshot)=>{if(active)setData(body);}).catch(()=>{if(active)setError("无法读取业务快照。请确认后端已启动且合成数据库已 Seed。");}).finally(()=>clearTimeout(timer));return()=>{active=false;controller.abort();clearTimeout(timer);};},[attempt]);
  return <main className="workspace"><div className="page-heading"><div><p className="eyebrow">The data behind the investigation</p><h1>Business snapshot</h1><p className="subtitle">Agent 查询的实际结构化数据库。所有指标由冻结 READ Tools 计算，不由模型生成。</p></div><span className="badge neutral">01–31 AUG 2026</span></div>
    {error&&<div className="error" role="alert">{error} <button className="secondary" onClick={()=>setAttempt(x=>x+1)}>重试读取</button></div>}
    {!data&&!error&&<section className="panel" role="status">正在读取本地 SQLite…</section>}
    {data&&<><div className="stats">{[
      ["AUGUST REVENUE",`¥${money(data.overview.current.revenue)}`,`${change(data.overview.changes.revenue_growth_pct)} vs previous period`],
      ["AUGUST PROFIT",`¥${money(data.overview.current.profit)}`,`${change(data.overview.changes.profit_growth_pct)} vs previous period`],
      ["PROFIT MARGIN",percent(data.overview.current.margin),`${change(data.overview.changes.margin_change_pp," pp")} vs previous period`],
      ["AUGUST ORDERS",String(data.overview.current.order_count),"Actual synthetic order records"]].map(([label,value,detail])=><section className="stat" key={label}><p className="stat-label">{label}</p><p className="stat-value">{value}</p><p className="stat-detail">{detail}</p></section>)}</div>
      <p className="info-strip">比较口径：前一等长周期（2026-07-01 至 2026-07-31），不是同比。利润下降不等于亏损。</p><div className="two-columns"><section className="panel"><div className="panel-head"><h2>Regional performance</h2><span className="small muted">August 2026</span></div><BusinessTable rows={data.regions.regions} type="REGION"/></section><section className="panel"><div className="panel-head"><h2>Product performance</h2><span className="small muted">August 2026</span></div><BusinessTable rows={data.products.products} type="PRODUCT"/></section></div></>}
    <p className="note">Deterministic synthetic enterprise dataset · 不包含现实企业或客户数据。这里的快照不会导入 Agent Prompt，也不提供预期答案。</p></main>;
}
