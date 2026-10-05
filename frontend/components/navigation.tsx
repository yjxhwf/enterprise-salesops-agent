"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
export function Navigation() {
  const pathname = usePathname();
  return <header className="topbar"><Link className="brand" href="/agent"><span className="brand-mark">S</span><span style={{whiteSpace:"nowrap"}}>SalesOps<br/>智能体<span className="brand-sub">经营分析工作台</span></span></Link><nav aria-label="主导航">{[["/agent", "智能调查"], ["/business", "业务快照"], ["/eval", "系统评测"]].map(([href, label]) => <Link key={href} href={href} aria-current={pathname === href ? "page" : undefined}>{label}</Link>)}</nav><span className="demo-tag">● 合成数据演示</span></header>;
}
