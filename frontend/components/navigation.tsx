"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
export function Navigation() {
  const pathname = usePathname();
  return <header className="topbar"><Link className="brand" href="/agent"><span className="brand-mark">S</span><span>SalesOps<span className="brand-sub">INVESTIGATION WORKSPACE</span></span></Link><nav aria-label="Main navigation">{[["/agent", "Investigation"], ["/business", "Business snapshot"], ["/eval", "Evaluation"]].map(([href, label]) => <Link key={href} href={href} aria-current={pathname === href ? "page" : undefined}>{label}</Link>)}</nav><span className="demo-tag">● Synthetic data demo</span></header>;
}
