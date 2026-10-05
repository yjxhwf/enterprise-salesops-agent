import type { Metadata } from "next";
import "./globals.css";
import { Navigation } from "@/components/navigation";

export const metadata: Metadata = {
  title: "Enterprise SalesOps Agent",
  description: "基于业务证据的销售经营分析、政策检索与待审批行动。Portfolio / Engineering Demo。",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="zh-CN"><body><Navigation />{children}<footer className="site-footer"><span>Enterprise SalesOps Agent</span><span>Portfolio / Engineering Demo · Local, bounded, evidence-backed</span></footer></body></html>;
}
