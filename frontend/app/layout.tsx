import type { Metadata } from "next";
import "./globals.css";
import { Navigation } from "@/components/navigation";

export const metadata: Metadata = {
  title: "SalesOps 智能体",
  description: "基于业务证据的销售经营分析、政策检索与待审批行动。作品集 / 工程演示。",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="zh-CN"><body><Navigation />{children}<footer className="site-footer"><span>SalesOps 智能体</span><span>作品集 / 工程演示 · 本地运行、自主性有界、证据支撑</span></footer></body></html>;
}
