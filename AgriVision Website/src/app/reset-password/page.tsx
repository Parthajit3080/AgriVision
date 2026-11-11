import type { Metadata } from "next";
import { AuthPage } from "@/components/auth-pages";
export const metadata: Metadata = { title: "Choose a new password · AgriVision", robots: { index: false, follow: false } };
export default function ResetPasswordPage() { return <AuthPage kind="reset"/>; }
