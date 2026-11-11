import { forwardRef, type InputHTMLAttributes } from "react";

/** Shared input primitive styled by the AgriVision global design system. */
export const Input = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(
  ({ className = "", ...props }, ref) => <input ref={ref} className={`field ${className === "field" ? "" : className}`} {...props} />,
);
Input.displayName = "Input";
