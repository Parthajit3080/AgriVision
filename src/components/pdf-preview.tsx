"use client";

import { useEffect, useRef, useState } from "react";
import { Download, LoaderCircle, Printer, X } from "lucide-react";

export function PdfPreview({ url, onClose }: { url: string; onClose: () => void }) {
  const pages = useRef<HTMLDivElement>(null);
  const dialog = useRef<HTMLDivElement>(null);
  const closeButton = useRef<HTMLButtonElement>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const previouslyFocused = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const oldOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    closeButton.current?.focus();

    function keepDialogAccessible(event: KeyboardEvent) {
      if (event.key === "Escape") {
        event.preventDefault();
        onClose();
        return;
      }
      if (event.key !== "Tab") return;
      const focusable = dialog.current?.querySelectorAll<HTMLElement>('a[href], button:not([disabled]), [tabindex]:not([tabindex="-1"])');
      if (!focusable?.length) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }

    document.addEventListener("keydown", keepDialogAccessible);
    return () => {
      document.removeEventListener("keydown", keepDialogAccessible);
      document.body.style.overflow = oldOverflow;
      previouslyFocused?.focus();
    };
  }, [onClose]);

  useEffect(() => {
    let cancelled = false;
    async function renderPdf() {
      try {
        const pdfjs = await import("pdfjs-dist");
        pdfjs.GlobalWorkerOptions.workerSrc = "/pdf.worker.min.mjs";
        const pdf = await pdfjs.getDocument({ url }).promise;
        if (!pages.current || cancelled) return;
        pages.current.replaceChildren();
        for (let pageNo = 1; pageNo <= pdf.numPages; pageNo++) {
          const page = await pdf.getPage(pageNo);
          const viewport = page.getViewport({ scale: 1.2 });
          const canvas = document.createElement("canvas");
          const context = canvas.getContext("2d");
          if (!context) continue;
          canvas.width = viewport.width;
          canvas.height = viewport.height;
          canvas.className = "pdf-page shadow-lg";
          pages.current.append(canvas);
          await page.render({ canvas, canvasContext: context, viewport }).promise;
          if (cancelled) return;
        }
      } catch (cause) {
        if (!cancelled) setError(cause instanceof Error ? cause.message : "PDF preview could not be rendered.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    renderPdf();
    return () => { cancelled = true; };
  }, [url]);

  return <div ref={dialog} className="pdf-preview fixed inset-0 z-50 flex flex-col bg-[#15251c]/95 backdrop-blur-sm" role="dialog" aria-modal="true" aria-label="PDF field report preview" aria-labelledby="pdf-preview-title" aria-busy={loading}>
    <div className="flex min-h-16 items-center justify-between border-b border-white/10 px-5 text-white sm:px-8">
      <div><b id="pdf-preview-title" className="text-sm">Field report PDF</b><span className="ml-3 hidden text-xs text-white/55 sm:inline">Rendered with PDF.js</span></div>
      <div className="pdf-preview-actions flex items-center gap-2"><a href={url} download="agrivision-field-report.pdf" className="inline-flex items-center gap-2 rounded-lg bg-[#d7e5c0] px-3.5 py-2.5 text-xs font-bold text-[#24432f]"><Download size={15}/> Download PDF</a><button onClick={() => window.print()} className="inline-flex items-center gap-2 rounded-lg border border-white/20 px-3.5 py-2.5 text-xs font-semibold text-white"><Printer size={15}/> Print</button><button ref={closeButton} onClick={onClose} className="rounded-lg p-2 text-white/75 hover:bg-white/10" aria-label="Close PDF preview"><X size={18}/></button></div>
    </div>
    <div className="pdf-preview-content flex-1 overflow-y-auto px-4 py-6 sm:px-8"><div ref={pages} className="mx-auto grid max-w-[850px] justify-items-center gap-6"/>{loading && <div role="status" aria-live="polite" className="pdf-loading mt-8 flex items-center justify-center gap-2 text-sm text-white/70"><LoaderCircle size={18} className="animate-spin"/>Rendering pages…</div>}{error && <p role="alert" className="mx-auto mt-8 max-w-xl rounded-xl bg-white p-4 text-sm text-[#8b4d39]">PDF preview error: {error} You can still download the generated PDF.</p>}</div>
    <style jsx global>{`@media print { body > * { visibility:hidden !important } [role="dialog"][aria-label="PDF field report preview"], [role="dialog"][aria-label="PDF field report preview"] * { visibility:visible !important } [role="dialog"][aria-label="PDF field report preview"] { position:absolute !important; inset:0 !important; overflow:visible !important; background:#fff !important } [role="dialog"][aria-label="PDF field report preview"] > div:first-child { display:none !important } [role="dialog"][aria-label="PDF field report preview"] > div:nth-child(2) { overflow:visible !important; padding:0 !important } .pdf-page { width:100% !important; height:auto !important; box-shadow:none !important; break-after:page } }`}</style>
  </div>;
}
