import { jsPDF } from "jspdf";
import autoTable from "jspdf-autotable";

export type FieldReport = Record<string, any>;

const INK: [number, number, number] = [37, 56, 43];
const GREEN: [number, number, number] = [49, 90, 60];
const MUTED: [number, number, number] = [101, 116, 103];
const GRID: [number, number, number] = [225, 231, 222];

function drawHeader(doc: jsPDF, title: string, subtitle: string) {
  doc.setFillColor(25, 59, 43); doc.rect(0, 0, 210, 29, "F");
  doc.setTextColor(248, 249, 242); doc.setFont("helvetica", "bold"); doc.setFontSize(16); doc.text(title, 15, 13);
  doc.setFont("helvetica", "normal"); doc.setFontSize(8.5); doc.setTextColor(211, 225, 206); doc.text(subtitle, 15, 21);
}

function footer(doc: jsPDF, page: number) {
  doc.setFont("helvetica", "normal"); doc.setFontSize(8); doc.setTextColor(120, 132, 122);
  doc.text("AgriVision · Prototype decision support · Validate with local agronomy", 15, 289);
  doc.text(String(page), 195, 289, { align: "right" });
}

function drawYieldChart(doc: jsPDF, ranking: any[], y: number) {
  doc.setFont("helvetica", "bold"); doc.setFontSize(11); doc.setTextColor(...INK); doc.text("Top crop yield estimates", 15, y);
  doc.setFont("helvetica", "normal"); doc.setFontSize(8); doc.setTextColor(...MUTED); doc.text("Tonnes per hectare · horizontal scale starts at zero", 15, y + 5);
  const max = Math.max(1, ...ranking.map(row => Number(row.estimated_yield_t_ha ?? 0))) * 1.08;
  const xLabel = 15, xBar = 54, barWidth = 115, rowTop = y + 13, rowGap = 17, barHeight = 6;
  for (let tick = 0; tick <= 4; tick++) {
    const x = xBar + barWidth * tick / 4;
    doc.setDrawColor(...GRID); doc.setLineWidth(.2); doc.line(x, rowTop - 2, x, rowTop + rowGap * ranking.length - 4);
    doc.setFont("helvetica", "normal"); doc.setFontSize(7); doc.setTextColor(...MUTED); doc.text((max * tick / 4).toFixed(1), x, rowTop + rowGap * ranking.length - 1, { align: "center" });
  }
  ranking.forEach((row, i) => {
    const yRow = rowTop + i * rowGap;
    doc.setFont("helvetica", i === 0 ? "bold" : "normal"); doc.setFontSize(8.5); doc.setTextColor(...INK); doc.text(String(row.crop), xLabel, yRow + 5);
    const width = Math.max(1, barWidth * Number(row.estimated_yield_t_ha ?? 0) / max);
    doc.setFillColor(...(i === 0 ? GREEN : [151, 177, 137] as [number, number, number])); doc.roundedRect(xBar, yRow, width, barHeight, 1.5, 1.5, "F");
    doc.setFont("helvetica", "bold"); doc.setFontSize(8); doc.setTextColor(...INK); doc.text(Number(row.estimated_yield_t_ha ?? 0).toFixed(2), xBar + width + 2, yRow + 5);
  });
}

function drawClimateChart(doc: jsPDF, climate: any, selectedSeason: string, y: number) {
  const periods = climate?.risk_by_period ?? [];
  const outlook = climate?.seasonal_outlook ?? [];
  const seasons = ["Kharif", "Rabi", "Summer"];
  const colors: Record<string, [number, number, number]> = { Kharif: [77, 129, 145], Rabi: [170, 135, 80], Summer: [115, 145, 107] };
  const drawSeries = (field: "rainfall" | "humidity", label: string, unit: string, color: [number, number, number], top: number) => {
    const allValues = outlook.map((row: any) => Number(row[field] ?? 0));
    const low = allValues.length ? Math.min(...allValues) : 0;
    const high = allValues.length ? Math.max(...allValues) : 1;
    const pad = Math.max((high - low) * .35, high * .025, field === "rainfall" ? 20 : 2);
    const min = Math.max(0, Math.floor(low - pad));
    const max = field === "humidity" ? Math.min(100, Math.ceil(high + pad)) : Math.ceil(high + pad);
    const x = 31, width = 145, height = 27, bottom = top + height;
    doc.setFont("helvetica", "bold"); doc.setFontSize(8.5); doc.setTextColor(...color); doc.text(`${label} · ${unit}`, 15, top - 2);
    for (let tick = 0; tick <= 2; tick++) {
      const yy = bottom - height * tick / 2;
      doc.setDrawColor(...GRID); doc.setLineWidth(.2); doc.line(x, yy, x + width, yy);
      doc.setFont("helvetica", "normal"); doc.setFontSize(6.5); doc.setTextColor(...MUTED);
      doc.text(String(Math.round(min + (max - min) * tick / 2)), x - 2, yy + 1, { align: "right" });
    }
    const xAt = (i: number) => x + (periods.length <= 1 ? width / 2 : width * i / (periods.length - 1));
    const yAt = (v: number) => bottom - (v - min) / Math.max(1, max - min) * height;
    seasons.forEach((season, seasonIndex) => {
      const seasonRows = outlook.filter((row: any) => row.season === season);
      const valuesByYear = new Map(seasonRows.map((row: any) => [row.date.slice(0, 4), Number(row[field])]));
      const series = periods.map((row: any) => valuesByYear.get(row.date.slice(0, 4)));
      const lineColor = colors[season];
      let previous: { index: number; value: number } | null = null;
      series.forEach((rawValue: any, i: number) => {
        const value = Number(rawValue);
        if (!Number.isFinite(value)) { previous = null; return; }
        if (previous) { doc.setDrawColor(...lineColor); doc.setLineWidth(season === selectedSeason ? .9 : .5); doc.line(xAt(previous.index), yAt(previous.value), xAt(i), yAt(value)); }
        if (season === selectedSeason) { doc.setFillColor(...lineColor); doc.circle(xAt(i), yAt(value), .7, "F"); }
        previous = { index: i, value };
      });
      const legendX = 77 + seasonIndex * 38;
      doc.setDrawColor(...lineColor); doc.setLineWidth(season === selectedSeason ? 1 : .6); doc.line(legendX, top - 2, legendX + 6, top - 2);
      doc.setFont("helvetica", season === selectedSeason ? "bold" : "normal"); doc.setFontSize(6.5); doc.setTextColor(...lineColor); doc.text(season, legendX + 8, top);
    });
    periods.forEach((row: any, i: number) => {
      if (i === 0 || i === periods.length - 1 || i % 2 === 0) {
        doc.setFont("helvetica", "normal"); doc.setFontSize(6.5); doc.setTextColor(...MUTED);
        doc.text(String(row.date.slice(0, 4)), xAt(i), bottom + 4, { align: "center" });
      }
    });
  };
  doc.setFont("helvetica", "bold"); doc.setFontSize(11); doc.setTextColor(...INK); doc.text("Seasonal rainfall and humidity comparison", 15, y);
  doc.setFont("helvetica", "normal"); doc.setFontSize(7.5); doc.setTextColor(...MUTED);
  doc.text("State historical climate with simulated annual variability · selected season highlighted", 15, y + 5);
  drawSeries("rainfall", "Rainfall", "mm/season", [77, 129, 145], y + 17);
  drawSeries("humidity", "Humidity", "%", [154, 128, 81], y + 58);
}
export function buildFieldReportPdf(report: FieldReport): Blob {
  const doc = new jsPDF({ unit: "mm", format: "a4", compress: true });
  const margin = 15;
  const ranking = report.crop_profit_ranking?.recommendations ?? [];
  const plan = report.fertilizer_nutrient_plan?.plan;
  const climate = report.weather_climate_risk ?? {};
  const risk = climate.risk_guidance ?? {};
  const location = report.inputs?.location ?? {};
  const subtitle = `${location.district ?? "District not supplied"}, ${location.state ?? "State not supplied"} · ${report.inputs?.season ?? "Season"} ${report.inputs?.target_year ?? ""} · ${report.inputs?.land_size ?? "—"} ${report.inputs?.land_unit ?? ""}`;

  drawHeader(doc, "AgriVision Field Report", subtitle);
  doc.setFont("helvetica", "bold"); doc.setFontSize(13); doc.setTextColor(...INK); doc.text("Crop ranking · top five", margin, 41);
  autoTable(doc, {
    startY: 45, margin: { left: margin, right: margin },
    head: [["Crop", "Yield t/ha", "Market price / tonne", "Revenue / ha", "Profit / ha", "Risk / 100"]],
    body: ranking.map((row: any) => [row.crop, Number(row.estimated_yield_t_ha).toFixed(2), row.price_per_ton == null ? "—" : `INR ${Math.round(row.price_per_ton).toLocaleString("en-IN")}`, row.expected_gross_revenue_per_ha == null ? "—" : `INR ${Math.round(row.expected_gross_revenue_per_ha).toLocaleString("en-IN")}`, row.expected_profit_per_ha == null ? "Not supplied" : `INR ${Math.round(row.expected_profit_per_ha).toLocaleString("en-IN")}`, Number(row.downside_risk_score ?? 1).toFixed(1)]),
    theme: "grid", styles: { fontSize: 8.5, cellPadding: 2.5, textColor: [48, 62, 51] }, headStyles: { fillColor: GREEN, textColor: 255 }, alternateRowStyles: { fillColor: [246, 248, 243] },
  });
  let y = (doc as any).lastAutoTable.finalY + 8;
  doc.setFont("helvetica", "bold"); doc.setFontSize(10); doc.setTextColor(...INK); doc.text(`Why ${ranking[0]?.crop ?? "this crop"} is the top pick`, margin, y);
  doc.setFont("helvetica", "normal"); doc.setFontSize(9); doc.setTextColor(...MUTED);
  const rationaleLines = doc.splitTextToSize(report.crop_profit_ranking?.top_pick_reason ?? "Ranked first by modeled crop and market criteria.", 180);
  doc.text(rationaleLines, margin, y + 5);
  drawYieldChart(doc, ranking, y + 12 + rationaleLines.length * 4);
  footer(doc, 1);

  doc.addPage();
  drawHeader(doc, "Climate exposure & actions", subtitle);
  const score = Math.max(1, Number(risk.score ?? climate.risk_score ?? 1));
  const level = climate.risk_level ?? "Low";
  doc.setFillColor(...(level === "High" ? [250, 238, 231] as [number, number, number] : level === "Moderate" ? [249, 246, 231] as [number, number, number] : [239, 246, 235] as [number, number, number]));
  doc.roundedRect(15, 35, 180, 29, 3, 3, "F");
  doc.setFont("helvetica", "bold"); doc.setFontSize(22); doc.setTextColor(...INK); doc.text(score.toFixed(1), 21, 52);
  doc.setFontSize(9); doc.setTextColor(...MUTED); doc.text("/ 100", 47, 52);
  doc.setFontSize(12); doc.setTextColor(...INK); doc.text(`${level} climate exposure`, 67, 46);
  doc.setFont("helvetica", "normal"); doc.setFontSize(8.5); doc.setTextColor(...MUTED);
  doc.text(doc.splitTextToSize(risk.disclaimer ?? "Prototype climate planning index; not a probability of loss.", 120), 67, 53);

  const targetYear = report.inputs?.target_year;
  doc.setFont("helvetica", "bold"); doc.setFontSize(8.5); doc.setTextColor(...INK); doc.text(`Risk by season · ${targetYear}`, 15, 72);
  ["Kharif", "Rabi", "Summer"].forEach((season, index) => {
    const item = (climate.seasonal_outlook ?? []).find((row: any) => row.season === season && row.date.startsWith(String(targetYear)));
    const x = 15 + index * 61;
    doc.setFillColor(246, 248, 243); doc.roundedRect(x, 75, 56, 12, 2, 2, "F");
    doc.setFont("helvetica", "normal"); doc.setFontSize(7); doc.setTextColor(...MUTED); doc.text(season, x + 3, 80);
    doc.setFont("helvetica", "bold"); doc.setFontSize(8.5); doc.setTextColor(...INK); doc.text(`${Number(item?.risk_score ?? 0).toFixed(0)} / 100`, x + 3, 85);
  });
  doc.setFont("helvetica", "bold"); doc.setFontSize(11); doc.setTextColor(...INK); doc.text("What to do · what to avoid", 15, 94);
  let guidanceY = 102;
  for (const driver of (risk.drivers ?? []).slice(0, 3)) {
    doc.setFont("helvetica", "bold"); doc.setFontSize(9); doc.setTextColor(...GREEN); doc.text(`${driver.driver} · contribution ${Number(driver.score ?? 0).toFixed(1)}`, 15, guidanceY);
    doc.setFont("helvetica", "normal"); doc.setFontSize(8.5); doc.setTextColor(...INK);
    const doLines = doc.splitTextToSize(`DO: ${driver.action}`, 180);
    const avoidLines = doc.splitTextToSize(`AVOID: ${driver.avoid}`, 180);
    doc.text(doLines, 15, guidanceY + 5); doc.text(avoidLines, 15, guidanceY + 5 + doLines.length * 4);
    guidanceY += 13 + (doLines.length + avoidLines.length - 2) * 4;
  }
  drawClimateChart(doc, climate, report.inputs?.season ?? "Kharif", Math.max(guidanceY + 3, 155));
  footer(doc, 2);

  doc.addPage();
  drawHeader(doc, "Field inputs & seasonal outlook", subtitle);
  doc.setFont("helvetica", "bold"); doc.setFontSize(13); doc.setTextColor(...INK); doc.text(`Fertilizer & nutrient plan${plan?.crop ? ` · ${plan.crop}` : ""}`, margin, 40);
  const rates = Object.entries(plan?.fertilizer_kg_ha ?? {});
  autoTable(doc, {
    startY: 44, margin: { left: margin, right: margin },
    head: [["Product", "Rate kg/ha", "N added", "P added", "K added"]],
    body: rates.length ? rates.map(([name, rate]: [string, any]) => [name, rate, plan?.added_soil_NPK?.N ?? 0, plan?.added_soil_NPK?.P ?? 0, plan?.added_soil_NPK?.K ?? 0]) : [["No products selected by optimizer", "0.00", "0.00", "0.00", "0.00"]],
    theme: "grid", styles: { fontSize: 8.5, cellPadding: 2.5, textColor: [48, 62, 51] }, headStyles: { fillColor: GREEN, textColor: 255 },
  });
  const tableY = (doc as any).lastAutoTable.finalY + 11;
  doc.setFont("helvetica", "bold"); doc.setFontSize(12); doc.setTextColor(...INK); doc.text("Season-specific state estimates", margin, tableY);
  autoTable(doc, {
    startY: tableY + 4, margin: { left: margin, right: margin },
    head: [["Year", "Temperature °C", "Season rain mm", "Humidity %", "Climate class"]],
    body: (climate.risk_by_period ?? []).map((row: any) => {
      const year = row.date.slice(0, 4);
      const lookup = (variable: string) => climate.forecast?.find((item: any) => item.date.startsWith(year) && item.variable === variable)?.forecast;
      return [year, Number(lookup("temperature") ?? 0).toFixed(1), Number(lookup("rainfall") ?? 0).toFixed(0), Number(lookup("humidity") ?? 0).toFixed(0), row.risk_class];
    }),
    theme: "grid", styles: { fontSize: 8, cellPadding: 2, textColor: [48, 62, 51] }, headStyles: { fillColor: GREEN, textColor: 255 },
  });
  const advisoryY = Math.min((doc as any).lastAutoTable.finalY + 8, 272);
  doc.setFont("helvetica", "bold"); doc.setFontSize(9); doc.setTextColor(...INK); doc.text("Advisory", margin, advisoryY);
  doc.setFont("helvetica", "normal"); doc.setFontSize(8); doc.setTextColor(...MUTED);
  const advisory = "Yield holdout MAE 1.19 t/ha is a prototype metric. Downside scores blend climate exposure and relative yield-model error; when costs are supplied, simulated financial loss is also included. These are planning indices, not calibrated probabilities. Seasonal values use state historical climate and deterministic simulated year-to-year variability because raw annual records are unavailable; they are not observed seasonal data or a local forecast. Confirm soil values locally and follow current state guidance and product labels. Profit requires user-supplied production costs. Embedded mandi prices are historical snapshots.";
  doc.text(doc.splitTextToSize(advisory, 180), margin, advisoryY + 5);
  footer(doc, 3);
  return doc.output("blob");
}
