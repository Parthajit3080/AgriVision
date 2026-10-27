"use client";

import { useMemo } from "react";
import {
  BarElement, CategoryScale, Chart as ChartJS, Filler, Legend, LinearScale, LineElement, PointElement,
  Tooltip, type ChartOptions,
} from "chart.js";
import { Bar, Line } from "react-chartjs-2";

ChartJS.register(CategoryScale, LinearScale, BarElement, LineElement, PointElement, Tooltip, Legend, Filler);

const grid = "rgba(61, 88, 64, .11)";
const SEASONS = ["Kharif", "Rabi", "Summer"];
const SEASON_COLORS: Record<string, string> = { Kharif: "#4e8058", Rabi: "#bd8950", Summer: "#668c87" };
const shared: ChartOptions<"bar"> = {
  responsive: true, maintainAspectRatio: false,
  plugins: { legend: { display: false }, tooltip: { backgroundColor: "#1f392a", padding: 12, cornerRadius: 9, titleFont: { size: 12 }, bodyFont: { size: 12 } } },
  scales: { x: { grid: { display: false }, ticks: { color: "#65756a", font: { size: 12 } } }, y: { beginAtZero: true, grid: { color: grid }, ticks: { color: "#65756a", font: { size: 11 } }, title: { display: true, text: "t/ha", color: "#65756a" } } },
};

export function CropYieldChart({ rows }: { rows: Array<Record<string, any>> }) {
  const reducedMotion = typeof window !== "undefined"
    && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const data = useMemo(() => ({
    labels: rows.map(row => row.crop),
    datasets: [{ label: "Estimated yield", data: rows.map(row => row.estimated_yield_t_ha), backgroundColor: rows.map((_, i) => i === 0 ? "#315a3c" : "#a8bea0"), borderRadius: 7, borderSkipped: false, maxBarThickness: 26 }],
  }), [rows]);
  const options = useMemo<ChartOptions<"bar">>(() => ({
    ...shared,
    animation: { duration: reducedMotion ? 0 : 650, easing: "easeOutQuart" },
  }), [reducedMotion]);
  if (rows.length === 0) return <div role="status" className="chart-empty grid h-full place-items-center rounded-xl px-4 text-center text-[13px] text-[#78847a]">Crop yield estimates are not available for this report.</div>;
  return <Bar data={data} options={options} aria-label="Crop yield estimate comparison chart" />;
}

function ClimateSeriesChart({ periods, outlook, selectedSeason, field, unit, ariaLabel }: {
  periods: Array<Record<string, any>>; outlook: Array<Record<string, any>>; selectedSeason: string;
  field: "rainfall" | "humidity"; unit: string; ariaLabel: string;
}) {
  const years = [...new Set([
    ...periods.map(row => String(row.date ?? "").slice(0, 4)),
    ...outlook.map(row => String(row.date ?? "").slice(0, 4)),
  ])].filter(year => /^\d{4}$/.test(year)).sort();
  const values = outlook.map(row => Number(row[field])).filter(value => Number.isFinite(value));
  if (!years.length || !values.length) {
    return <div role="status" className="grid h-full place-items-center rounded-xl bg-[#f7f8f4] px-4 text-center text-[13px] text-[#78847a]">Climate outlook data is not available for this selection.</div>;
  }

  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  const padding = Math.max((maxValue - minValue) * .35, maxValue * .025, field === "rainfall" ? 20 : 2);
  const min = Math.max(0, Math.floor(minValue - padding));
  const max = field === "humidity" ? Math.min(100, Math.ceil(maxValue + padding)) : Math.ceil(maxValue + padding);
  const selectedColor = SEASON_COLORS[selectedSeason] ?? SEASON_COLORS.Kharif;
  const axisLabel = field === "rainfall" ? "Seasonal rainfall" : "Seasonal humidity";
  const data = useMemo(() => ({
    labels: years,
    datasets: SEASONS.map(season => {
      const rowsByYear = new Map<string, Record<string, any>>();
      outlook.filter(row => row.season === season).forEach(row => {
        rowsByYear.set(String(row.date ?? "").slice(0, 4), row);
      });
      const isSelected = season.toLowerCase() === selectedSeason.toLowerCase();
      return {
        label: season,
        data: years.map(year => {
          const value = Number(rowsByYear.get(year)?.[field]);
          return Number.isFinite(value) ? value : null;
        }),
        borderColor: SEASON_COLORS[season],
        backgroundColor: SEASON_COLORS[season] + "18",
        borderWidth: isSelected ? 3 : 1.8,
        borderDash: isSelected ? [] : [5, 4],
        pointRadius: isSelected ? 3 : 2,
        pointHoverRadius: 5,
        pointBackgroundColor: SEASON_COLORS[season],
        tension: 0,
        fill: false,
        spanGaps: false,
      };
    }),
  }), [years, outlook, selectedSeason, field]);
  const reducedMotion = typeof window !== "undefined"
    && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const options: ChartOptions<"line"> = {
    responsive: true, maintainAspectRatio: false,
    animation: { duration: reducedMotion ? 0 : 500, easing: "easeOutQuart" },
    plugins: {
      legend: { display: true, position: "bottom", labels: { usePointStyle: true, boxWidth: 8, padding: 15, color: "#536357", font: { size: 11 } } },
      tooltip: {
        backgroundColor: "#203f2e", padding: 11,
        callbacks: {
          label: context => context.dataset.label + ": "
            + Number(context.parsed.y).toLocaleString("en-IN", { maximumFractionDigits: field === "rainfall" ? 0 : 1 }) + unit,
        },
      },
    },
    scales: {
      x: { grid: { display: false }, ticks: { color: "#6d7b70", font: { size: 12 }, maxTicksLimit: 7 } },
      y: {
        min, max, grid: { color: grid },
        ticks: { color: selectedColor, font: { size: 11 }, callback: value => String(value) + unit },
        title: { display: true, text: axisLabel + " · " + unit.trim(), color: selectedColor },
      },
    },
  };
  return <Line data={data} options={options} aria-label={ariaLabel} />;
}

export function RainfallChart({ periods, outlook, selectedSeason }: { periods: Array<Record<string, any>>; outlook: Array<Record<string, any>>; selectedSeason: string }) {
  return <ClimateSeriesChart periods={periods} outlook={outlook} selectedSeason={selectedSeason} field="rainfall" unit=" mm" ariaLabel="Seasonal rainfall estimates comparing Kharif, Rabi, and Summer" />;
}

export function HumidityChart({ periods, outlook, selectedSeason }: { periods: Array<Record<string, any>>; outlook: Array<Record<string, any>>; selectedSeason: string }) {
  return <ClimateSeriesChart periods={periods} outlook={outlook} selectedSeason={selectedSeason} field="humidity" unit="%" ariaLabel="Seasonal humidity estimates comparing Kharif, Rabi, and Summer" />;
}
