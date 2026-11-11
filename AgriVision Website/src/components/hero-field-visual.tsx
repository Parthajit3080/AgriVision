import { CloudRain, Leaf, ScanLine, Waves, Wheat } from "lucide-react";

export function HeroFieldVisual() {
  return <div className="hero-field-wrap relative mx-auto w-full max-w-[520px] lg:ml-auto">
    <div className="hero-field relative aspect-[1.12] overflow-hidden rounded-[30px] border border-[#d7f0c746] shadow-[0_28px_100px_#071b1270]">
      <div className="hero-sky absolute inset-0"/>
      <svg className="field-topography absolute inset-0 h-full w-full" viewBox="0 0 600 520" preserveAspectRatio="none" aria-hidden="true">
        <defs>
          <linearGradient id="fieldSlope" x1="0" y1="0" x2="0.8" y2="1"><stop stopColor="#416c45"/><stop offset=".53" stopColor="#214c32"/><stop offset="1" stopColor="#142f23"/></linearGradient>
          <linearGradient id="contourGlow" x1="0" x2="1"><stop stopColor="#d1e9a9" stopOpacity="0"/><stop offset=".47" stopColor="#d1e9a9" stopOpacity=".66"/><stop offset="1" stopColor="#8ce0bd" stopOpacity="0"/></linearGradient>
          <radialGradient id="holoLight"><stop stopColor="#eaf4c9" stopOpacity=".8"/><stop offset="1" stopColor="#eaf4c9" stopOpacity="0"/></radialGradient>
        </defs>
        <path d="M0 292 C110 244 166 282 275 244 S475 217 600 195 V520 H0Z" fill="url(#fieldSlope)"/>
        <g fill="none" stroke="url(#contourGlow)" strokeWidth="1.4" opacity=".86">
          <path d="M-25 349 C80 277 164 311 251 281 S452 257 626 221"/><path d="M-18 374 C92 302 165 336 258 305 S468 286 622 246"/><path d="M-10 400 C102 326 174 360 266 331 S479 312 618 272"/><path d="M0 427 C114 353 184 384 277 357 S486 338 615 300"/><path d="M10 454 C127 380 197 411 289 384 S496 364 611 329"/><path d="M20 482 C140 408 209 439 301 412 S504 393 606 357"/>
        </g>
        <g className="field-ridges" fill="none" stroke="#c0d692" strokeOpacity=".36" strokeWidth="1">
          <path d="M-20 322 C108 261 174 289 276 259 S474 239 622 205"/><path d="M-10 390 C98 336 178 365 275 337 S477 319 615 281"/><path d="M10 459 C132 400 206 428 301 401 S500 386 608 348"/>
        </g>
        <circle cx="395" cy="204" r="74" fill="url(#holoLight)" opacity=".65"/>
        <g className="radar-orbit" fill="none" stroke="#d5efb5" strokeOpacity=".6"><circle cx="400" cy="232" r="60"/><circle cx="400" cy="232" r="87" strokeDasharray="3 8"/></g>
        <path d="M400 232 L445 190" stroke="#d6f2c8" strokeWidth="1.3"/><circle className="radar-pulse" cx="400" cy="232" r="5" fill="#e6f7bd"/>
        <path d="M400 0V520M0 260H600" stroke="#d9f0d2" strokeOpacity=".08" strokeDasharray="3 10"/>
      </svg>
      <div className="scan-beam pointer-events-none absolute inset-x-0 top-0 h-20"/>
      <div className="absolute left-5 top-5 flex items-center gap-2 rounded-full border border-white/20 bg-[#193c2c]/50 px-3 py-2 text-[11px] font-semibold text-[#eff7e5] backdrop-blur-lg"><span className="h-1.5 w-1.5 animate-pulse rounded-full bg-[#d2eea8]"/> FIELD SENSOR · LIVE VIEW</div>
      <div className="absolute right-5 top-5 rounded-xl border border-white/25 bg-white/10 p-2.5 text-[#e7f5d2] backdrop-blur-md"><ScanLine size={17}/></div>
      <div className="absolute left-5 top-[31%] rounded-xl border border-white/20 bg-[#183e2d]/55 px-3 py-2.5 text-white backdrop-blur-lg"><div className="flex items-center gap-2 text-[11px] text-[#d2e2ce]"><Waves size={14}/> Soil signal</div><strong className="mt-1 block text-[14px]">NPK mapped</strong></div>
      <div className="absolute right-5 top-[48%] rounded-xl border border-white/20 bg-[#183e2d]/55 px-3 py-2.5 text-white backdrop-blur-lg"><div className="flex items-center gap-2 text-[11px] text-[#d2e2ce]"><CloudRain size={14}/> Climate layer</div><strong className="mt-1 block text-[14px]">Season-aware</strong></div>
      <div className="absolute bottom-0 inset-x-0 flex items-end justify-between bg-gradient-to-t from-[#10271e]/80 to-transparent px-5 pb-5 pt-16"><span className="text-[11px] font-semibold uppercase tracking-[.15em] text-[#e3edd7]">Field conditions, brought into focus</span><Wheat size={19} className="text-[#d5e3be]"/></div>
      <span className="holo-node node-one"/><span className="holo-node node-two"/><span className="holo-node node-three"/>
    </div>
    <div className="hero-chip chip-weather"><CloudRain size={15}/><span><small>Season outlook</small><b>Risk layer active</b></span></div>
    <div className="hero-chip chip-soil"><Leaf size={15}/><span><small>Soil profile</small><b>4 inputs mapped</b></span></div>
    <div className="holo-caption">AGRI·VISION <span>01 / FIELD</span></div>
  </div>;
}
