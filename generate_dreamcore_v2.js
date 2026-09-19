#!/usr/bin/env node
/**
 * Dreamcore v2 Suite: Professional 33x33x33 .CUBE LUTs
 * 
 * Variants:
 * 1. Dreamcore_Anemoia.cube   — Balanced nostalgic, protected skin tones, creamy highlights, film fade
 * 2. Dreamcore_Eternal3PM.cube — Warm, hazy golden-hour overcast, amber glow, soft contrast
 * 3. Dreamcore_Poolrooms.cube  — Cooler fluorescent/cyan liminal tone, deep subtle greens
 */

const fs = require('fs');

function clamp(val, min = 0.0, max = 1.0) {
  return Math.max(min, Math.min(max, val));
}

function rgbToHsl(r, g, b) {
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  let h = 0, s = 0;
  const l = (max + min) / 2;

  if (max !== min) {
    const d = max - min;
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
    switch (max) {
      case r: h = ((g - b) / d + (g < b ? 6 : 0)); break;
      case g: h = ((b - r) / d + 2); break;
      case b: h = ((r - g) / d + 4); break;
    }
    h *= 60;
  }
  return [h, s, l];
}

function hueToRgb(p, q, t) {
  if (t < 0) t += 1;
  if (t > 1) t -= 1;
  if (t < 1/6) return p + (q - p) * 6 * t;
  if (t < 1/2) return q;
  if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
  return p;
}

function hslToRgb(h, s, l) {
  h = (h % 360 + 360) % 360;
  const hNorm = h / 360;
  if (s === 0) return [l, l, l];
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;
  const r = hueToRgb(p, q, hNorm + 1/3);
  const g = hueToRgb(p, q, hNorm);
  const b = hueToRgb(p, q, hNorm - 1/3);
  return [r, g, b];
}

// Multi-point filmic tone curve with gentle lifted black floor and smooth highlight shoulder
function applyFilmicCurve(val, liftedFloor = 0.035, highlightCeiling = 0.96) {
  val = clamp(val);
  const xp = [0.00, 0.20, 0.45, 0.70, 0.88, 1.00];
  const yp = [liftedFloor, 0.18, 0.43, 0.68, 0.85, highlightCeiling];

  for (let i = 0; i < xp.length - 1; i++) {
    if (val >= xp[i] && val <= xp[i + 1]) {
      const t = (val - xp[i]) / (xp[i + 1] - xp[i]);
      const smoothT = t * t * (3 - 2 * t);
      return yp[i] + smoothT * (yp[i + 1] - yp[i]);
    }
  }
  return val;
}

const HUE_CENTERS = {
  red: 0.0,
  orange: 30.0,
  yellow: 60.0,
  green: 120.0,
  aqua: 180.0,
  blue: 240.0,
  purple: 285.0,
  magenta: 325.0
};

function getHueWeights(hue) {
  const h = (hue % 360 + 360) % 360;
  let weights = {};
  let total = 0;
  for (const [name, center] of Object.entries(HUE_CENTERS)) {
    let diff = Math.abs(h - center);
    if (diff > 180) diff = 360 - diff;
    const w = Math.max(0, 1.0 - diff / 35.0);
    weights[name] = w;
    total += w;
  }
  if (total > 0) {
    for (const name in weights) weights[name] /= total;
  }
  return weights;
}

function processPreset(inR, inG, inB, config) {
  let [r, g, b] = [inR, inG, inB];

  // 1. White Balance
  const rGain = 1.0 + (config.temp / 100.0) * 0.08 + (config.tint / 100.0) * 0.04;
  const gGain = 1.0 - (config.tint / 100.0) * 0.04;
  const bGain = 1.0 - (config.temp / 100.0) * 0.08;
  r = clamp(r * rGain);
  g = clamp(g * gGain);
  b = clamp(b * bGain);

  // 2. Exposure & Contrast (Controlled, no blown highlights)
  const luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  
  // Highlights compression & soft shadows
  const hMask = clamp((luma - 0.40) / 0.60);
  const sMask = clamp((0.60 - luma) / 0.60);

  r = clamp(r * (1.0 + (config.highlights / 100.0) * 0.25 * hMask) + (1.0 - r) * (config.shadows / 100.0) * 0.15 * sMask);
  g = clamp(g * (1.0 + (config.highlights / 100.0) * 0.25 * hMask) + (1.0 - g) * (config.shadows / 100.0) * 0.15 * sMask);
  b = clamp(b * (1.0 + (config.highlights / 100.0) * 0.25 * hMask) + (1.0 - b) * (config.shadows / 100.0) * 0.15 * sMask);

  // 3. Filmic Tone Curve
  r = applyFilmicCurve(r, config.liftedFloor, config.highlightCeiling);
  g = applyFilmicCurve(g, config.liftedFloor, config.highlightCeiling);
  b = applyFilmicCurve(b, config.liftedFloor, config.highlightCeiling);

  // 4. Split Toning: Warm cream highlights + subtle cool/cyan shadows
  const curLuma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  const highWeight = Math.pow(curLuma, 1.5);
  const shadWeight = Math.pow(1.0 - curLuma, 1.8);

  // Highlights: warm cream tint
  r = clamp(r + config.highTint[0] * highWeight);
  g = clamp(g + config.highTint[1] * highWeight);
  b = clamp(b + config.highTint[2] * highWeight);

  // Shadows: cool cyan/green tint
  r = clamp(r + config.shadTint[0] * shadWeight);
  g = clamp(g + config.shadTint[1] * shadWeight);
  b = clamp(b + config.shadTint[2] * shadWeight);

  // 5. HSL Selective Color & Skin Tone Protection
  let [h, s, l] = rgbToHsl(r, g, b);
  const weights = getHueWeights(h);

  let deltaH = 0, deltaS = 0, deltaL = 0;
  for (const [name, w] of Object.entries(weights)) {
    if (w > 0 && config.hsl[name]) {
      deltaH += w * config.hsl[name].hue;
      deltaS += w * config.hsl[name].sat;
      deltaL += w * config.hsl[name].lum;
    }
  }

  // Hue adjust
  h = (h + (deltaH / 360.0) * 30.0 + 360.0) % 360.0;

  // Saturation adjust (Restrained & natural)
  let satMult = 1.0 + (config.globalSat / 100.0) + (deltaS / 100.0);
  s = clamp(s * Math.max(0.0, satMult));

  // Luminance adjust
  l = clamp(l * (1.0 + (deltaL / 100.0) * 0.30));

  [r, g, b] = hslToRgb(h, s, l);
  return [clamp(r), clamp(g), clamp(b)];
}

const PRESETS = {
  'Dreamcore_Anemoia': {
    title: 'Dreamcore - Anemoia v2 (Balanced Nostalgic)',
    temp: 3.0,
    tint: -1.0,
    highlights: -28.0,
    shadows: 8.0,
    globalSat: -8.0,
    liftedFloor: 0.035,
    highlightCeiling: 0.96,
    highTint: [0.03, 0.02, -0.015], // Warm cream
    shadTint: [-0.015, 0.02, 0.025], // Subtle cyan/green
    hsl: {
      red:     { hue: 0.0,  sat: -8.0,  lum: 0.0 },
      orange:  { hue: -2.0, sat: -12.0, lum: 5.0 }, // Natural skin protection
      yellow:  { hue: -4.0, sat: -10.0, lum: -3.0 },
      green:   { hue: 10.0, sat: -15.0, lum: 0.0 }, // Non-neon greens
      aqua:    { hue: 5.0,  sat: -5.0,  lum: 0.0 },
      blue:    { hue: 0.0,  sat: -10.0, lum: 5.0 },
      purple:  { hue: 0.0,  sat: 0.0,   lum: 0.0 },
      magenta: { hue: 0.0,  sat: 0.0,   lum: 0.0 }
    }
  },
  'Dreamcore_Eternal3PM': {
    title: 'Dreamcore - Eternal 3PM v2 (Warm Overcast Haze)',
    temp: 8.0,
    tint: 2.0,
    highlights: -35.0,
    shadows: 14.0,
    globalSat: -5.0,
    liftedFloor: 0.045,
    highlightCeiling: 0.95,
    highTint: [0.045, 0.03, -0.02], // Golden amber cream
    shadTint: [0.01, 0.015, -0.01], // Soft warm shadows
    hsl: {
      red:     { hue: 0.0,  sat: -6.0,  lum: 0.0 },
      orange:  { hue: 2.0,  sat: -10.0, lum: 4.0 },
      yellow:  { hue: -2.0, sat: -5.0,  lum: 2.0 },
      green:   { hue: 8.0,  sat: -12.0, lum: 0.0 },
      aqua:    { hue: 0.0,  sat: -8.0,  lum: 2.0 },
      blue:    { hue: -5.0, sat: -12.0, lum: 0.0 },
      purple:  { hue: 0.0,  sat: 0.0,   lum: 0.0 },
      magenta: { hue: 0.0,  sat: 0.0,   lum: 0.0 }
    }
  },
  'Dreamcore_Poolrooms': {
    title: 'Dreamcore - Poolrooms v2 (Cool Fluorescent Liminal)',
    temp: -6.0,
    tint: -3.0,
    highlights: -24.0,
    shadows: 10.0,
    globalSat: -10.0,
    liftedFloor: 0.040,
    highlightCeiling: 0.96,
    highTint: [0.01, 0.03, 0.04],  // Soft pale cyan/white
    shadTint: [-0.02, 0.03, 0.04], // Fluorescent cyan/phosphor
    hsl: {
      red:     { hue: 0.0,  sat: -12.0, lum: 0.0 },
      orange:  { hue: -3.0, sat: -15.0, lum: 6.0 },
      yellow:  { hue: -8.0, sat: -15.0, lum: -5.0 },
      green:   { hue: 15.0, sat: -8.0,  lum: 2.0 },
      aqua:    { hue: 12.0, sat: 8.0,   lum: 4.0 },
      blue:    { hue: 0.0,  sat: -5.0,  lum: 6.0 },
      purple:  { hue: 5.0,  sat: -5.0,  lum: 0.0 },
      magenta: { hue: 0.0,  sat: 0.0,   lum: 0.0 }
    }
  }
};

function generateLutFile(name, config, lutSize = 33) {
  const outputPath = `/root/${name}.cube`;
  let lines = [];
  lines.push(`# ${config.title}`);
  lines.push(`TITLE "${name}"`);
  lines.push(`LUT_3D_SIZE ${lutSize}`);
  lines.push('DOMAIN_MIN 0.0 0.0 0.0');
  lines.push('DOMAIN_MAX 1.0 1.0 1.0');
  lines.push('');

  for (let bIdx = 0; bIdx < lutSize; bIdx++) {
    const bVal = bIdx / (lutSize - 1);
    for (let gIdx = 0; gIdx < lutSize; gIdx++) {
      const gVal = gIdx / (lutSize - 1);
      for (let rIdx = 0; rIdx < lutSize; rIdx++) {
        const rVal = rIdx / (lutSize - 1);
        const [outR, outG, outB] = processPreset(rVal, gVal, bVal, config);
        lines.push(`${outR.toFixed(6)} ${outG.toFixed(6)} ${outB.toFixed(6)}`);
      }
    }
  }

  fs.writeFileSync(outputPath, lines.join('\n') + '\n', 'utf8');
  console.log(`Generated ${outputPath} (${lutSize}x${lutSize}x${lutSize})`);
}

function main() {
  for (const [name, config] of Object.entries(PRESETS)) {
    generateLutFile(name, config, 33);
  }
}

main();
