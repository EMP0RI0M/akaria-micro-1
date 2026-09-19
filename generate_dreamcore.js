#!/usr/bin/env node
/**
 * Dreamcore LUT Generator for Nothing Camera
 * Translates exact Lightroom parameters into a standard 33x33x33 .CUBE 3D LUT
 */

const fs = require('fs');

function clamp(val, min = 0.0, max = 1.0) {
  return Math.max(min, Math.min(max, val));
}

// Convert RGB [0..1] to HSL [h in 0..360, s in 0..1, l in 0..1]
function rgbToHsl(r, g, b) {
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  let h = 0;
  let s = 0;
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

// Convert HSL [h in 0..360, s in 0..1, l in 0..1] to RGB [0..1]
function hslToRgb(h, s, l) {
  h = (h % 360 + 360) % 360;
  const hNorm = h / 360;
  if (s === 0) {
    return [l, l, l];
  }
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;
  const r = hueToRgb(p, q, hNorm + 1/3);
  const g = hueToRgb(p, q, hNorm);
  const b = hueToRgb(p, q, hNorm - 1/3);
  return [r, g, b];
}

// 6 Control points for Tone Curve (Lifted blacks, matte look, smooth highlight ceiling)
const curveX = [0.00, 0.18, 0.38, 0.62, 0.82, 1.00];
const curveY = [0.08, 0.22, 0.42, 0.63, 0.83, 0.97];

function evaluateToneCurve(val) {
  val = clamp(val, 0.0, 1.0);
  for (let i = 0; i < curveX.length - 1; i++) {
    if (val >= curveX[i] && val <= curveX[i + 1]) {
      const t = (val - curveX[i]) / (curveX[i + 1] - curveX[i]);
      // Smooth Hermite interpolation between control points
      const smoothT = t * t * (3 - 2 * t);
      return curveY[i] + smoothT * (curveY[i + 1] - curveY[i]);
    }
  }
  return val;
}

// Hue center targets and falloff
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
  h = (hue % 360 + 360) % 360;
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
    for (const name in weights) {
      weights[name] /= total;
    }
  }
  return weights;
}

function processColor(inR, inG, inB) {
  let [r, g, b] = [inR, inG, inB];

  // 1. White Balance (Temp: -8, Tint: +5)
  const temp = -8.0;
  const tint = 5.0;
  const rGain = 1.0 + (temp / 100.0) * 0.12 + (tint / 100.0) * 0.08;
  const gGain = 1.0 - (tint / 100.0) * 0.08;
  const bGain = 1.0 - (temp / 100.0) * 0.12 - (tint / 100.0) * 0.04;
  r = clamp(r * rGain);
  g = clamp(g * gGain);
  b = clamp(b * bGain);

  // 2. Exposure (+0.60 EV)
  const exposure = 0.60;
  const expFactor = Math.pow(2.0, exposure);
  r *= expFactor;
  g *= expFactor;
  b *= expFactor;

  // 3. Dehaze (-22: adds airy haze by lifting floor and softening micro-contrast)
  const dehaze = -22.0;
  const haze = (dehaze / 100.0) * 0.12;
  r = clamp(r - haze * (1.0 - r));
  g = clamp(g - haze * (1.0 - g));
  b = clamp(b - haze * (1.0 - b));

  // 4. Basic Contrast (-30)
  const contrast = -30.0;
  const c = (contrast / 100.0) * 0.40;
  r = clamp((r - 0.45) * (1.0 + c) + 0.45);
  g = clamp((g - 0.45) * (1.0 + c) + 0.45);
  b = clamp((b - 0.45) * (1.0 + c) + 0.45);

  // 5. Highlights (-54), Shadows (+34), Whites (+13), Blacks (+78)
  const luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  const highlights = -54.0;
  const shadows = 34.0;
  const whites = 13.0;
  const blacks = 78.0;

  const hMask = clamp((luma - 0.45) / 0.55);
  const sMask = clamp((0.55 - luma) / 0.55);
  const wMask = clamp(luma);
  const bMask = clamp(1.0 - luma);

  const hAdj = (highlights / 100.0) * 0.35 * hMask;
  const sAdj = (shadows / 100.0) * 0.30 * sMask;
  const wAdj = (whites / 100.0) * 0.15 * (wMask * wMask);
  const bAdj = (blacks / 100.0) * 0.25 * Math.pow(bMask, 1.8);

  r = clamp(r * (1.0 + hAdj) + (1.0 - r) * sAdj + wAdj + bAdj);
  g = clamp(g * (1.0 + hAdj) + (1.0 - g) * sAdj + wAdj + bAdj);
  b = clamp(b * (1.0 + hAdj) + (1.0 - b) * sAdj + wAdj + bAdj);

  // 6. Tone Curve (Dreamcore Matte Curve)
  r = evaluateToneCurve(r);
  g = evaluateToneCurve(g);
  b = evaluateToneCurve(b);

  // 7. HSL Adjustments + Vibrance (+46) & Saturation (+49)
  let [h, s, l] = rgbToHsl(r, g, b);
  const weights = getHueWeights(h);

  const HSL_SETTINGS = {
    red:     { hue: 0.0,   sat: 0.0,   lum: 0.0 },
    orange:  { hue: 13.0,  sat: -13.0, lum: 0.0 },
    yellow:  { hue: -6.0,  sat: 6.0,   lum: -9.0 },
    green:   { hue: 0.0,   sat: 26.0,  lum: 0.0 },
    aqua:    { hue: 10.0,  sat: 14.0,  lum: 0.0 },
    blue:    { hue: 0.0,   sat: -13.0, lum: 11.0 },
    purple:  { hue: 0.0,   sat: 15.0,  lum: 0.0 },
    magenta: { hue: 0.0,   sat: 0.0,   lum: 0.0 }
  };

  let deltaH = 0;
  let deltaS = 0;
  let deltaL = 0;
  for (const [name, w] of Object.entries(weights)) {
    if (w > 0 && HSL_SETTINGS[name]) {
      deltaH += w * HSL_SETTINGS[name].hue;
      deltaS += w * HSL_SETTINGS[name].sat;
      deltaL += w * HSL_SETTINGS[name].lum;
    }
  }

  // Hue shift
  h = (h + (deltaH / 360.0) * 30.0 + 360.0) % 360.0;

  // Saturation with Global Saturation (+49) & Vibrance (+46)
  const saturationGlobal = 49.0;
  const vibrance = 46.0;
  let satMult = 1.0 + (saturationGlobal / 100.0) * 0.70 + (deltaS / 100.0) * 0.50;
  if (vibrance !== 0) {
    satMult += (vibrance / 100.0) * 0.50 * (1.0 - s);
  }
  s = clamp(s * Math.max(0.0, satMult));

  // Luminance shift
  l = clamp(l * (1.0 + (deltaL / 100.0) * 0.40));

  [r, g, b] = hslToRgb(h, s, l);
  return [clamp(r), clamp(g), clamp(b)];
}

function main() {
  const lutSize = 33;
  const outputPath = '/root/Dreamcore.cube';

  let lines = [];
  lines.push('# Created for Nothing Camera preset - Dreamcore');
  lines.push('TITLE "Dreamcore"');
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
        const [outR, outG, outB] = processColor(rVal, gVal, bVal);
        lines.push(`${outR.toFixed(6)} ${outG.toFixed(6)} ${outB.toFixed(6)}`);
      }
    }
  }

  fs.writeFileSync(outputPath, lines.join('\n') + '\n', 'utf8');
  console.log(`Generated ${outputPath} successfully with ${lutSize}x${lutSize}x${lutSize} points.`);
}

main();
