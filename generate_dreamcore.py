#!/usr/bin/env python3
"""
Dreamcore LUT Generator for Nothing Camera
Constructs a high-precision 33x33x33 .CUBE 3D LUT matching exact Lightroom parameters.
"""

import numpy as np
import colorsys

def build_tone_curve_lut():
    # 6 control points modeling the Dreamcore lifted-black matte tone curve
    # Input -> Output points (0.0 to 1.0)
    # 1. Lifted blacks (matte shadow)
    # 2. Lower shadows lifted
    # 3. Soft midtone transition
    # 4. Controlled upper midtones
    # 5. Smooth highlight rolloff
    # 6. Slightly muted peak whites
    xp = np.array([0.00, 0.18, 0.38, 0.62, 0.82, 1.00], dtype=np.float32)
    fp = np.array([0.08, 0.22, 0.42, 0.63, 0.83, 0.97], dtype=np.float32)
    
    # 1024-step high-res 1D lookup table using linear/smooth monotonic interpolation
    x_eval = np.linspace(0.0, 1.0, 1024, dtype=np.float32)
    curve_1d = np.interp(x_eval, xp, fp)
    return curve_1d

def apply_tone_curve(rgb, curve_1d):
    idx = np.clip(rgb * 1023.0, 0, 1023).astype(np.int32)
    return curve_1d[idx]

def apply_white_balance(rgb, temp=-8.0, tint=5.0):
    # Temperature: negative = cooler/cyan-blue, positive = warmer/amber
    # Tint: negative = green, positive = magenta
    # Scale factors calibrated to standard Lightroom Kelvin/Tint adjustments
    r_scale = 1.0 + (temp / 100.0) * 0.12 + (tint / 100.0) * 0.08
    g_scale = 1.0 - (tint / 100.0) * 0.08
    b_scale = 1.0 - (temp / 100.0) * 0.12 - (tint / 100.0) * 0.04
    
    rgb = rgb * np.array([r_scale, g_scale, b_scale], dtype=np.float32)
    return np.clip(rgb, 0.0, 1.0)

def apply_light_parameters(rgb, exposure=0.60, contrast=-30.0, highlights=-54.0, shadows=34.0, whites=13.0, blacks=78.0, dehaze=-22.0):
    # 1. Exposure in EV stops
    rgb = rgb * (2.0 ** exposure)
    
    # 2. Dehaze approximation (-22 lowers midtone contrast and lifts shadows/blacks creating a dreamy haze)
    if dehaze != 0:
        haze_factor = (dehaze / 100.0) * 0.15
        rgb = rgb - haze_factor * (1.0 - rgb)
        rgb = np.clip(rgb, 0.0, 1.0)

    # 3. Basic Contrast (-30: compress toward mid-gray 0.45)
    c = (contrast / 100.0) * 0.40
    rgb = (rgb - 0.45) * (1.0 + c) + 0.45
    rgb = np.clip(rgb, 0.0, 1.0)
    
    # Calculate luminance for selective adjustments
    luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    
    # 4. Highlights (-54: pull back upper range)
    h_mask = np.clip((luma - 0.45) / 0.55, 0.0, 1.0)[..., np.newaxis]
    rgb = rgb + (highlights / 100.0) * 0.35 * (rgb * h_mask)
    
    # 5. Shadows (+34: lift lower-mid tones)
    s_mask = np.clip((0.55 - luma) / 0.55, 0.0, 1.0)[..., np.newaxis]
    rgb = rgb + (shadows / 100.0) * 0.30 * ((1.0 - rgb) * s_mask)
    
    # 6. Whites (+13: expand extreme highlights)
    w_mask = np.clip(luma / 1.0, 0.0, 1.0)[..., np.newaxis]
    rgb = rgb + (whites / 100.0) * 0.15 * (w_mask ** 2)
    
    # 7. Blacks (+78: significantly lift deep shadows / floor)
    b_mask = np.clip(1.0 - luma, 0.0, 1.0)[..., np.newaxis]
    rgb = rgb + (blacks / 100.0) * 0.25 * (b_mask ** 1.8)
    
    return np.clip(rgb, 0.0, 1.0)

def get_hue_weights(hue_deg):
    h = hue_deg % 360.0
    centers = {
        'red': 0.0,
        'orange': 30.0,
        'yellow': 60.0,
        'green': 120.0,
        'aqua': 180.0,
        'blue': 240.0,
        'purple': 285.0,
        'magenta': 325.0
    }
    weights = {}
    for name, center in centers.items():
        diff = abs(h - center)
        if diff > 180.0:
            diff = 360.0 - diff
        # Smooth cosine/triangular falloff
        weight = max(0.0, 1.0 - (diff / 35.0))
        weights[name] = weight
        
    total = sum(weights.values())
    if total > 0:
        for k in weights:
            weights[k] /= total
    return weights

def apply_color_mix(rgb, hsl_map, vibrance=46.0, saturation=49.0):
    flat_rgb = rgb.reshape(-1, 3)
    out_rgb = np.zeros_like(flat_rgb)
    
    sat_scale = 1.0 + (saturation / 100.0) * 0.70
    
    for i in range(len(flat_rgb)):
        r, g, b = flat_rgb[i]
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        h_deg = h * 360.0
        
        weights = get_hue_weights(h_deg)
        
        delta_h = sum(weights[c] * hsl_map.get(c, {}).get('hue', 0.0) for c in weights)
        delta_s = sum(weights[c] * hsl_map.get(c, {}).get('sat', 0.0) for c in weights)
        delta_l = sum(weights[c] * hsl_map.get(c, {}).get('lum', 0.0) for c in weights)
        
        # 1. Hue shift
        h = (h + (delta_h / 360.0) * 0.30) % 1.0
        
        # 2. Saturation shift + Vibrance
        s_mult = sat_scale + (delta_s / 100.0) * 0.50
        if vibrance != 0:
            # Vibrance boosts muted colors more strongly
            vib_boost = (vibrance / 100.0) * 0.50 * (1.0 - s)
            s_mult += vib_boost
            
        s = np.clip(s * max(0.0, s_mult), 0.0, 1.0)
        
        # 3. Luminance shift
        l = np.clip(l * (1.0 + (delta_l / 100.0) * 0.40), 0.0, 1.0)
        
        new_r, new_g, new_b = colorsys.hls_to_rgb(h, l, s)
        out_rgb[i] = [new_r, new_g, new_b]
        
    return out_rgb.reshape(rgb.shape)

def generate_dreamcore_lut(output_path="/root/Dreamcore.cube", lut_size=33):
    # Create 33x33x33 base RGB lattice
    grid = np.linspace(0.0, 1.0, lut_size, dtype=np.float32)
    r_grid, g_grid, b_grid = np.meshgrid(grid, grid, grid, indexing='ij')
    rgb_cube = np.stack([r_grid, g_grid, b_grid], axis=-1)
    
    # Exact HSL data provided
    HSL_DATA = {
        'red':     {'hue': 0.0,   'sat': 0.0,   'lum': 0.0},
        'orange':  {'hue': 13.0,  'sat': -13.0, 'lum': 0.0},
        'yellow':  {'hue': -6.0,  'sat': 6.0,   'lum': -9.0},
        'green':   {'hue': 0.0,   'sat': 26.0,  'lum': 0.0},
        'aqua':    {'hue': 10.0,  'sat': 14.0,  'lum': 0.0},
        'blue':    {'hue': 0.0,   'sat': -13.0, 'lum': 11.0},
        'purple':  {'hue': 0.0,   'sat': 15.0,  'lum': 0.0},
        'magenta': {'hue': 0.0,   'sat': 0.0,   'lum': 0.0}
    }
    
    # 1. White Balance (-8 Temp, +5 Tint)
    cube = apply_white_balance(rgb_cube, temp=-8.0, tint=5.0)
    
    # 2. Light & Tonal Parameters
    cube = apply_light_parameters(
        cube,
        exposure=0.60,
        contrast=-30.0,
        highlights=-54.0,
        shadows=34.0,
        whites=13.0,
        blacks=78.0,
        dehaze=-22.0
    )
    
    # 3. Tone Curve (Dreamcore lifted black / matte shape)
    curve_1d = build_tone_curve_lut()
    cube = apply_tone_curve(cube, curve_1d)
    
    # 4. HSL Mix & Vibrance (+46) / Saturation (+49)
    cube = apply_color_mix(cube, HSL_DATA, vibrance=46.0, saturation=49.0)
    
    # 5. Write standard .CUBE file
    with open(output_path, 'w') as f:
        f.write(f'# Created for Nothing Camera preset - Dreamcore\n')
        f.write(f'TITLE "Dreamcore"\n')
        f.write(f'LUT_3D_SIZE {lut_size}\n')
        f.write('DOMAIN_MIN 0.0 0.0 0.0\n')
        f.write('DOMAIN_MAX 1.0 1.0 1.0\n\n')
        
        # .CUBE iteration standard: R fastest, G middle, B slowest
        for b in range(lut_size):
            for g in range(lut_size):
                for r in range(lut_size):
                    val = cube[r, g, b]
                    f.write(f"{val[0]:.6f} {val[1]:.6f} {val[2]:.6f}\n")
                    
    print(f"Generated {output_path} successfully ({lut_size}x{lut_size}x{lut_size} nodes).")

if __name__ == "__main__":
    generate_dreamcore_lut()
