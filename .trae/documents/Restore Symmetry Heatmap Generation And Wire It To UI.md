## Goal
Fix “热力图生成失败” by re‑introducing symmetry heatmap generation in the engine and ensuring the UI consumes it reliably, with metric/heatmap computed in the same color space.

## Current Findings
- Engine currently computes `color_symmetry` using Luv but does not return any heatmap image.
- UI tries to render `data.vis_symmetry_heatmap` in Tab 1 and falls back only to a warning, so it shows “热力图生成失败”.

## Plan
### Engine: OmniReport + Analyze
1. Add `vis_symmetry_heatmap: Optional[np.ndarray]` to OmniReport (fenxi/omni_engine.py near the report dataclass).
2. In `analyze(...)`, after computing `img_small` and `img_luv`, compute symmetry heatmap in Luv to match the existing `color_symmetry` metric:
   - Strictly align halves: `cx = process_width // 2`, `left = img_luv[:, :cx]`, `right = img_luv[:, -cx:]`, `right_flipped = flip(right, 1)`.
   - Use float32 to avoid uint8 wraparound: `diff = norm(left_float - right_float, axis=2)`.
   - Build full heatmap: mirror `diff` back to full width with `hstack`, normalize to 0–255, apply `COLORMAP_JET`, convert to RGB.
3. Keep `color_symmetry` score consistent with the same `diff` (mean ΔE in Luv with threshold scaling) and add the new `vis_symmetry_heatmap` to the return object.
4. Ensure returned image is `uint8` RGB shape `(H, W, 3)`.

### UI: fenxi/app.py
5. Keep Tab 1 rendering: show heatmap when `data.vis_symmetry_heatmap` is present; otherwise show fallback message (already implemented).
6. Ensure the top metrics use the same symmetry metric as the heatmap (no changes if we keep Luv in both places).

### Validation
7. Run the app and upload a test image with clear left/right differences to verify:
   - `color_symmetry` score changes meaningfully with symmetric vs non‑symmetric images.
   - Heatmap shows red in asymmetric areas and blue in symmetric areas.
8. Optional: Add guards for edge cases (very small images, width < 2*cx) and confirm performance with `process_width=512`.

## Notes
- Computing heatmap in Luv keeps metric and visualization consistent and avoids user confusion.
- If you prefer RGB symmetry instead, we can switch both metric and heatmap to RGB together (kept as alternative if requested).