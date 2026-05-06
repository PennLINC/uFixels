//! Per-vertex inward normals from paired wm/pial surfaces.
//!
//! By convention `wm[i] - pial[i]` points *into* the brain (pial is outside,
//! wm is inside). The result is unit-length; degenerate pairs return `None`
//! at that index.

const MIN_LEN_MM: f32 = 1.0e-4;

pub fn inward_normals(
    wm: &[[f32; 3]],
    pial: &[[f32; 3]],
) -> Result<Vec<Option<[f32; 3]>>, String> {
    if wm.len() != pial.len() {
        return Err(format!(
            "paired-vertex mismatch: wm has {} vertices, pial has {}",
            wm.len(),
            pial.len()
        ));
    }
    Ok(wm
        .iter()
        .zip(pial.iter())
        .map(|(w, p)| {
            let dx = w[0] - p[0];
            let dy = w[1] - p[1];
            let dz = w[2] - p[2];
            let len = (dx * dx + dy * dy + dz * dz).sqrt();
            if len < MIN_LEN_MM {
                None
            } else {
                let inv = 1.0 / len;
                Some([dx * inv, dy * inv, dz * inv])
            }
        })
        .collect())
}
