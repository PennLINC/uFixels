//! Thin GIFTI helpers: load paired surfaces in RAS+mm and write per-vertex scalars.

use anyhow::{anyhow, Context, Result};
use gifti_rs::cras::{has_cras, read_cras};
use gifti_rs::{
    intent, ArrayData, ArrayIndexOrder, DataArray, DataType, Encoding, Endian, GiftiImage,
};
use std::path::Path;

/// Vertex coordinates in RAS+mm (C_RAS offset applied if present), plus triangle indices.
pub struct LoadedSurface {
    pub vertices: Vec<[f32; 3]>,
    pub triangles: Vec<[u32; 3]>,
}

pub fn read_surface(path: &Path) -> Result<LoadedSurface> {
    let img = gifti_rs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let pointset = img
        .find_array(intent::POINTSET)
        .ok_or_else(|| anyhow!("{}: no POINTSET array", path.display()))?;
    let triangle = img
        .find_array(intent::TRIANGLE)
        .ok_or_else(|| anyhow!("{}: no TRIANGLE array", path.display()))?;

    let coords = match &pointset.data {
        ArrayData::Float32(v) => v,
        other => {
            return Err(anyhow!(
                "{}: POINTSET dtype is {:?}; expected Float32",
                path.display(),
                other.dtype()
            ))
        }
    };
    if coords.len() % 3 != 0 {
        return Err(anyhow!(
            "{}: POINTSET length {} not divisible by 3",
            path.display(),
            coords.len()
        ));
    }

    let cras = if has_cras(pointset) {
        read_cras(pointset)
    } else {
        [0.0, 0.0, 0.0]
    };

    let mut vertices = Vec::with_capacity(coords.len() / 3);
    for c in coords.chunks_exact(3) {
        vertices.push([
            c[0] + cras[0] as f32,
            c[1] + cras[1] as f32,
            c[2] + cras[2] as f32,
        ]);
    }

    let triangles: Vec<[u32; 3]> = match &triangle.data {
        ArrayData::Int32(v) => {
            if v.len() % 3 != 0 {
                return Err(anyhow!(
                    "{}: TRIANGLE length {} not divisible by 3",
                    path.display(),
                    v.len()
                ));
            }
            v.chunks_exact(3)
                .map(|c| [c[0] as u32, c[1] as u32, c[2] as u32])
                .collect()
        }
        other => {
            return Err(anyhow!(
                "{}: TRIANGLE dtype is {:?}; expected Int32",
                path.display(),
                other.dtype()
            ))
        }
    };

    Ok(LoadedSurface {
        vertices,
        triangles,
    })
}

pub fn write_shape(path: &Path, values: &[f32]) -> Result<()> {
    let img = build_shape_image(values);
    gifti_rs::write(&img, path).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

fn build_shape_image(values: &[f32]) -> GiftiImage {
    let n = values.len();
    let array = DataArray {
        intent: intent::SHAPE,
        datatype: DataType::Float32 as i32,
        array_index_order: ArrayIndexOrder::RowMajor,
        dims: vec![n],
        encoding: Encoding::GZipBase64Binary,
        endian: Endian::Little,
        ext_filename: None,
        ext_offset: None,
        coordsys: vec![],
        meta: vec![],
        data: ArrayData::Float32(values.to_vec()),
    };
    GiftiImage {
        version: "1.0".to_string(),
        num_data_arrays: 1,
        meta: vec![],
        label_table: None,
        data_arrays: vec![array],
    }
}
