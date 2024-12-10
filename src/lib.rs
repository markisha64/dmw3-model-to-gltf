use base64::{engine::general_purpose::STANDARD, Engine as _};
use binread::{io::Cursor, BinRead};
use dmw3_model::{AnimationFrame, AnimationFrames, AnimationInst, Header, Part};
use dmw3_pack::Packed;
use gltf_json as json;
use image::RgbaImage;
use json::material::{EmissiveFactor, PbrBaseColorFactor, PbrMetallicRoughness, StrengthFactor};
use json::texture::Info;
use json::validation::Checked::Valid;
use json::validation::USize64;
use json::Index;
use rlen::rlen_decode;
use std::mem;
use tim::Tim;

#[derive(Copy, Clone, Debug)]
#[repr(C)]
struct Vertex {
    v: [f32; 3],
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
struct Vec3 {
    values: [f32; 3],
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
struct Vec4 {
    values: [f32; 4],
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
struct Vec4u16 {
    values: [u16; 4],
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
struct Texel {
    v: [f32; 2],
}

struct Animation {
    frames: Vec<Option<u16>>,
}

const MULTIPLIER: f32 = 1.0 / 256.0;
const SCALE_MULTIPLIER: f32 = 1.0 / 4096.0;
const FPS: f32 = 60.0;
const ROTATION: f32 = 1.0 / 1024.0;

type Point = (u32, u32);
type PointFloat = (f64, f64);

fn bytes_to_base64_uri(bytes: Vec<u8>) -> String {
    format!(
        "data:application/octet-stream;base64,{}",
        STANDARD.encode(bytes)
    )
}

fn point_in_triangle(p1: PointFloat, p2: PointFloat, p3: PointFloat, px: f64, py: f64) -> bool {
    let area_orig =
        0.5 * (-p2.1 * p3.0 + p1.1 * (-p2.0 + p3.0) + p1.0 * (p2.1 - p3.1) + p2.0 * p3.1);
    let area1 = 0.5 * (-p2.1 * p3.0 + py * (-p2.0 + p3.0) + px * (p2.1 - p3.1) + p2.0 * p3.1);
    let area2 = 0.5 * (p1.1 * p3.0 + py * (p1.0 - p3.0) + px * (p3.1 - p1.1) + p1.0 * (-p3.1 + py));
    let area3 =
        0.5 * (p1.1 * p2.0 + py * (p3.0 - p1.0) + px * (p1.1 - p2.1) + p1.0 * (-p2.1 + p3.1));

    (area_orig - (area1 + area2 + area3)) < 0.0001
}

fn pixels_in_triangle(p1: PointFloat, p2: PointFloat, p3: PointFloat) -> Vec<Point> {
    // Find bounding box
    let min_x = p1.0.min(p2.0).min(p3.0).floor() as i32;
    let max_x = p1.0.max(p2.0).max(p3.0).ceil() as i32;
    let min_y = p1.1.min(p2.1).min(p3.1).floor() as i32;
    let max_y = p1.1.max(p2.1).max(p3.1).ceil() as i32;

    let mut pixels = Vec::new();

    // Iterate over bounding box
    for x in min_x..=max_x {
        for y in min_y..=max_y {
            let px = x as f64 + 0.5;
            let py = y as f64 + 0.5;
            if point_in_triangle(p1, p2, p3, px, py) {
                pixels.push((px as u32, py as u32));
            }
        }
    }

    pixels
}

fn to_padded_byte_vector<T>(vec: Vec<T>) -> Vec<u8> {
    let byte_length = vec.len() * mem::size_of::<T>();
    let byte_capacity = vec.capacity() * mem::size_of::<T>();
    let alloc = vec.into_boxed_slice();
    let ptr = Box::<[T]>::into_raw(alloc) as *mut u8;
    let mut new_vec = unsafe { Vec::from_raw_parts(ptr, byte_length, byte_capacity) };

    while new_vec.len() % 4 != 0 {
        // pad to multiple of four bytes
        new_vec.push(0);
    }

    new_vec
}

fn clut_idx_to_rgb(clut: &[u8], idx: usize) -> image::Rgba<u8> {
    let normalized = (idx & 0xf) * 2;

    let raw = u16::from_le_bytes([clut[normalized], clut[normalized + 1]]);

    let _stp = (raw >> 15) > 0;

    let r = raw & 0x1f;
    let g = (raw >> 5) & 0x1f;
    let b = (raw >> 10) & 0x1f;

    let r_norm = ((r * 255) / 0x1f) as u8;
    let g_norm = ((g * 255) / 0x1f) as u8;
    let b_norm = ((b * 255) / 0x1f) as u8;

    if r == 0 && g == 0 && b == 0 {
        return image::Rgba([r_norm, g_norm, b_norm, 0]);
    }

    image::Rgba([r_norm, g_norm, b_norm, 255])
}

fn color_tex_tris(
    texture: &mut RgbaImage,
    tim: &Tim,
    clut: &[u8],
    tex1: PointFloat,
    tex2: PointFloat,
    tex3: PointFloat,
) {
    let pixels = pixels_in_triangle(tex1, tex2, tex3);

    for pixel in pixels {
        let pixel_idx_byte = tim.image.bytes[((pixel.0 + pixel.1 * 256) / 2) as usize];

        let pixel_idx = match pixel.0 % 2 {
            0 => pixel_idx_byte & 0xf,
            _ => (pixel_idx_byte >> 4) & 0xf,
        } as usize;

        let color = clut_idx_to_rgb(clut, pixel_idx);

        texture.put_pixel(pixel.0, pixel.1, color);
    }
}

fn grab_frames(animations: Packed) -> anyhow::Result<Vec<Animation>> {
    let returnv: anyhow::Result<Vec<Animation>> = animations
        .iter()
        .map(|idx| -> anyhow::Result<Animation> {
            let animation = &animations.files[idx];

            let mut reader = Cursor::new(&animation);

            let mut instructions = Vec::new();

            for _ in 0..animation.len() / 8 {
                let instruction = AnimationInst::read(&mut reader)?;

                instructions.push(instruction);
            }

            let mut frames = Vec::new();

            for instruction in instructions.iter() {
                if instruction.t == 0x7fff {
                    break;
                }

                if instruction.len == 0 {
                    frames.push(Some(instruction.c));

                    break;
                }

                if instruction.d == 0 {
                    for _ in 1..instruction.len {
                        frames.push(None);
                    }
                } else {
                    for i in 0..instruction.len {
                        frames.push(Some(instruction.c + i));
                    }
                }
            }

            Ok(Animation { frames })
        })
        .collect();

    Ok(returnv?)
}

fn find_or_interpolate_frame<'a>(
    files: &Vec<Vec<Vec<AnimationFrame>>>,
    animation: usize,
    frame_id: u16,
    index: usize,
) -> anyhow::Result<AnimationFrame> {
    let (idx, best_match) = files[animation][index]
        .iter()
        .enumerate()
        .find(|(_idx, frame)| frame.id as i16 > frame_id as i16)
        .unwrap_or((
            files[animation][index].len() - 1,
            files[animation][index].last().unwrap(),
        ));

    if best_match.id == frame_id || idx == 0 {
        return Ok(best_match.clone());
    }

    let previous_frame = &files[animation][index][idx - 1];

    let m = (frame_id - previous_frame.id) as i32;
    let d = ((best_match.id - previous_frame.id) as i32) * 4096;

    let delta = AnimationFrame {
        vx: best_match.vx - previous_frame.vx,
        vy: best_match.vy - previous_frame.vy,
        vz: best_match.vz - previous_frame.vz,
        id: best_match.id - previous_frame.id,
    };

    Ok(AnimationFrame {
        vx: previous_frame.vx + ((m * (delta.vx as i32)) / d) as i16,
        vy: previous_frame.vy + ((m * (delta.vy as i32)) / d) as i16,
        vz: previous_frame.vz + ((m * (delta.vz as i32)) / d) as i16,
        id: previous_frame.id,
    })
}

fn to_quaternion(frame: &AnimationFrame) -> Vec4 {
    Vec4 {
        values: [
            frame.vx as f32 * ROTATION,
            frame.vy as f32 * ROTATION,
            frame.vz as f32 * ROTATION,
            1.0,
        ],
    }
}

pub fn create_gltf(header: &Header, unpacked: &Packed) -> anyhow::Result<Vec<u8>> {
    let mut root = json::Root::default();

    let animations = grab_frames(Packed::from(unpacked.files[0].clone()))?;

    let animation_idxs: Vec<u32> = header.parts.iter().map(|x| x.animation).collect();

    let animation_files: Vec<Vec<Vec<AnimationFrame>>> = unpacked
        .iter()
        .map(|idx| {
            if !animation_idxs.contains(&(idx as u32)) {
                return Vec::new();
            }

            let upkg = Packed::from(unpacked.files[idx].clone());

            let mut returnv: Vec<Vec<AnimationFrame>> = Vec::new();

            for fidx in 0..3 {
                let mut reader = Cursor::new(upkg.files[fidx].clone());

                let mut res = Vec::new();

                loop {
                    match AnimationFrame::read(&mut reader) {
                        Ok(frame) => res.push(frame),
                        Err(_) => break,
                    }
                }

                returnv.push(res);
            }

            returnv
        })
        .collect();

    let texture_packed_raw = unpacked.files[header.texture_offset as usize].clone();

    let texture_packed = Packed::from(texture_packed_raw.clone());

    let texture_raw = match rlen_decode(&texture_packed.files[0]) {
        Ok(file) => file,
        Err(_) => Vec::from(texture_packed_raw),
    };

    let texture_tim = Tim::from(texture_raw);

    let mut texture_png: RgbaImage = RgbaImage::new(256, 256);

    let mut nodes = Vec::new();

    let mut parts: Vec<Part> = Vec::new();
    let mut vertices: Vec<Vertex> = Vec::new();
    let mut tex_coords: Vec<Texel> = Vec::new();
    let mut joints: Vec<Vec4u16> = Vec::new();
    let mut weights: Vec<Vec4> = Vec::new();

    let vertices_ref = &mut vertices;
    let tex_coords_ref = &mut tex_coords;

    let mut part_children: Vec<Vec<usize>> = std::iter::repeat(0)
        .take(header.parts.len() + 1)
        .map(|_| Vec::new())
        .collect();

    for (idx, part) in header.parts.iter().enumerate() {
        part_children[part.parent_index as usize].push(idx + 1);
    }

    nodes.push(
        root.push(json::Node {
            mesh: None,
            children: Some(
                part_children[0]
                    .iter()
                    .map(|pc| Index::new(*pc as u32))
                    .collect(),
            ),
            ..Default::default()
        }),
    );

    for i in 1..header.parts.len() + 1 {
        let node = root.push(json::Node {
            mesh: None,
            children: Some(
                part_children[i]
                    .iter()
                    .map(|pc| Index::new(*pc as u32))
                    .collect(),
            ),
            ..Default::default()
        });

        nodes.push(node);
    }

    for (idx, part) in header.parts.iter().enumerate() {
        let vfile = &unpacked.files[part.geometry as usize];

        let verts_offset = u32::from_le_bytes([vfile[0], vfile[1], vfile[2], vfile[3]]) as usize;

        let verts = &vfile[verts_offset..];

        let vert_count = u16::from_le_bytes([verts[0], verts[1]]) as usize;

        let vertices_sparse: Vec<Vertex> = verts[6..]
            .chunks_exact(6)
            .take(vert_count)
            .map(|chunk| {
                let x: f32 = i16::from_le_bytes([chunk[0], chunk[1]]).into();
                let y: f32 = i16::from_le_bytes([chunk[2], chunk[3]]).into();
                let z: f32 = i16::from_le_bytes([chunk[4], chunk[5]]).into();

                return Vertex {
                    v: [x * MULTIPLIER, y * MULTIPLIER, z * MULTIPLIER],
                };
            })
            .collect();

        let faces_offset = u32::from_le_bytes([vfile[8], vfile[9], vfile[10], vfile[11]]) as usize;

        let face_file = &vfile[faces_offset..];

        let old_vertex_len = vertices_ref.len();
        let old_tex_len = tex_coords_ref.len();

        let mut tex_origin_1 = 0;
        let mut tex_origin_2 = 0;

        let mut clut_x = 0;
        let mut clut_y = 0;

        let mut clut = &texture_tim.image.bytes[2 * (64 * clut_y + clut_x)..];

        let mut is_quad = 0;
        let mut has_textures = 0;
        // let mut s3 = 0;
        // let mut s4 = 0;
        let mut has_normals = 0;
        // let mut s6 = 0;
        // let mut s7 = 0;
        let mut face = [
            Vertex { v: [0.0, 0.0, 0.0] },
            Vertex { v: [0.0, 0.0, 0.0] },
            Vertex { v: [0.0, 0.0, 0.0] },
            Vertex { v: [0.0, 0.0, 0.0] },
        ];

        let mut face_tex = [
            Texel { v: [0.0, 0.0] },
            Texel { v: [0.0, 0.0] },
            Texel { v: [0.0, 0.0] },
            Texel { v: [0.0, 0.0] },
        ];

        let mut i = 0;

        while face_file[i] != 0xff {
            let ctype = face_file[i] >> 4;
            let t = (face_file[i] & 0xf) as usize;

            match ctype {
                0 => {
                    if t == 1 {
                        tex_origin_1 = face_file[i + 1] as u32 + 256 * (face_file[i + 2] as u32);
                        tex_origin_2 = face_file[i + 3] as u32;

                        clut_x = face_file[i + 4] as usize;
                        clut_y = face_file[i + 5] as usize;

                        clut = &texture_tim.image.bytes[2 * (64 * clut_y + clut_x)..];

                        i += 7;
                    } else if t == 0 {
                        if is_quad > 0 {
                            face[0] = vertices_sparse[face_file[i + 1] as usize];
                            face[1] = vertices_sparse[face_file[i + 2] as usize];
                            face[2] = vertices_sparse[face_file[i + 3] as usize];
                            face[3] = vertices_sparse[face_file[i + 4] as usize];
                        } else {
                            face[0] = vertices_sparse[face_file[i + 1] as usize];
                            face[1] = vertices_sparse[face_file[i + 2] as usize];
                            face[2] = vertices_sparse[face_file[i + 3] as usize];
                        }

                        if has_textures > 0 {
                            let offset = match has_normals {
                                0 => is_quad + 4 + i,
                                _ => is_quad * 2 + 7 + i,
                            };

                            if is_quad > 0 {
                                let tex1 = (
                                    (face_file[offset] as u32 + tex_origin_1) % 256,
                                    (face_file[offset + 1] as u32 + tex_origin_2) % 256,
                                );
                                let tex2 = (
                                    (face_file[offset + 2] as u32 + tex_origin_1) % 256,
                                    (face_file[offset + 3] as u32 + tex_origin_2) % 256,
                                );
                                let tex3 = (
                                    (face_file[offset + 4] as u32 + tex_origin_1) % 256,
                                    (face_file[offset + 5] as u32 + tex_origin_2) % 256,
                                );
                                let tex4 = (
                                    (face_file[offset + 6] as u32 + tex_origin_1) % 256,
                                    (face_file[offset + 7] as u32 + tex_origin_2) % 256,
                                );

                                face_tex[0] = Texel {
                                    v: [(tex1.0 as f32) / 255.0, (tex1.1 as f32) / 255.0],
                                };
                                face_tex[1] = Texel {
                                    v: [(tex2.0 as f32) / 255.0, (tex2.1 as f32) / 255.0],
                                };
                                face_tex[2] = Texel {
                                    v: [(tex3.0 as f32) / 255.0, (tex3.1 as f32) / 255.0],
                                };
                                face_tex[3] = Texel {
                                    v: [(tex4.0 as f32) / 255.0, (tex4.1 as f32) / 255.0],
                                };
                            } else {
                                let tex1 = (
                                    (face_file[offset] as u32 + tex_origin_1) % 256,
                                    (face_file[offset + 1] as u32 + tex_origin_2) % 256,
                                );
                                let tex2 = (
                                    (face_file[offset + 2] as u32 + tex_origin_1) % 256,
                                    (face_file[offset + 3] as u32 + tex_origin_2) % 256,
                                );
                                let tex3 = (
                                    (face_file[offset + 4] as u32 + tex_origin_1) % 256,
                                    (face_file[offset + 5] as u32 + tex_origin_2) % 256,
                                );

                                face_tex[0] = Texel {
                                    v: [(tex1.0 as f32) / 255.0, (tex1.1 as f32) / 255.0],
                                };
                                face_tex[1] = Texel {
                                    v: [(tex2.0 as f32) / 255.0, (tex2.1 as f32) / 255.0],
                                };
                                face_tex[2] = Texel {
                                    v: [(tex3.0 as f32) / 255.0, (tex3.1 as f32) / 255.0],
                                };
                            }
                        }

                        if is_quad > 0 {
                            vertices_ref.push(face[0]);
                            vertices_ref.push(face[1]);
                            vertices_ref.push(face[2]);

                            vertices_ref.push(face[1]);
                            vertices_ref.push(face[2]);
                            vertices_ref.push(face[3]);

                            tex_coords_ref.push(face_tex[0]);
                            tex_coords_ref.push(face_tex[1]);
                            tex_coords_ref.push(face_tex[2]);

                            tex_coords_ref.push(face_tex[1]);
                            tex_coords_ref.push(face_tex[2]);
                            tex_coords_ref.push(face_tex[3]);

                            color_tex_tris(
                                &mut texture_png,
                                &texture_tim,
                                clut,
                                (
                                    (face_tex[0].v[0] * 255.0) as f64,
                                    (face_tex[0].v[1] * 255.0) as f64,
                                ),
                                (
                                    (face_tex[1].v[0] * 255.0) as f64,
                                    (face_tex[1].v[1] * 255.0) as f64,
                                ),
                                (
                                    (face_tex[2].v[0] * 255.0) as f64,
                                    (face_tex[2].v[1] * 255.0) as f64,
                                ),
                            );

                            color_tex_tris(
                                &mut texture_png,
                                &texture_tim,
                                clut,
                                (
                                    (face_tex[1].v[0] * 255.0) as f64,
                                    (face_tex[1].v[1] * 255.0) as f64,
                                ),
                                (
                                    (face_tex[2].v[0] * 255.0) as f64,
                                    (face_tex[2].v[1] * 255.0) as f64,
                                ),
                                (
                                    (face_tex[3].v[0] * 255.0) as f64,
                                    (face_tex[3].v[1] * 255.0) as f64,
                                ),
                            );
                        } else {
                            vertices_ref.push(face[0]);
                            vertices_ref.push(face[1]);
                            vertices_ref.push(face[2]);

                            tex_coords_ref.push(face_tex[0]);
                            tex_coords_ref.push(face_tex[1]);
                            tex_coords_ref.push(face_tex[2]);

                            color_tex_tris(
                                &mut texture_png,
                                &texture_tim,
                                clut,
                                (
                                    (face_tex[0].v[0] * 255.0) as f64,
                                    (face_tex[0].v[1] * 255.0) as f64,
                                ),
                                (
                                    (face_tex[1].v[0] * 255.0) as f64,
                                    (face_tex[1].v[1] * 255.0) as f64,
                                ),
                                (
                                    (face_tex[2].v[0] * 255.0) as f64,
                                    (face_tex[2].v[1] * 255.0) as f64,
                                ),
                            );
                        }

                        let face_length_in_bytes = is_quad + 3;
                        let j = i;
                        i = j + face_length_in_bytes;

                        if has_normals != 0 {
                            i = j + face_length_in_bytes * 2;
                        }

                        if has_textures != 0 {
                            i = i + face_length_in_bytes * 2;
                        }

                        i += 1;
                    } else if t < 6 {
                        i += 4;
                    }
                }
                _ => {
                    match ctype {
                        8 => is_quad = t,
                        9 => has_textures = t,
                        0xc => has_normals = t,
                        _ => {}
                    }

                    i += 1;
                }
            }
        }

        let vert_len = vertices_ref.len() - old_vertex_len;

        joints.extend(
            std::iter::repeat(Vec4u16 {
                values: [
                    (idx + 1) as u16,
                    (idx + 1) as u16,
                    (idx + 1) as u16,
                    (idx + 1) as u16,
                ],
            })
            .take(vert_len),
        );
        weights.extend(
            std::iter::repeat(Vec4 {
                values: [1.0, 1.0, 1.0, 1.0],
            })
            .take(vert_len),
        );

        parts.push(Part {
            _animation: part.animation as usize,
            vert_offset: old_vertex_len,
            vert_len,
            tex_offset: old_tex_len,
            tex_len: tex_coords_ref.len() - old_tex_len,
        });
    }

    let tex_coords_bin = to_padded_byte_vector(tex_coords.clone());
    let texture_buffer_length = tex_coords.len() * mem::size_of::<Texel>();
    let texture_buffer = root.push(json::Buffer {
        byte_length: USize64::from(texture_buffer_length),
        extensions: Default::default(),
        extras: Default::default(),
        uri: Some(bytes_to_base64_uri(tex_coords_bin)),
    });

    let texture_view = root.push(json::buffer::View {
        buffer: texture_buffer,
        byte_length: USize64::from(texture_buffer_length),
        byte_offset: None,
        byte_stride: Some(json::buffer::Stride(mem::size_of::<Texel>())),
        extensions: Default::default(),
        extras: Default::default(),
        target: Some(Valid(json::buffer::Target::ArrayBuffer)),
    });

    let vertex_bin = to_padded_byte_vector(vertices.clone());
    let vertex_buffer_length = vertices.len() * mem::size_of::<Vertex>();
    let vertex_buffer = root.push(json::Buffer {
        byte_length: USize64::from(vertex_buffer_length),
        extensions: Default::default(),
        extras: Default::default(),
        uri: Some(bytes_to_base64_uri(vertex_bin)),
    });

    let vertex_view = root.push(json::buffer::View {
        buffer: vertex_buffer,
        byte_length: USize64::from(vertex_buffer_length),
        byte_offset: None,
        byte_stride: Some(json::buffer::Stride(mem::size_of::<Vertex>())),
        extensions: Default::default(),
        extras: Default::default(),
        target: Some(Valid(json::buffer::Target::ArrayBuffer)),
    });

    let joints_bin = to_padded_byte_vector(joints.clone());
    let joints_buffer_length = joints.len() * mem::size_of::<Vec4u16>();
    let joints_buffer = root.push(json::Buffer {
        byte_length: USize64::from(joints_buffer_length),
        uri: Some(bytes_to_base64_uri(joints_bin)),
        extensions: Default::default(),
        extras: Default::default(),
    });

    let joints_view = root.push(json::buffer::View {
        buffer: joints_buffer,
        byte_length: USize64::from(joints_buffer_length),
        byte_offset: None,
        byte_stride: Some(json::buffer::Stride(mem::size_of::<Vec4u16>())),
        extensions: Default::default(),
        extras: Default::default(),
        target: Some(Valid(json::buffer::Target::ArrayBuffer)),
    });

    let weights_bin = to_padded_byte_vector(weights.clone());
    let weights_buffer_length = weights.len() * mem::size_of::<Vec4>();
    let weights_buffer = root.push(json::Buffer {
        byte_length: USize64::from(weights_buffer_length),
        uri: Some(bytes_to_base64_uri(weights_bin)),
        extensions: Default::default(),
        extras: Default::default(),
    });

    let weights_view = root.push(json::buffer::View {
        buffer: weights_buffer,
        byte_length: USize64::from(weights_buffer_length),
        byte_offset: None,
        byte_stride: Some(json::buffer::Stride(mem::size_of::<Vec4>())),
        extensions: Default::default(),
        extras: Default::default(),
        target: Some(Valid(json::buffer::Target::ArrayBuffer)),
    });

    let skin = root.push(json::Skin {
        inverse_bind_matrices: None,
        joints: (0..parts.len() + 1)
            .map(|pc| Index::new(pc as u32))
            .collect(),
        extensions: Default::default(),
        extras: Default::default(),
        skeleton: None,
    });

    let mut image_bin = Vec::new();
    let mut image_cursor = Cursor::new(&mut image_bin);
    texture_png.write_to(&mut image_cursor, image::ImageFormat::Png)?;

    let image = root.push(json::Image {
        buffer_view: None,
        mime_type: None,
        uri: Some(bytes_to_base64_uri(image_bin)),
        extensions: Default::default(),
        extras: Default::default(),
    });

    let sampler = root.push(json::texture::Sampler {
        mag_filter: Some(Valid(json::texture::MagFilter::Nearest)),
        min_filter: Some(Valid(json::texture::MinFilter::Nearest)),
        wrap_s: Valid(json::texture::WrappingMode::Repeat),
        wrap_t: Valid(json::texture::WrappingMode::Repeat),
        extensions: Default::default(),
        extras: Default::default(),
    });

    let texture = root.push(json::Texture {
        sampler: Some(sampler),
        source: image,
        extensions: Default::default(),
        extras: Default::default(),
    });

    let pbr = PbrMetallicRoughness {
        base_color_texture: Some(Info {
            index: texture,
            tex_coord: 0,
            extensions: Default::default(),
            extras: Default::default(),
        }),
        base_color_factor: PbrBaseColorFactor {
            0: [1.0, 1.0, 1.0, 1.0],
        },
        metallic_factor: StrengthFactor(0.0),
        metallic_roughness_texture: None,
        roughness_factor: StrengthFactor(0.0),
        extensions: Default::default(),
        extras: Default::default(),
    };

    let material = root.push(json::Material {
        alpha_cutoff: None,
        alpha_mode: Valid(json::material::AlphaMode::Mask),
        double_sided: true,
        pbr_metallic_roughness: pbr,
        extensions: Default::default(),
        extras: Default::default(),
        normal_texture: None,
        emissive_texture: None,
        occlusion_texture: None,
        emissive_factor: EmissiveFactor([0.0, 0.0, 0.0]),
    });

    for part in parts.iter() {
        let positions = root.push(json::Accessor {
            buffer_view: Some(vertex_view),
            byte_offset: Some(USize64::from(part.vert_offset * mem::size_of::<Vertex>())),
            count: USize64::from(part.vert_len),
            component_type: Valid(json::accessor::GenericComponentType(
                json::accessor::ComponentType::F32,
            )),
            extensions: Default::default(),
            extras: Default::default(),
            type_: Valid(json::accessor::Type::Vec3),
            min: None,
            max: None,
            normalized: false,
            sparse: None,
        });

        let tex_coords_accessor = root.push(json::Accessor {
            buffer_view: Some(texture_view),
            byte_offset: Some(USize64::from(part.tex_offset * mem::size_of::<Texel>())),
            count: USize64::from(part.tex_len),
            component_type: Valid(json::accessor::GenericComponentType(
                json::accessor::ComponentType::F32,
            )),
            extensions: Default::default(),
            extras: Default::default(),
            type_: Valid(json::accessor::Type::Vec2),
            min: None,
            max: None,
            normalized: false,
            sparse: None,
        });

        let joints_accessor = root.push(json::Accessor {
            buffer_view: Some(joints_view),
            byte_offset: Some(USize64::from(part.vert_offset * mem::size_of::<Vec4u16>())),
            count: USize64::from(part.vert_len),
            component_type: Valid(json::accessor::GenericComponentType(
                json::accessor::ComponentType::U16,
            )),
            extensions: Default::default(),
            extras: Default::default(),
            type_: Valid(json::accessor::Type::Vec4),
            min: None,
            max: None,
            normalized: false,
            sparse: None,
        });

        let weights_accessor = root.push(json::Accessor {
            buffer_view: Some(weights_view),
            byte_offset: Some(USize64::from(part.vert_offset * mem::size_of::<Vec4>())),
            count: USize64::from(part.vert_len),
            component_type: Valid(json::accessor::GenericComponentType(
                json::accessor::ComponentType::F32,
            )),
            extensions: Default::default(),
            extras: Default::default(),
            type_: Valid(json::accessor::Type::Vec4),
            min: None,
            max: None,
            normalized: false,
            sparse: None,
        });

        let primitive = json::mesh::Primitive {
            attributes: {
                let mut map = std::collections::BTreeMap::new();
                map.insert(Valid(json::mesh::Semantic::Positions), positions);
                map.insert(
                    Valid(json::mesh::Semantic::TexCoords(0)),
                    tex_coords_accessor,
                );
                map.insert(Valid(json::mesh::Semantic::Joints(0)), joints_accessor);
                map.insert(Valid(json::mesh::Semantic::Weights(0)), weights_accessor);
                map
            },
            extensions: Default::default(),
            extras: Default::default(),
            indices: None,
            material: Some(material),
            mode: Valid(json::mesh::Mode::Triangles),
            targets: None,
        };

        let mesh = Some(root.push(json::Mesh {
            extensions: Default::default(),
            extras: Default::default(),
            primitives: vec![primitive],
            weights: None,
        }));

        let node = root.push(json::Node {
            mesh,
            skin: Some(skin),
            ..Default::default()
        });

        nodes.push(node);
    }

    let mut animations_parsed: Vec<Vec<Vec<Option<AnimationFrames>>>> = Vec::new();

    for anim in animations.iter() {
        animations_parsed.push(
            header
                .parts
                .iter()
                .map(|x| -> anyhow::Result<Vec<Option<AnimationFrames>>> {
                    anim.frames
                        .iter()
                        .map(|frame| -> anyhow::Result<Option<AnimationFrames>> {
                            if let Some(frame_id) = frame {
                                let translation = find_or_interpolate_frame(
                                    &animation_files,
                                    x.animation as usize,
                                    *frame_id,
                                    0,
                                )?;

                                let rotation = find_or_interpolate_frame(
                                    &animation_files,
                                    x.animation as usize,
                                    *frame_id,
                                    1,
                                )?;

                                let scale = find_or_interpolate_frame(
                                    &animation_files,
                                    x.animation as usize,
                                    *frame_id,
                                    2,
                                )?;

                                return Ok(Some(AnimationFrames {
                                    translation,
                                    rotation,
                                    scale,
                                }));
                            }

                            Ok(None)
                        })
                        .collect::<anyhow::Result<Vec<Option<AnimationFrames>>>>()
                })
                .collect::<anyhow::Result<Vec<Vec<Option<AnimationFrames>>>>>()?,
        );
    }

    let mut animation_timestamps: Vec<f32> = Vec::new();
    let mut animation_translations: Vec<Vec3> = Vec::new();
    let mut animation_rotation: Vec<Vec4> = Vec::new();
    let mut animation_scale: Vec<Vec3> = Vec::new();

    for animation in animations_parsed {
        for part in animation {
            for (idx, v) in part.iter().enumerate() {
                if v.is_some() {
                    animation_timestamps.push(idx as f32 * (1.0 / FPS));
                }
            }

            animation_translations.extend(
                part.iter()
                    .filter_map(|x| match x {
                        Some(s) => Some(s),
                        None => None,
                    })
                    .map(|x| Vec3 {
                        values: [
                            x.translation.vx as f32 * MULTIPLIER,
                            x.translation.vy as f32 * MULTIPLIER,
                            x.translation.vz as f32 * MULTIPLIER,
                        ],
                    }),
            );

            animation_rotation.extend(
                part.iter()
                    .filter_map(|x| match x {
                        Some(s) => Some(s),
                        None => None,
                    })
                    .map(|x| to_quaternion(&x.rotation)),
            );

            animation_scale.extend(
                part.iter()
                    .filter_map(|x| match x {
                        Some(s) => Some(s),
                        None => None,
                    })
                    .map(|x| Vec3 {
                        values: [
                            x.scale.vx as f32 * SCALE_MULTIPLIER,
                            x.scale.vy as f32 * SCALE_MULTIPLIER,
                            x.scale.vz as f32 * SCALE_MULTIPLIER,
                        ],
                    }),
            );
        }
    }

    let timestamps_bin = to_padded_byte_vector(animation_timestamps.clone());
    let timestamp_buffer_length = animation_timestamps.len() * mem::size_of::<f32>();
    let timestamp_buffer = root.push(json::Buffer {
        byte_length: USize64::from(timestamp_buffer_length),
        uri: Some(bytes_to_base64_uri(timestamps_bin)),
        extensions: Default::default(),
        extras: Default::default(),
    });

    let timestamp_buffer_view = root.push(json::buffer::View {
        buffer: timestamp_buffer,
        byte_length: USize64::from(timestamp_buffer_length),
        byte_offset: None,
        byte_stride: Some(json::buffer::Stride(mem::size_of::<f32>())),
        extensions: Default::default(),
        extras: Default::default(),
        target: Some(Valid(json::buffer::Target::ArrayBuffer)),
    });

    let translations_bin = to_padded_byte_vector(animation_translations.clone());
    let translations_buffer_length = animation_translations.len() * mem::size_of::<Vec3>();
    let translations_buffer = root.push(json::Buffer {
        byte_length: USize64::from(translations_buffer_length),
        uri: Some(bytes_to_base64_uri(translations_bin)),
        extensions: Default::default(),
        extras: Default::default(),
    });

    let translations_buffer_view = root.push(json::buffer::View {
        buffer: translations_buffer,
        byte_length: USize64::from(translations_buffer_length),
        byte_offset: None,
        byte_stride: Some(json::buffer::Stride(mem::size_of::<Vec3>())),
        extensions: Default::default(),
        extras: Default::default(),
        target: Some(Valid(json::buffer::Target::ArrayBuffer)),
    });

    let rotation_bin = to_padded_byte_vector(animation_rotation.clone());
    let rotation_buffer_length = animation_rotation.len() * mem::size_of::<Vec4>();
    let rotation_buffer = root.push(json::Buffer {
        byte_length: USize64::from(rotation_buffer_length),
        uri: Some(bytes_to_base64_uri(rotation_bin)),
        extensions: Default::default(),
        extras: Default::default(),
    });

    let rotation_buffer_view = root.push(json::buffer::View {
        buffer: rotation_buffer,
        byte_length: USize64::from(rotation_buffer_length),
        byte_offset: None,
        byte_stride: Some(json::buffer::Stride(mem::size_of::<Vec4>())),
        extensions: Default::default(),
        extras: Default::default(),
        target: Some(Valid(json::buffer::Target::ArrayBuffer)),
    });

    let scale_bin = to_padded_byte_vector(animation_scale.clone());
    let scale_buffer_length = animation_scale.len() * mem::size_of::<Vec3>();
    let scale_buffer = root.push(json::Buffer {
        byte_length: USize64::from(scale_buffer_length),
        uri: Some(bytes_to_base64_uri(scale_bin)),
        extensions: Default::default(),
        extras: Default::default(),
    });

    let scale_buffer_view = root.push(json::buffer::View {
        buffer: scale_buffer,
        byte_length: USize64::from(scale_buffer_length),
        byte_offset: None,
        byte_stride: Some(json::buffer::Stride(mem::size_of::<Vec3>())),
        extensions: Default::default(),
        extras: Default::default(),
        target: Some(Valid(json::buffer::Target::ArrayBuffer)),
    });

    let mut part_iter = 0;
    for animation in animations.iter() {
        let mut samplers = Vec::new();
        let mut channels = Vec::new();
        for i in 1..parts.len() + 1 {
            let anim_frames_len = animation.frames.iter().filter(|x| x.is_some()).count();

            let timestamp_accessor = root.push(json::Accessor {
                buffer_view: Some(timestamp_buffer_view),
                byte_offset: Some(USize64::from(part_iter * mem::size_of::<f32>())),
                count: USize64::from(anim_frames_len),
                component_type: Valid(json::accessor::GenericComponentType(
                    json::accessor::ComponentType::F32,
                )),
                type_: Valid(json::accessor::Type::Scalar),
                min: None,
                max: None,
                normalized: false,
                sparse: None,
                extensions: Default::default(),
                extras: Default::default(),
            });

            let translation_accessor = root.push(json::Accessor {
                buffer_view: Some(translations_buffer_view),
                byte_offset: Some(USize64::from(part_iter * mem::size_of::<Vec3>())),
                count: USize64::from(anim_frames_len),
                component_type: Valid(json::accessor::GenericComponentType(
                    json::accessor::ComponentType::F32,
                )),
                type_: Valid(json::accessor::Type::Vec3),
                min: None,
                max: None,
                normalized: false,
                sparse: None,
                extensions: Default::default(),
                extras: Default::default(),
            });

            let translation_sampler = json::animation::Sampler {
                input: timestamp_accessor,
                interpolation: Valid(json::animation::Interpolation::Linear),
                output: translation_accessor,
                extensions: Default::default(),
                extras: Default::default(),
            };

            let rotation_accessor = root.push(json::Accessor {
                buffer_view: Some(rotation_buffer_view),
                byte_offset: Some(USize64::from(part_iter * mem::size_of::<Vec4>())),
                count: USize64::from(anim_frames_len),
                component_type: Valid(json::accessor::GenericComponentType(
                    json::accessor::ComponentType::F32,
                )),
                type_: Valid(json::accessor::Type::Vec4),
                min: None,
                max: None,
                normalized: false,
                sparse: None,
                extensions: Default::default(),
                extras: Default::default(),
            });

            let rotation_sampler = json::animation::Sampler {
                input: timestamp_accessor,
                interpolation: Valid(json::animation::Interpolation::Linear),
                output: rotation_accessor,
                extensions: Default::default(),
                extras: Default::default(),
            };

            let scale_accessor = root.push(json::Accessor {
                buffer_view: Some(scale_buffer_view),
                byte_offset: Some(USize64::from(part_iter * mem::size_of::<Vec3>())),
                count: USize64::from(anim_frames_len),
                component_type: Valid(json::accessor::GenericComponentType(
                    json::accessor::ComponentType::F32,
                )),
                type_: Valid(json::accessor::Type::Vec3),
                min: None,
                max: None,
                normalized: false,
                sparse: None,
                extensions: Default::default(),
                extras: Default::default(),
            });

            let scale_sampler = json::animation::Sampler {
                input: timestamp_accessor,
                interpolation: Valid(json::animation::Interpolation::Linear),
                output: scale_accessor,
                extensions: Default::default(),
                extras: Default::default(),
            };

            let translation_channel = json::animation::Channel {
                target: json::animation::Target {
                    node: Index::new(i as u32),
                    path: Valid(json::animation::Property::Translation),
                    extensions: Default::default(),
                    extras: Default::default(),
                },
                sampler: Index::new((i as u32 - 1) * 3),
                extensions: Default::default(),
                extras: Default::default(),
            };

            let rotation_channel = json::animation::Channel {
                target: json::animation::Target {
                    node: Index::new(i as u32),
                    path: Valid(json::animation::Property::Rotation),
                    extensions: Default::default(),
                    extras: Default::default(),
                },
                sampler: Index::new((i as u32) * 3 - 2),
                extensions: Default::default(),
                extras: Default::default(),
            };

            let scale_channel = json::animation::Channel {
                target: json::animation::Target {
                    node: Index::new(i as u32),
                    path: Valid(json::animation::Property::Scale),
                    extensions: Default::default(),
                    extras: Default::default(),
                },
                sampler: Index::new((i as u32) * 3 - 1),
                extensions: Default::default(),
                extras: Default::default(),
            };

            samplers.push(translation_sampler);
            samplers.push(rotation_sampler);
            samplers.push(scale_sampler);
            channels.push(translation_channel);
            channels.push(rotation_channel);
            channels.push(scale_channel);

            part_iter += anim_frames_len;
        }

        root.push(json::Animation {
            channels,
            samplers,
            extensions: Default::default(),
            extras: Default::default(),
        });
    }

    root.push(json::Scene {
        extensions: Default::default(),
        extras: Default::default(),
        nodes: vec![nodes[0]],
    });

    let mut buf = Vec::new();
    let cursor = Cursor::new(&mut buf);

    json::serialize::to_writer_pretty(cursor, &root)?;

    Ok(buf)
}

pub fn find_header_index(file: &Packed) -> anyhow::Result<usize> {
    let found = file.iter().find(|idx| {
        if file.files[*idx].len() < 8 {
            return false;
        }

        let s = &file.files[*idx];

        let len = u32::from_le_bytes([s[4], s[5], s[6], s[7]]) as usize;

        if (8 + len * 12) != file.files[*idx].len() {
            return false;
        }

        let header = match Header::read(&mut Cursor::new(s)) {
            Ok(s) => s,
            Err(_) => return false,
        };

        match header
            .parts
            .iter()
            .find(|x| x.parent_index >= header._part_count)
        {
            Some(_) => false,
            None => true,
        }
    });

    let found_unwrapped = match found {
        Some(x) => x,
        None => return Err(anyhow::anyhow!("automatic header find failed")),
    };

    Ok(found_unwrapped)
}
