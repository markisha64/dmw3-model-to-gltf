mod pack;

use binread::{io::Cursor, BinRead};
use clap::Parser;
use gltf_json as json;
use image::RgbaImage;
use json::material::{EmissiveFactor, PbrBaseColorFactor, PbrMetallicRoughness, StrengthFactor};
use json::texture::Info;
use json::validation::Checked::Valid;
use json::validation::USize64;
use json::Index;
use rlen::rlen_decode;
use std::io::Write;
use std::path::PathBuf;
use std::{fs, mem};
use tim::Tim;

#[derive(Parser, Debug, Clone)]
struct Args {
    file: PathBuf,
}

#[derive(BinRead)]
struct PartPacked {
    parent_index: u32,
    geometry: u32,
    animation: u32,
}

#[derive(BinRead)]
struct AnimationInst {
    t: u16,
    len: u16,
    c: i16,
    d: i16,
}

#[derive(BinRead, Debug, Clone)]
struct AnimationFrame {
    vx: i16,
    vy: i16,
    vz: i16,
    id: u16,
}

#[derive(Debug, Clone)]
struct AnimationFrames {
    translation: AnimationFrame,
    rotation: AnimationFrame,
    scale: AnimationFrame,
}

#[derive(Clone, Debug)]
struct Part {
    vert_offset: usize,
    vert_len: usize,
    tex_offset: usize,
    tex_len: usize,
    _animation: usize,
}

#[derive(BinRead)]
struct Header {
    texture_offset: u32,
    _part_count: u32,
    #[br(count=_part_count)]
    parts: Vec<PartPacked>,
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
struct Vertex {
    position: [f32; 3],
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
struct Mat4 {
    values: [[f32; 4]; 4],
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
struct Texel {
    position: [f32; 2],
}

const TARGET: &str = "new";
const MULTIPLIER: f32 = 1.0 / 256.0;
const SCALE_MULTIPLIER: f32 = 1.0 / 4096.0;
const FPS: f32 = 60.0;
const ROTATION: f32 = 1.0 / 1024.0;

type Point = (u32, u32);
type PointFloat = (f64, f64);

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

    let stp = (raw >> 15) > 0;

    let r = raw & 0x1f;
    let g = (raw >> 5) & 0x1f;
    let b = (raw >> 10) & 0x1f;

    if (r == 0 && g == 0 && b == 0) != stp {
        return image::Rgba([0, 0, 0, 0]);
    }

    let r_norm = ((r * 255) / 0x1f) as u8;
    let g_norm = ((g * 255) / 0x1f) as u8;
    let b_norm = ((b * 255) / 0x1f) as u8;

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

fn grab_frames(animations: pack::Packed) -> Vec<Vec<u16>> {
    animations
        .files
        .iter()
        .map(|animation| -> Vec<u16> {
            let mut reader = Cursor::new(&animation);

            let instructions: Vec<AnimationInst> = (0..animation.len() / 16)
                .map(|_| AnimationInst::read(&mut reader).unwrap())
                .collect();

            let mut frames = Vec::new();

            for instruction in instructions {
                if instruction.t == 0x7fff {
                    break;
                }

                if instruction.len == 0 {
                    frames.push(instruction.c as u16);

                    break;
                }

                if instruction.d == 0 {
                    for i in 1..instruction.len {
                        frames.push(
                            (((i as u32 * 4096) / (instruction.len as u32 + 1)) | 0x8000) as u16,
                        );
                    }
                } else {
                    for i in 0..instruction.len {
                        frames.push(instruction.c as u16 + i);
                    }
                }
            }

            frames
        })
        .collect()
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

fn create_gltf(header: &Header, filename: &str, unpacked: &pack::Packed) {
    let mut root = json::Root::default();

    let animations = grab_frames(pack::Packed::from(unpacked.files[0].clone()));

    let animation_idxs: Vec<u32> = header.parts.iter().map(|x| x.animation).collect();

    let animation_files: Vec<Vec<Vec<AnimationFrame>>> = unpacked
        .files
        .iter()
        .enumerate()
        .map(|(idx, part)| {
            if !animation_idxs.contains(&(idx as u32)) {
                return Vec::new();
            }

            let upkg = pack::Packed::from(part.clone());

            upkg.files
                .iter()
                .take(3)
                .map(|buf| {
                    let mut reader = Cursor::new(&buf);

                    let mut res = Vec::new();

                    for _ in 0..buf.len() / 16 {
                        res.push(AnimationFrame::read(&mut reader).unwrap());
                    }

                    res
                })
                .collect()
        })
        .collect();

    let texture_packed_raw = &unpacked.files[header.texture_offset as usize];

    let texture_packed = pack::Packed::from(texture_packed_raw.clone());

    let texture_raw = match rlen_decode(&texture_packed.files[0]) {
        Ok(file) => file,
        Err(_) => texture_packed_raw.clone(),
    };

    let texture_tim = Tim::from(texture_raw);

    let mut texture_png: RgbaImage = RgbaImage::new(256, 256);

    let mut nodes = Vec::new();

    let image = root.push(json::Image {
        buffer_view: None,
        mime_type: None,
        uri: Some("texture.png".into()),
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
                    position: [x * MULTIPLIER, y * MULTIPLIER, z * MULTIPLIER],
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
        let mut is_sparse = 0;
        // let mut s3 = 0;
        // let mut s4 = 0;
        let mut is_raw = 0;
        // let mut s6 = 0;
        // let mut s7 = 0;
        let mut face = [
            Vertex {
                position: [0.0, 0.0, 0.0],
            },
            Vertex {
                position: [0.0, 0.0, 0.0],
            },
            Vertex {
                position: [0.0, 0.0, 0.0],
            },
            Vertex {
                position: [0.0, 0.0, 0.0],
            },
        ];

        let mut face_tex = [
            Texel {
                position: [0.0, 0.0],
            },
            Texel {
                position: [0.0, 0.0],
            },
            Texel {
                position: [0.0, 0.0],
            },
            Texel {
                position: [0.0, 0.0],
            },
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
                        if is_raw > 0 {}

                        if is_sparse > 0 {
                            let offset = match is_raw {
                                0 => is_quad + 4 + i,
                                _ => is_quad * 2 + 7 + i,
                            };

                            if is_quad > 0 {
                                face = [
                                    vertices_sparse[face_file[i + 1] as usize],
                                    vertices_sparse[face_file[i + 2] as usize],
                                    vertices_sparse[face_file[i + 3] as usize],
                                    vertices_sparse[face_file[i + 4] as usize],
                                ];

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

                                face_tex = [
                                    Texel {
                                        position: [
                                            (tex1.0 as f32) / 256.0,
                                            (tex1.1 as f32) / 256.0,
                                        ],
                                    },
                                    Texel {
                                        position: [
                                            (tex2.0 as f32) / 256.0,
                                            (tex2.1 as f32) / 256.0,
                                        ],
                                    },
                                    Texel {
                                        position: [
                                            (tex3.0 as f32) / 256.0,
                                            (tex3.1 as f32) / 256.0,
                                        ],
                                    },
                                    Texel {
                                        position: [
                                            (tex4.0 as f32) / 256.0,
                                            (tex4.1 as f32) / 256.0,
                                        ],
                                    },
                                ];
                            } else {
                                face = [
                                    vertices_sparse[face_file[i + 1] as usize],
                                    vertices_sparse[face_file[i + 2] as usize],
                                    vertices_sparse[face_file[i + 3] as usize],
                                    Vertex {
                                        position: [0.0, 0.0, 0.0],
                                    },
                                ];

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

                                face_tex = [
                                    Texel {
                                        position: [
                                            (tex1.0 as f32) / 256.0,
                                            (tex1.1 as f32) / 256.0,
                                        ],
                                    },
                                    Texel {
                                        position: [
                                            (tex2.0 as f32) / 256.0,
                                            (tex2.1 as f32) / 256.0,
                                        ],
                                    },
                                    Texel {
                                        position: [
                                            (tex3.0 as f32) / 256.0,
                                            (tex3.1 as f32) / 256.0,
                                        ],
                                    },
                                    Texel {
                                        position: [0.0, 0.0],
                                    },
                                ];
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
                                    (face_tex[0].position[0] * 256.0) as f64,
                                    (face_tex[0].position[1] * 256.0) as f64,
                                ),
                                (
                                    (face_tex[1].position[0] * 256.0) as f64,
                                    (face_tex[1].position[1] * 256.0) as f64,
                                ),
                                (
                                    (face_tex[2].position[0] * 256.0) as f64,
                                    (face_tex[2].position[1] * 256.0) as f64,
                                ),
                            );

                            color_tex_tris(
                                &mut texture_png,
                                &texture_tim,
                                clut,
                                (
                                    (face_tex[1].position[0] * 256.0) as f64,
                                    (face_tex[1].position[1] * 256.0) as f64,
                                ),
                                (
                                    (face_tex[2].position[0] * 256.0) as f64,
                                    (face_tex[2].position[1] * 256.0) as f64,
                                ),
                                (
                                    (face_tex[3].position[0] * 256.0) as f64,
                                    (face_tex[3].position[1] * 256.0) as f64,
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
                                    (face_tex[0].position[0] * 256.0) as f64,
                                    (face_tex[0].position[1] * 256.0) as f64,
                                ),
                                (
                                    (face_tex[1].position[0] * 256.0) as f64,
                                    (face_tex[1].position[1] * 256.0) as f64,
                                ),
                                (
                                    (face_tex[2].position[0] * 256.0) as f64,
                                    (face_tex[2].position[1] * 256.0) as f64,
                                ),
                            );
                        }

                        let face_length_in_bytes = is_quad + 3;
                        let j = i;
                        i = j + face_length_in_bytes;

                        if is_raw != 0 {
                            i = j + face_length_in_bytes * 2;
                        }

                        if is_sparse != 0 {
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
                        9 => is_sparse = t,
                        0xc => is_raw = t,
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

    let texture_buffer_length = tex_coords_ref.len() * mem::size_of::<Texel>();
    let texture_buffer = root.push(json::Buffer {
        byte_length: USize64::from(texture_buffer_length),
        extensions: Default::default(),
        extras: Default::default(),
        uri: Some("tex_buffer.bin".into()),
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

    let vertex_buffer_length = vertices_ref.len() * mem::size_of::<Vertex>();
    let vertex_buffer = root.push(json::Buffer {
        byte_length: USize64::from(vertex_buffer_length),
        extensions: Default::default(),
        extras: Default::default(),
        uri: Some(format!("vertex_buffer.bin")),
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

    let joints_buffer_length = joints.len() * mem::size_of::<Vec4u16>();
    let joints_buffer = root.push(json::Buffer {
        byte_length: USize64::from(joints_buffer_length),
        uri: Some(format!("joints.bin")),
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

    let weights_buffer_length = weights.len() * mem::size_of::<Vec4>();
    let weights_buffer = root.push(json::Buffer {
        byte_length: USize64::from(weights_buffer_length),
        uri: Some(format!("weights.bin")),
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

    let mut animations_parsed: Vec<Vec<Vec<AnimationFrames>>> = Vec::new();

    for anim in animations.iter() {
        animations_parsed.push(
            header
                .parts
                .iter()
                .map(|x| -> Vec<AnimationFrames> {
                    anim.iter()
                        .map(|y| {
                            let translation = animation_files[x.animation as usize][0]
                                .iter()
                                .find(|z| z.id >= *y)
                                .or(animation_files[x.animation as usize][0].last())
                                .unwrap()
                                .clone();

                            let _rotation = animation_files[x.animation as usize][1]
                                .iter()
                                .find(|z| z.id >= *y)
                                .or(animation_files[x.animation as usize][1].last())
                                .unwrap()
                                .clone();

                            let scale = animation_files[x.animation as usize][2]
                                .iter()
                                .find(|z| z.id >= *y)
                                .or(animation_files[x.animation as usize][2].last())
                                .unwrap()
                                .clone();

                            AnimationFrames {
                                translation,
                                rotation: _rotation,
                                scale,
                            }
                        })
                        .collect()
                })
                .collect(),
        );
    }

    let mut animation_timestamps: Vec<f32> = Vec::new();
    let mut animation_translations: Vec<Vec3> = Vec::new();
    let mut animation_rotation: Vec<Vec4> = Vec::new();
    let mut animation_scale: Vec<Vec3> = Vec::new();

    let longest_animation = animations
        .iter()
        .max_by(|x, y| x.len().cmp(&y.len()))
        .unwrap()
        .len();

    animation_timestamps.extend((0..longest_animation).map(|x| x as f32 * (1.0 / FPS)));

    for animation in animations_parsed {
        for part in animation {
            animation_translations.extend(part.iter().map(|x| Vec3 {
                values: [
                    x.translation.vx as f32 * MULTIPLIER,
                    x.translation.vy as f32 * MULTIPLIER,
                    x.translation.vz as f32 * MULTIPLIER,
                ],
            }));

            animation_rotation.extend(part.iter().map(|x| to_quaternion(&x.rotation)));

            animation_scale.extend(part.iter().map(|x| Vec3 {
                values: [
                    x.scale.vx as f32 * SCALE_MULTIPLIER,
                    x.scale.vy as f32 * SCALE_MULTIPLIER,
                    x.scale.vz as f32 * SCALE_MULTIPLIER,
                ],
            }));
        }
    }

    let timestamp_buffer_length = animation_timestamps.len() * mem::size_of::<f32>();
    let timestamp_buffer = root.push(json::Buffer {
        byte_length: USize64::from(timestamp_buffer_length),
        uri: Some("ts.bin".into()),
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

    let translations_buffer_length = animation_translations.len() * mem::size_of::<Vec3>();
    let translations_buffer = root.push(json::Buffer {
        byte_length: USize64::from(translations_buffer_length),
        uri: Some("trans.bin".into()),
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

    let rotation_buffer_length = animation_rotation.len() * mem::size_of::<Vec4>();
    let rotation_buffer = root.push(json::Buffer {
        byte_length: USize64::from(rotation_buffer_length),
        uri: Some("rotation.bin".into()),
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

    let scale_buffer_length = animation_scale.len() * mem::size_of::<Vec3>();
    let scale_buffer = root.push(json::Buffer {
        byte_length: USize64::from(scale_buffer_length),
        uri: Some("scale.bin".into()),
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
        let timestamp_accessor = root.push(json::Accessor {
            buffer_view: Some(timestamp_buffer_view),
            byte_offset: Some(USize64(0)),
            count: USize64::from(animation.len()),
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

        let mut samplers = Vec::new();
        let mut channels = Vec::new();
        for i in 1..parts.len() + 1 {
            let translation_accessor = root.push(json::Accessor {
                buffer_view: Some(translations_buffer_view),
                byte_offset: Some(USize64::from(part_iter * mem::size_of::<Vec3>())),
                count: USize64::from(animation.len()),
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
                count: USize64::from(animation.len()),
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
                count: USize64::from(animation.len()),
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

            part_iter += animation.len();
        }

        root.push(json::Animation {
            channels,
            samplers,
            extensions: Default::default(),
            extras: Default::default(),
        });
    }

    let vertex_bin = to_padded_byte_vector(vertices);
    let mut vertex_writer =
        fs::File::create(format!("{TARGET}/{filename}/vertex_buffer.bin")).unwrap();
    vertex_writer.write_all(&vertex_bin).unwrap();

    let tex_coords_bin = to_padded_byte_vector(tex_coords);
    let mut tex_coords_writer =
        fs::File::create(format!("{TARGET}/{filename}/tex_buffer.bin")).unwrap();
    tex_coords_writer.write_all(&tex_coords_bin).unwrap();

    let joints_bin = to_padded_byte_vector(joints);
    let mut joints_writer = fs::File::create(format!("{TARGET}/{filename}/joints.bin")).unwrap();
    joints_writer.write_all(&joints_bin).unwrap();

    let weights_bin = to_padded_byte_vector(weights);
    let mut weights_writer = fs::File::create(format!("{TARGET}/{filename}/weights.bin")).unwrap();
    weights_writer.write_all(&weights_bin).unwrap();

    let timestamps_bin = to_padded_byte_vector(animation_timestamps);
    let mut timestamps_writer = fs::File::create(format!("{TARGET}/{filename}/ts.bin")).unwrap();
    timestamps_writer.write_all(&timestamps_bin).unwrap();

    let translations_bin = to_padded_byte_vector(animation_translations);
    let mut translations_writer =
        fs::File::create(format!("{TARGET}/{filename}/trans.bin")).unwrap();
    translations_writer.write_all(&translations_bin).unwrap();

    let rotation_bin = to_padded_byte_vector(animation_rotation);
    let mut rotation_writer =
        fs::File::create(format!("{TARGET}/{filename}/rotation.bin")).unwrap();
    rotation_writer.write_all(&rotation_bin).unwrap();

    let scale_bin = to_padded_byte_vector(animation_scale);
    let mut scale_writer = fs::File::create(format!("{TARGET}/{filename}/scale.bin")).unwrap();
    scale_writer.write_all(&scale_bin).unwrap();

    texture_png
        .save(format!("{TARGET}/{filename}/texture.png"))
        .unwrap();

    root.push(json::Scene {
        extensions: Default::default(),
        extras: Default::default(),
        nodes: vec![nodes[0]],
    });

    let writer = fs::File::create(format!("{TARGET}/{filename}/model.gltf")).unwrap();
    json::serialize::to_writer_pretty(writer, &root).unwrap();
}

fn find_header<'a>(file: &'a pack::Packed) -> &'a Vec<u8> {
    file.files
        .iter()
        .find(|x| {
            if x.len() < 8 {
                return false;
            }

            let len = u32::from_le_bytes([x[4], x[5], x[6], x[7]]) as u64;

            (8 + len * 12) == x.len() as u64
        })
        .unwrap()
}

fn main() {
    let args = Args::parse();

    let file = fs::read(&args.file).unwrap();

    let unpacked = pack::Packed::from(file);

    let header_raw = find_header(&unpacked);

    let mut header_reader = Cursor::new(&header_raw);

    let header = Header::read(&mut header_reader).unwrap();

    let filename: &str = args.file.file_stem().clone().unwrap().try_into().unwrap();

    fs::create_dir_all(format!("{TARGET}/{filename}")).unwrap();

    create_gltf(&header, filename, &unpacked);
}
