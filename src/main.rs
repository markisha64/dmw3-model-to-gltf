mod pack;
mod rlen;
mod tim;

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
    header_index: usize,
}

#[derive(BinRead)]
struct PartPacked {
    parent_index: u32,
    geometry: u32,
    animation: u32,
}

#[derive(BinRead)]
struct AnimationInst {
    _t: i16,
    len: i16,
    c: i16,
    d: i16,
}

#[derive(BinRead, Debug)]
struct AnimationFrame {
    vx: i16,
    vy: i16,
    id: i16,
    vz: i16,
}

#[derive(Clone, Debug)]
struct Part {
    parent_index: Option<usize>,
    vert_offset: usize,
    vert_len: usize,
    tex_offset: usize,
    tex_len: usize,
    animation: usize,
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
struct Texel {
    position: [f32; 2],
}

const TARGET: &str = "new";
const MULTIPLIER: f32 = 1.0 / 256.0;

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

    let r = raw & 0x1f;
    let g = (raw >> 5) & 0x1f;
    let b = (raw >> 10) & 0x1f;

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

fn create_gltf(header: &Header, filename: &str, unpacked: &pack::Packed) {
    let mut root = json::Root::default();

    let animation_packed = pack::Packed::from(unpacked.files[0].clone());

    let mut animation_0_reader = Cursor::new(&animation_packed.files[0]);

    let mut instructions = Vec::new();

    for _ in 0..animation_packed.files[0].len() / 16 {
        instructions.push(AnimationInst::read(&mut animation_0_reader).unwrap());
    }

    let frame = instructions
        .iter()
        .find(|x| x.d != 0 && x.len > 0)
        .unwrap()
        .c;

    let animations: Vec<u32> = header.parts.iter().map(|x| x.animation).collect();

    let animation_files: Vec<Option<Vec<Vec<AnimationFrame>>>> = unpacked
        .files
        .iter()
        .enumerate()
        .map(|(idx, part)| {
            if !animations.contains(&(idx as u32)) {
                return None;
            }

            let upkg = pack::Packed::from(part.clone());

            Some(
                upkg.files
                    .iter()
                    .take(2)
                    .map(|buf| {
                        let mut reader = Cursor::new(&buf);

                        let mut res = Vec::new();

                        for _ in 0..buf.len() / 16 {
                            res.push(AnimationFrame::read(&mut reader).unwrap());
                        }

                        res
                    })
                    .collect(),
            )
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
            0: [0.0, 0.0, 0.0, 0.0],
        },
        metallic_factor: StrengthFactor(0.0),
        metallic_roughness_texture: None,
        roughness_factor: StrengthFactor(0.0),
        extensions: Default::default(),
        extras: Default::default(),
    };

    let material = root.push(json::Material {
        alpha_cutoff: None,
        alpha_mode: Valid(json::material::AlphaMode::Opaque),
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

    let vertices_ref = &mut vertices;
    let tex_coords_ref = &mut tex_coords;

    // root
    parts.push(Part {
        animation: 0,
        parent_index: None,
        vert_offset: 0,
        vert_len: 0,
        tex_offset: 0,
        tex_len: 0,
    });

    for part in header.parts.iter() {
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

        let mut is_quad = 1;
        let mut is_textured = 1;
        // let mut s3 = 0;
        // let mut s4 = 0;
        let mut has_normals = 1;
        // let mut s6 = 0;
        // let mut s7 = 0;

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
                    } else if t < 2 {
                        let offset = match has_normals {
                            0 => is_quad + 4 + i,
                            _ => is_quad * 2 + 7 + i,
                        };

                        if is_quad > 0 {
                            vertices_ref.push(vertices_sparse[face_file[i + 1] as usize]);
                            vertices_ref.push(vertices_sparse[face_file[i + 2] as usize]);
                            vertices_ref.push(vertices_sparse[face_file[i + 3] as usize]);

                            vertices_ref.push(vertices_sparse[face_file[i + 2] as usize]);
                            vertices_ref.push(vertices_sparse[face_file[i + 3] as usize]);
                            vertices_ref.push(vertices_sparse[face_file[i + 4] as usize]);

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

                            tex_coords_ref.push(Texel {
                                position: [(tex1.0 as f32) / 256.0, (tex1.1 as f32) / 256.0],
                            });
                            tex_coords_ref.push(Texel {
                                position: [(tex2.0 as f32) / 256.0, (tex2.1 as f32) / 256.0],
                            });
                            tex_coords_ref.push(Texel {
                                position: [(tex3.0 as f32) / 256.0, (tex3.1 as f32) / 256.0],
                            });

                            tex_coords_ref.push(Texel {
                                position: [(tex2.0 as f32) / 256.0, (tex2.1 as f32) / 256.0],
                            });
                            tex_coords_ref.push(Texel {
                                position: [(tex3.0 as f32) / 256.0, (tex3.1 as f32) / 256.0],
                            });
                            tex_coords_ref.push(Texel {
                                position: [(tex4.0 as f32) / 256.0, (tex4.1 as f32) / 256.0],
                            });

                            let tex1f = (tex1.0 as f64, tex1.1 as f64);
                            let tex2f = (tex2.0 as f64, tex2.1 as f64);
                            let tex3f = (tex3.0 as f64, tex3.1 as f64);
                            let tex4f = (tex4.0 as f64, tex4.1 as f64);

                            color_tex_tris(
                                &mut texture_png,
                                &texture_tim,
                                clut,
                                tex1f,
                                tex2f,
                                tex3f,
                            );
                            color_tex_tris(
                                &mut texture_png,
                                &texture_tim,
                                clut,
                                tex2f,
                                tex3f,
                                tex4f,
                            );
                        } else {
                            vertices_ref.push(vertices_sparse[face_file[i + 1] as usize]);
                            vertices_ref.push(vertices_sparse[face_file[i + 2] as usize]);
                            vertices_ref.push(vertices_sparse[face_file[i + 3] as usize]);

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
                            tex_coords_ref.push(Texel {
                                position: [(tex1.0 as f32) / 256.0, (tex1.1 as f32) / 256.0],
                            });
                            tex_coords_ref.push(Texel {
                                position: [(tex2.0 as f32) / 256.0, (tex2.1 as f32) / 256.0],
                            });
                            tex_coords_ref.push(Texel {
                                position: [(tex3.0 as f32) / 256.0, (tex3.1 as f32) / 256.0],
                            });

                            let tex1f = (tex1.0 as f64, tex1.1 as f64);
                            let tex2f = (tex2.0 as f64, tex2.1 as f64);
                            let tex3f = (tex3.0 as f64, tex3.1 as f64);

                            color_tex_tris(
                                &mut texture_png,
                                &texture_tim,
                                clut,
                                tex1f,
                                tex2f,
                                tex3f,
                            );
                        }

                        let mut face_length_in_bytes = is_quad + 3;

                        if is_textured != 0 {
                            face_length_in_bytes *= 2;
                        }

                        if has_normals != 0 {
                            face_length_in_bytes *= 2;
                        }

                        i += face_length_in_bytes + 1;
                    } else if t < 6 {
                        i += 4;
                    }
                }
                _ => {
                    match ctype {
                        8 => is_quad = t,
                        9 => is_textured = t,
                        0xc => has_normals = t,
                        _ => {}
                    }

                    i += 1;
                }
            }
        }

        parts.push(Part {
            animation: part.animation as usize,
            parent_index: Some(part.parent_index as usize),
            vert_offset: old_vertex_len,
            vert_len: vertices_ref.len() - old_vertex_len,
            tex_offset: old_tex_len,
            tex_len: tex_coords_ref.len() - old_tex_len,
        });
    }

    let mut part_children: Vec<Vec<usize>> = parts.iter().map(|_| Vec::new()).collect();

    for (idx, part) in parts.iter().enumerate() {
        if let Some(parent) = part.parent_index {
            part_children[parent].push(idx);
        }
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

    for (idx, part) in parts.iter().enumerate() {
        let mesh = match part.vert_len != 0 {
            true => {
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

                let primitive = json::mesh::Primitive {
                    attributes: {
                        let mut map = std::collections::BTreeMap::new();
                        map.insert(Valid(json::mesh::Semantic::Positions), positions);
                        map.insert(
                            Valid(json::mesh::Semantic::TexCoords(0)),
                            tex_coords_accessor,
                        );
                        map
                    },
                    extensions: Default::default(),
                    extras: Default::default(),
                    indices: None,
                    material: Some(material),
                    mode: Valid(json::mesh::Mode::Triangles),
                    targets: None,
                };

                Some(root.push(json::Mesh {
                    extensions: Default::default(),
                    extras: Default::default(),
                    primitives: vec![primitive],
                    weights: None,
                }))
            }
            false => None,
        };

        let translation = match part.vert_len != 0 {
            true => {
                let anim_frame = animation_files[part.animation].as_ref().unwrap()[0]
                    .iter()
                    .find(|x| x.id >= frame)
                    .or(animation_files[part.animation].as_ref().unwrap()[0].last())
                    .unwrap();

                Some([
                    anim_frame.vx as f32 * MULTIPLIER,
                    anim_frame.vy as f32 * MULTIPLIER,
                    anim_frame.vz as f32 * MULTIPLIER,
                ])
            }
            false => None,
        };

        let node = root.push(json::Node {
            mesh,
            children: Some(
                part_children[idx]
                    .iter()
                    .map(|pc| Index::new(*pc as u32))
                    .collect(),
            ),
            translation,
            ..Default::default()
        });

        nodes.push(node);
    }

    let vertex_bin = to_padded_byte_vector(vertices);
    let mut vertex_writer =
        fs::File::create(format!("{TARGET}/{filename}/vertex_buffer.bin")).unwrap();
    vertex_writer.write_all(&vertex_bin).unwrap();

    let tex_coords_bin = to_padded_byte_vector(tex_coords);
    let mut tex_coords_writer =
        fs::File::create(format!("{TARGET}/{filename}/tex_buffer.bin")).unwrap();
    tex_coords_writer.write_all(&tex_coords_bin).unwrap();

    texture_png
        .save(format!("{TARGET}/{filename}/texture.png"))
        .unwrap();

    root.push(json::Scene {
        extensions: Default::default(),
        extras: Default::default(),
        nodes,
    });

    let writer = fs::File::create(format!("{TARGET}/{filename}/model.gltf")).unwrap();
    json::serialize::to_writer_pretty(writer, &root).unwrap();
}

fn main() {
    let args = Args::parse();

    let file = fs::read(&args.file).unwrap();

    let unpacked = pack::Packed::from(file);

    let header_raw = &unpacked.files[args.header_index];

    let mut header_reader = Cursor::new(&header_raw);

    let header = Header::read(&mut header_reader).unwrap();

    let filename: &str = args.file.file_stem().clone().unwrap().try_into().unwrap();

    fs::create_dir_all(format!("{TARGET}/{filename}")).unwrap();

    create_gltf(&header, filename, &unpacked);
}
