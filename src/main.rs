mod pack;
mod tim;

use binread::{io::Cursor, BinRead};
use clap::Parser;
use gltf_json as json;
use json::validation::Checked::Valid;
use json::validation::USize64;
use std::io::Write;
use std::path::PathBuf;
use std::{fs, mem};

#[derive(Parser, Debug, Clone)]
struct Args {
    file: PathBuf,
    header_index: usize,
}

#[derive(BinRead)]
struct PartPacked {
    _id: u32,
    triangles: u32,
    _f2: u32,
}

#[derive(BinRead)]
struct Header {
    _texture_offset: u32,
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
struct Triangle {
    indices: [u16; 3],
}

const TARGET: &str = "new";
const DIVISOR: f32 = 256.0;

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

fn parts_to_gltf(header: &Header, filename: &str, unpacked: &pack::Packed) {
    let mut root = json::Root::default();

    let mut nodes = Vec::new();

    for (idx, part) in header.parts.iter().enumerate() {
        let vfile = &unpacked.files[part.triangles as usize];

        let verts_offset = u32::from_le_bytes([vfile[0], vfile[1], vfile[2], vfile[3]]) as usize;

        let verts = &vfile[verts_offset..];

        let vert_count = u16::from_le_bytes([verts[0], verts[1]]) as usize;

        let vertices: Vec<Vertex> = verts[6..]
            .chunks_exact(6)
            .take(vert_count)
            .map(|chunk| {
                let x: f32 = i16::from_le_bytes([chunk[0], chunk[1]]).into();
                let y: f32 = i16::from_le_bytes([chunk[2], chunk[3]]).into();
                let z: f32 = i16::from_le_bytes([chunk[4], chunk[5]]).into();

                return Vertex {
                    position: [x / DIVISOR, y / DIVISOR, z / DIVISOR],
                };
            })
            .collect();

        let faces_offset = u32::from_le_bytes([vfile[8], vfile[9], vfile[10], vfile[11]]) as usize;

        let face_file = &vfile[faces_offset..];

        let mut faces: Vec<Triangle> = Vec::new();

        let mut is_quad = 1;
        let mut s2 = 1;
        // let mut s3 = 0;
        // let mut s4 = 0;
        let mut s5 = 1;
        // let mut s6 = 0;
        // let mut s7 = 0;

        let mut i = 0;

        while face_file[i] != 0xff {
            let ctype = face_file[i] >> 4;
            let t = (face_file[i] & 0xf) as usize;

            if ctype == 0 {
                if t == 1 {
                    i += 7;
                } else if t < 2 {
                    if is_quad > 0 {
                        faces.push(Triangle {
                            indices: [
                                face_file[i + 1] as u16,
                                face_file[i + 2] as u16,
                                face_file[i + 3] as u16,
                            ],
                        });
                        faces.push(Triangle {
                            indices: [
                                face_file[i + 2] as u16,
                                face_file[i + 3] as u16,
                                face_file[i + 4] as u16,
                            ],
                        });
                    } else {
                        faces.push(Triangle {
                            indices: [
                                face_file[i + 1] as u16,
                                face_file[i + 2] as u16,
                                face_file[i + 3] as u16,
                            ],
                        });
                    }

                    let mut face_length_in_bytes = is_quad + 3;

                    if s2 != 0 {
                        face_length_in_bytes *= 2;
                    }

                    if s5 != 0 {
                        face_length_in_bytes *= 2;
                    }

                    i += face_length_in_bytes + 1;
                } else if t < 6 {
                    i += 4;
                }
            } else {
                match ctype {
                    8 => is_quad = t,
                    9 => s2 = t,
                    0xc => s5 = t,
                    _ => {}
                }

                i += 1;
            }
        }

        let vertex_buffer_length = vertices.len() * mem::size_of::<Vertex>();
        let vertex_buffer = root.push(json::Buffer {
            byte_length: USize64::from(vertex_buffer_length),
            extensions: Default::default(),
            extras: Default::default(),
            uri: Some(format!("vertex_buffer{}.bin", idx)),
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

        let positions = root.push(json::Accessor {
            buffer_view: Some(vertex_view),
            byte_offset: Some(USize64(0)),
            count: USize64::from(vertices.len()),
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

        let faces_buffer_length = faces.len() * mem::size_of::<Triangle>();
        let faces_buffer = root.push(json::Buffer {
            byte_length: USize64::from(faces_buffer_length),
            extensions: Default::default(),
            extras: Default::default(),
            uri: Some(format!("face_buffer{}.bin", idx)),
        });

        let faces_view = root.push(json::buffer::View {
            buffer: faces_buffer,
            byte_length: USize64::from(faces_buffer_length),
            byte_offset: None,
            byte_stride: Some(json::buffer::Stride(mem::size_of::<u16>())),
            extensions: Default::default(),
            extras: Default::default(),
            target: Some(Valid(json::buffer::Target::ElementArrayBuffer)),
        });

        let indices = root.push(json::Accessor {
            buffer_view: Some(faces_view),
            byte_offset: Some(USize64(0)),
            count: USize64::from(
                (faces.len() * mem::size_of::<Triangle>()) / mem::size_of::<u16>(),
            ),
            component_type: Valid(json::accessor::GenericComponentType(
                json::accessor::ComponentType::U16,
            )),
            extensions: Default::default(),
            extras: Default::default(),
            type_: Valid(json::accessor::Type::Scalar),
            min: None,
            max: None,
            normalized: false,
            sparse: None,
        });

        let primitive = json::mesh::Primitive {
            attributes: {
                let mut map = std::collections::BTreeMap::new();
                map.insert(Valid(json::mesh::Semantic::Positions), positions);
                map
            },
            extensions: Default::default(),
            extras: Default::default(),
            indices: Some(indices),
            material: None,
            mode: Valid(json::mesh::Mode::Triangles),
            targets: None,
        };

        let mesh = root.push(json::Mesh {
            extensions: Default::default(),
            extras: Default::default(),
            primitives: vec![primitive],
            weights: None,
        });

        let node = root.push(json::Node {
            mesh: Some(mesh),
            ..Default::default()
        });

        nodes.push(node);

        let vertex_bin = to_padded_byte_vector(vertices);
        let mut vertex_writer =
            fs::File::create(format!("{TARGET}/{filename}/vertex_buffer{}.bin", idx)).unwrap();
        vertex_writer.write_all(&vertex_bin).unwrap();

        let faces_bin = to_padded_byte_vector(faces);
        let mut faces_writer =
            fs::File::create(format!("{TARGET}/{filename}/face_buffer{}.bin", idx)).unwrap();
        faces_writer.write_all(&faces_bin).unwrap();
    }

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

    parts_to_gltf(&header, filename, &unpacked);
}
