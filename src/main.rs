mod pack;
use binread::{io::Cursor, BinRead};
use clap::Parser;
use gltf::buffer::Data;
use gltf::json::mesh::Mode;
use gltf::{buffer, Document, Mesh, Primitive};
use std::path::PathBuf;
use std::{fs, mem};

#[derive(Parser, Debug, Clone)]
struct Args {
    file: PathBuf,
    header_index: usize,
}

#[derive(BinRead)]
struct PartPacked {
    id: u32,
    triangles: u32,
    f2: u32,
}

#[derive(BinRead)]
struct Header {
    texture_offset: u32,
    part_count: u32,
    #[br(count=part_count)]
    parts: Vec<PartPacked>,
}

type Tri = (usize, usize, usize);
type Quad = (usize, usize, usize, usize);

enum Face {
    Tri(Tri),
    Quad(Quad),
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
struct Vertex {
    position: [f32; 3],
}

const TARGET: &str = "new";
const DIVISOR: f32 = 256.0;

fn parts_to_obj(parts: &Vec<PartPacked>, filename: &str, unpacked: &pack::Packed) {
    for (idx, part) in parts.iter().enumerate() {
        let mut output: Vec<String> = Vec::new();

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
                    position: [x, y, z],
                };
            })
            .collect();

        let buffer_length = vertices.len() * mem::size_of::<Vertex>();
        let buffer = root.push(json::Buffer {
            byte_length: USize64::from(buffer_length),
            extensions: Default::default(),
            extras: Default::default(),
            uri: Some(format!("buffer{}.bin", idx)),
        });

        for chunk in verts[6..].chunks_exact(6).take(vert_count) {
            let x: f32 = i16::from_le_bytes([chunk[0], chunk[1]]).into();
            let y: f32 = i16::from_le_bytes([chunk[2], chunk[3]]).into();
            let z: f32 = i16::from_le_bytes([chunk[4], chunk[5]]).into();

            output.push(format!("v {} {} {}", x / DIVISOR, y / DIVISOR, z / DIVISOR));
        }

        let faces_offset = u32::from_le_bytes([vfile[8], vfile[9], vfile[10], vfile[11]]) as usize;

        let face_file = &vfile[faces_offset..];

        let mut faces: Vec<Face> = Vec::new();

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
                        faces.push(Face::Quad((
                            face_file[i + 1] as usize + 1,
                            face_file[i + 2] as usize + 1,
                            face_file[i + 4] as usize + 1,
                            face_file[i + 3] as usize + 1,
                        )));
                    } else {
                        faces.push(Face::Tri((
                            face_file[i + 1] as usize + 1,
                            face_file[i + 2] as usize + 1,
                            face_file[i + 3] as usize + 1,
                        )));
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

        for face in faces {
            match face {
                Face::Tri(tri) => output.push(format!("f {} {} {}", tri.0, tri.1, tri.2)),
                Face::Quad(quad) => {
                    output.push(format!("f {} {} {} {}", quad.0, quad.1, quad.2, quad.3))
                }
            }
        }

        let full_output = output.join("\n");

        fs::write(format!("{TARGET}/{filename}/{idx}.obj"), full_output).unwrap();
    }
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

    parts_to_obj(&header.parts, filename, &unpacked);
}
