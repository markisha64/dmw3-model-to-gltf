use std::fs;
use std::path::PathBuf;

use clap::Parser;

#[derive(Debug)]
pub enum BPP {
    B4 = 0,
    B8 = 1,
    B16 = 2,
    B24 = 3,
}

impl From<u8> for BPP {
    fn from(byte: u8) -> BPP {
        match byte {
            0 => BPP::B4,
            1 => BPP::B8,
            2 => BPP::B16,
            _ => BPP::B24,
        }
    }
}

#[derive(Debug)]
pub struct Header {
    _version: u8,
    bpp: BPP,
    _clp: bool,
}

#[derive(Debug)]
pub struct Clut {
    length: u32,
    _x: u16,
    _y: u16,
    _width: u16,
    _height: u16,
    _bytes: Vec<u8>,
}

#[derive(Debug)]
pub struct Image {
    length: u32,
    _x: u16,
    _y: u16,
    _width: u16,
    _height: u16,
    _bytes: Vec<u8>,
}

#[derive(Debug)]
pub struct Tim {
    header: Header,
    clut: Option<Clut>,
    image: Image,
}

impl From<Vec<u8>> for Tim {
    fn from(bytes: Vec<u8>) -> Self {
        let version = bytes[1];

        let byte_5 = bytes[4] as u32;

        let bpp = (byte_5 & 0x3) as u8;
        let clp = (byte_5 & 0x8) > 0;

        let header = Header {
            _version: version,
            bpp: BPP::from(bpp),
            _clp: clp,
        };

        let mut clut = None;
        if clp {
            let cbytes = &bytes[8..];

            let clut_length = u32::from_le_bytes([cbytes[0], cbytes[1], cbytes[2], cbytes[3]]);

            let clut_x = u16::from_le_bytes([cbytes[4], cbytes[5]]);
            let clut_y = u16::from_le_bytes([cbytes[6], cbytes[7]]);
            let clut_width = u16::from_le_bytes([cbytes[8], cbytes[9]]);
            let clut_height = u16::from_le_bytes([cbytes[10], cbytes[11]]);

            let bclut = Clut {
                length: clut_length,
                _x: clut_x,
                _y: clut_y,
                _width: clut_width,
                _height: clut_height,
                _bytes: cbytes[..clut_length as usize].into(),
            };

            clut = Some(bclut);
        }

        let ibytes = match clp {
            true => &bytes[(clut.as_ref().unwrap()).length as usize + 8..],
            _ => &bytes[8..],
        };

        let image_length = u32::from_le_bytes([ibytes[0], ibytes[1], ibytes[2], ibytes[3]]);

        let image_x = u16::from_le_bytes([ibytes[4], ibytes[5]]);
        let image_y = u16::from_le_bytes([ibytes[6], ibytes[7]]);
        let image_width = u16::from_le_bytes([ibytes[8], ibytes[9]]);
        let image_height = u16::from_le_bytes([ibytes[10], ibytes[11]]);

        let image = Image {
            length: image_length,
            _x: image_x,
            _y: image_y,
            _width: image_width,
            _height: image_height,
            _bytes: ibytes[..image_length as usize].into(),
        };

        Tim {
            header,
            clut,
            image,
        }
    }
}

#[derive(Parser, Debug, Clone)]
struct Args {
    file: PathBuf,
}

fn main() {
    let args = Args::parse();

    let file = fs::read(args.file).unwrap();

    let tim = Tim::from(file);

    dbg!(tim.image.length);
    dbg!(tim.image._x);
    dbg!(tim.image._y);
    dbg!(tim.image._width);
    dbg!(tim.image._height);
}
