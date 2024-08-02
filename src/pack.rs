use binread::BinRead;
use std::io::Cursor;

#[derive(Clone)]
pub struct Packed {
    pub buffer: Vec<u8>,
    pub assumed_length: Vec<usize>,
    pub offsets: Vec<usize>,
}

impl Packed {
    pub fn _file_size(&self) -> usize {
        self.buffer.len()
    }

    pub fn get_file(&self, idx: usize) -> &[u8] {
        &self.buffer[self.offsets[idx]..]
    }
}

impl From<Vec<u8>> for Packed {
    fn from(file: Vec<u8>) -> Self {
        let mut reader = Cursor::new(&file);

        let first_offset = u32::read(&mut reader).unwrap();
        let length = first_offset / 4;

        let mut offsets: Vec<usize> = vec![first_offset as usize];
        for _ in 1..length {
            let offset = u32::read(&mut reader).unwrap() as usize;

            offsets.push(offset);
        }

        let mut assumed_length = Vec::new();

        for i in 0..offsets.len() - 1 {
            assumed_length.push((offsets[i + 1] - offsets[i]).max(0));
        }

        assumed_length.push((offsets.len() - offsets.last().unwrap()).max(0));

        Packed {
            buffer: file,
            assumed_length,
            offsets,
        }
    }
}

impl Into<Vec<u8>> for Packed {
    fn into(self) -> Vec<u8> {
        self.buffer.clone()
    }
}
