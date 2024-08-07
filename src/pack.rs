use binread::BinRead;
use std::{io::Cursor, ops::Range};

#[derive(Clone)]
pub struct Packed {
    pub assumed_length: Vec<usize>,
    buffer: Vec<u8>,
    offsets: Vec<usize>,
}

impl Packed {
    pub fn _file_size(&self) -> usize {
        self.buffer.len()
    }

    pub fn get_file(&self, idx: usize) -> &[u8] {
        &self.buffer[self.offsets[idx]..]
    }

    pub fn iter(&self) -> Range<usize> {
        0..self.offsets.len()
    }
}

trait Length {
    fn len_t(&self) -> usize;
}

impl<T> Length for Vec<T> {
    fn len_t(&self) -> usize {
        self.len()
    }
}

impl<T> Length for &[T] {
    fn len_t(&self) -> usize {
        self.len()
    }
}

impl<T: AsRef<[u8]> + Length + Into<Vec<u8>>> From<T> for Packed {
    fn from(file: T) -> Self {
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
            assumed_length.push((offsets[i + 1] as i32 - offsets[i] as i32).max(0) as usize);
        }

        assumed_length
            .push((file.len_t() as i32 - *offsets.last().unwrap() as i32).max(0) as usize);

        Packed {
            buffer: file.into(),
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
