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

impl TryFrom<Vec<u8>> for Packed {
    type Error = anyhow::Error;

    fn try_from(file: Vec<u8>) -> Result<Self, Self::Error> {
        let mut reader = Cursor::new(&file);

        let first_offset = u32::read(&mut reader)?;
        let length = first_offset / 4;

        let mut offsets: Vec<usize> = vec![first_offset as usize];
        for _ in 1..length {
            let offset = u32::read(&mut reader)? as usize;

            offsets.push(offset);
        }

        let mut assumed_length = Vec::new();

        for i in 0..offsets.len() - 1 {
            let offset1 = match offsets.get(i + 1) {
                Some(k) => *k as i32,
                None => return Err(anyhow::anyhow!("invalid index")),
            };

            let offset2 = match offsets.get(i) {
                Some(k) => *k as i32,
                None => return Err(anyhow::anyhow!("invalid index")),
            };

            assumed_length.push((offset1 - offset2).max(0) as usize);
        }

        let last_offset = match offsets.last() {
            Some(k) => *k as i32,
            None => return Err(anyhow::anyhow!("empty vec")),
        };

        assumed_length.push((file.len() as i32 - last_offset).max(0) as usize);

        Ok(Packed {
            buffer: file.into(),
            assumed_length,
            offsets,
        })
    }
}

impl TryFrom<&[u8]> for Packed {
    type Error = anyhow::Error;

    fn try_from(file: &[u8]) -> Result<Self, Self::Error> {
        let mut reader = Cursor::new(&file);

        let first_offset = u32::read(&mut reader)?;
        let length = first_offset / 4;

        let mut offsets: Vec<usize> = vec![first_offset as usize];
        for _ in 1..length {
            let offset = u32::read(&mut reader)? as usize;

            offsets.push(offset);
        }

        let mut assumed_length = Vec::new();

        for i in 0..offsets.len() - 1 {
            let offset1 = match offsets.get(i + 1) {
                Some(k) => *k as i32,
                None => return Err(anyhow::anyhow!("invalid index")),
            };

            let offset2 = match offsets.get(i) {
                Some(k) => *k as i32,
                None => return Err(anyhow::anyhow!("invalid index")),
            };

            assumed_length.push((offset1 - offset2).max(0) as usize);
        }

        let last_offset = match offsets.last() {
            Some(k) => *k as i32,
            None => return Err(anyhow::anyhow!("empty vec")),
        };

        assumed_length.push((file.len() as i32 - last_offset).max(0) as usize);

        Ok(Packed {
            buffer: file.into(),
            assumed_length,
            offsets,
        })
    }
}

impl Into<Vec<u8>> for Packed {
    fn into(self) -> Vec<u8> {
        self.buffer.clone()
    }
}
