use std::{fs, iter, path::PathBuf};

use clap::Parser;

#[derive(Parser, Debug, Clone)]
struct Args {
    file: PathBuf,
}

pub fn rlen_decode(bytes: &Vec<u8>) -> Result<Vec<u8>, String> {
    if bytes[0..4] != *b"RLEN" {
        return Err("File not run length encoded".into());
    }

    let mut result = Vec::new();

    let final_length = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;

    let mut i = 8;
    let mut cur_pos = 0;

    while cur_pos < final_length {
        let blen = bytes[i] as usize;

        if blen < 0x80 {
            for j in i + 1..i + 1 + blen {
                result.push(bytes[j]);
            }

            i += blen + 1;
            cur_pos += blen;
        } else {
            let byte = bytes[i + 1];

            result.extend(iter::repeat(byte).take(blen - 0x80));

            i += 2;
            cur_pos += blen - 0x80;
        }
    }

    if result.len() != final_length {
        return Err("File length mismatch".into());
    }

    Ok(result)
}

fn main() {
    let args = Args::parse();

    let file = fs::read(args.file).expect("IO error");

    let unpacked = rlen_decode(&file).expect("RLEN error");

    fs::write("new/rlen_decoded", unpacked).expect("IO error");
}
