use std::fs;

use std::path::PathBuf;

use clap::Parser;

#[derive(Parser, Debug, Clone)]
pub struct Args {
    pub file: PathBuf,
}

use rlen::rlen_decode;

fn main() {
    let args = Args::parse();

    let file = fs::read(args.file).expect("IO error");

    let unpacked = rlen_decode(&file).expect("RLEN error");

    fs::write("new/rlen_decoded", unpacked).expect("IO error");
}
