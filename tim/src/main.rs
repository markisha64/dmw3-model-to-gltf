use std::fs;
use std::path::PathBuf;

use clap::Parser;
use tim::Tim;

#[derive(Parser, Debug, Clone)]
struct Args {
    file: PathBuf,
}

fn main() {
    let args = Args::parse();

    let file = fs::read(args.file).unwrap();

    let tim = Tim::from(file);

    let bytes: Vec<u8> = tim.into();

    fs::write("new/new.tim", bytes).unwrap();
}
