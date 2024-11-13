use binread::io::Cursor;
use binread::BinRead;
use clap::Parser;
use dmw3_model::Header;
use dmw3_pack::Packed;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::thread;

const TARGET: &str = "new";

#[derive(Parser, Debug, Clone)]
struct Args {
    file: PathBuf,

    #[arg(long)]
    header_index: Option<usize>,

    #[arg(short, long, default_value_t = 1)]
    threads: usize,
}

fn process_file(path: &PathBuf, header_index: Option<usize>) -> anyhow::Result<()> {
    println!("{}", path.to_str().unwrap_or("unk"));
    let file = fs::read(&path)?;

    let unpacked = Packed::from(file);

    let header_raw = match header_index {
        None => &unpacked.files[dmw3_model_to_gltf::find_header_index(&unpacked)?],
        Some(x) => &unpacked.files[x],
    };

    let mut header_reader = Cursor::new(header_raw);

    let header = Header::read(&mut header_reader)?;

    let filename_os = match path.file_stem() {
        Some(x) => x,
        None => return Err(anyhow::anyhow!("failed to get file stem")),
    };

    let filename: &str = filename_os.try_into()?;

    fs::create_dir_all(format!("{TARGET}/{filename}"))?;

    let buf = dmw3_model_to_gltf::create_gltf(&header, &unpacked)?;
    let mut writer = fs::File::create(format!("{TARGET}/{filename}/model.gltf"))?;
    writer.write(&buf[..])?;

    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    match args.file.is_file() {
        true => process_file(&args.file, args.header_index)?,
        false => {
            let mut entries: Vec<PathBuf> = Vec::new();

            for entry in fs::read_dir(args.file)? {
                let entryw = entry?;
                let fpath = entryw.path();

                let meta = entryw.metadata()?;
                let fname = match fpath.file_name() {
                    Some(s) => s.to_str().unwrap_or(""),
                    None => continue,
                };

                if meta.is_dir() || fname.starts_with(".") {
                    continue;
                }

                entries.push(fpath);
            }

            let chunk_size = (entries.len() + args.threads - 1) / args.threads;

            thread::scope(|s| {
                let threads: Vec<_> = entries
                    .chunks(chunk_size)
                    .map(|chunk| {
                        s.spawn(move || {
                            let mut success = 0;

                            for fpath in chunk {
                                match process_file(&fpath, None) {
                                    Ok(_) => success += 1,
                                    Err(e) => println!("{}", e),
                                };
                            }

                            success
                        })
                    })
                    .collect();

                let sum = threads
                    .into_iter()
                    .fold(0, |pv, thread| pv + thread.join().unwrap_or(0));

                println!("{}/{} success", sum, entries.len());
            })
        }
    }

    Ok(())
}
