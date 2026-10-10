//! tl-proto-gen <descriptor-set.binpb> <out-dir>
//!
//! Generates prost messages and tonic clients and servers for every file in
//! the descriptor set, one `<package>.rs` per proto package. No protoc: the
//! descriptors come from `buf build`.
use prost::Message;
use std::{env, fs, path::PathBuf};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: tl-proto-gen <descriptor-set.binpb> <out-dir>");
        std::process::exit(2);
    }
    let bytes = fs::read(&args[1]).expect("read descriptor set");
    let fds = prost_types::FileDescriptorSet::decode(bytes.as_slice()).expect("decode descriptor set");
    let out = PathBuf::from(&args[2]);
    fs::create_dir_all(&out).expect("create out dir");
    tonic_prost_build::configure()
        .build_client(true)
        .build_server(true)
        .build_transport(true)
        .out_dir(&out)
        .compile_fds(fds)
        .expect("generate");
}
