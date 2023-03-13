use std::fs;

fn main() {
    let paths = fs::read_dir("msdfgen/core").unwrap();
    let mut files = vec![];

    for path in paths {
        let p = path.unwrap().path();
        let p_ext = p.extension().unwrap();
        if p_ext.eq("cpp".into()) {
            files.push(p)
        }
    }

    cxx_build::bridge("src/lib.rs")
        .files(files)
        .flag_if_supported("-std=c++14")
        .define("MSDFGEN_PUBLIC", "")
        .compile("msdfgen");

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/bindings.cpp");
    println!("cargo:rerun-if-changed=src/bindings.h");
}
