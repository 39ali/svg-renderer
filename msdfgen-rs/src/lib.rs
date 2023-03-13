use std::time::Instant;

#[cxx::bridge()]
pub mod msdfgen {
    // // Shared structs with fields visible to both languages.
    // struct BlobMetadata {
    //     size: usize,
    //     tags: Vec<String>,
    // }

    // // Rust types and signatures exposed to C++.
    // extern "Rust" {
    //     type MultiBuf;

    //     fn next_chunk(buf: &mut MultiBuf) -> &[u8];
    // }

    // C++ types and signatures exposed to Rust.

    unsafe extern "C++" {
        include!("msdfgen-rs/src/bindings.h");

        type Shape<'a>;
        type Contour;
        type Point2;
        type Bitmapf3;
        fn new_shape<'a>() -> UniquePtr<Shape<'a>>;
        fn shape_guess_orientation(shape: &Shape, msdf: Pin<&mut Bitmapf3>);
        fn new_contour<'a>(shape: Pin<&'a mut Shape>) -> Pin<&'a mut Contour>;
        fn new_point2(x: f64, y: f64) -> UniquePtr<Point2>;

        fn add_linear_segment<'a>(contour: Pin<&'a mut Contour>, p0: &Point2, p1: &Point2);
        fn add_quadratic_segment<'a>(
            contour: Pin<&'a mut Contour>,
            p0: &Point2,
            p1: &Point2,
            c0: &Point2,
        );
        fn add_cubic_segment<'a>(
            contour: Pin<&'a mut Contour>,
            p0: &Point2,
            p1: &Point2,
            c0: &Point2,
            c1: &Point2,
        );

        fn normalize(self: Pin<&mut Shape>);
        fn edge_coloring_simple<'a>(shape: Pin<&'a mut Shape>, angleThreshold: f64);

        //bitmap
        fn new_bitmapf3(w: i32, h: i32) -> UniquePtr<Bitmapf3>;
        fn width(self: &Bitmapf3) -> i32;
        fn height(self: &Bitmapf3) -> i32;

        fn generate_msdf(
            output: Pin<&mut Bitmapf3>,
            shape: &Shape,
            range: f64,
            scalex: f64,
            scaley: f64,
            translatex: f64,
            translatey: f64,
        );

        fn bitmapf3_pixels(bitmap: &Bitmapf3) -> &[f32];
    }
}
