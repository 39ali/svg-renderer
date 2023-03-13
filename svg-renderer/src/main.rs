use core::slice::{self};
use cxx::UniquePtr;
use geo::{
    BooleanOps, ConcaveHull, Coord, LineString, Point, Polygon, Simplify, SimplifyIdx, SimplifyVW,
    SimplifyVWPreserve, Triangle,
};
use glam::Vec3;
use glow::{Context, HasContext, NativeProgram, NativeUniformLocation, NativeVertexArray};
use image;
use lyon::geom::euclid::{Point2D, UnknownUnit};
use lyon::geom::Line;
use msdfgen_rs::msdfgen::Bitmapf3;
use msdfgen_rs::*;
use std::cell::RefCell;
use std::f32::EPSILON;
use std::fs;

use std::time::Instant;

use glutin::event::{ElementState, Event, WindowEvent};
use glutin::event_loop::ControlFlow;

use memoffset::offset_of;

#[derive(Default, Debug, Clone, Copy)]
#[repr(C)]
struct BBox {
    x0: f32,
    x1: f32,
    y0: f32,
    y1: f32,
}

#[derive(Default, Clone)]
struct LinearGrad {
    bbox: glam::Vec4, // [x0,x1,y0,y1]
    stops: Vec<usvg::Stop>,
}

#[derive(Clone)]
struct Color {
    r: u8,
    g: u8,
    b: u8,
    a: f32,
}

impl Color {
    fn pack_to_4bytes(&self) -> f32 {
        let r = self.r as u32;
        let g = (self.g as u32) << 8;
        let b = (self.b as u32) << 16;
        let color: u32 = b | g | r;
        color as f32
    }

    fn pack_to_4bytes2(r: u8, g: u8, b: u8) -> f32 {
        let r = r as u32;
        let g = (g as u32) << 8;
        let b = (b as u32) << 16;
        let color: u32 = b | g | r;
        color as f32
    }
}

struct PathData {
    // msdf bitmap
    msdf: UniquePtr<Bitmapf3>,
    w: i32,
    h: i32,
    border: i32, // padding for texture

    // shape data
    fill_color: Option<Color>,
    bbox: BBox, // bbox for path
    linear: Option<LinearGrad>,
}
struct SvgData {
    paths: Vec<PathData>,
    size: usvg::Size,
    svg_bbox: BBox,
}

#[derive(Default, Clone)]
#[repr(C)]
struct GpuLinearGrad {
    bbox: glam::Vec4,
    offsets: glam::Vec4,
    colors: glam::Vec4,
    alphas: glam::Vec4,
    stop_count: i32,
}

#[derive(Clone)]
#[repr(C)]
struct GpuPath {
    uv_data: [f32; 4], // scaleu,offsetu,scalev,offsetv
    pos: glam::Vec3,
    scale: glam::Vec2,
    fill_color: [f32; 2], //rgb,alpha
    linear: GpuLinearGrad,
}

struct MsdfData {
    msdf: Vec<f32>,
    msdf_w: i32,
    msdf_h: i32,
    gpu_paths: Vec<GpuPath>,
}

fn handle_style(path: &PathData) -> ([f32; 2], GpuLinearGrad) {
    let fill_color = {
        let fill = &path.fill_color;
        match fill {
            Some(color) => [color.pack_to_4bytes(), color.a],
            None => [0.0, 1.0],
        }
    };

    let linear = match path.linear {
        Some(ref lg) => {
            if lg.stops.len() > 4 {
                println!("WARN: only 4 linear gradient offsets are supported");
            }
            let offset_count = lg.stops.len().min(4);

            let mut stop_offsets = glam::Vec4::default();
            let mut stop_colors = glam::Vec4::default();
            let mut stop_alphas = glam::Vec4::default();
            let mut stop_count = 0;

            //#1
            {
                let stop = lg.stops[0];
                stop_offsets.x = stop.offset.get() as f32;
                stop_colors.x =
                    Color::pack_to_4bytes2(stop.color.red, stop.color.green, stop.color.blue);
                stop_alphas.x = stop.opacity.get() as f32;
                stop_count += 1;
            }
            //#2
            {
                let stop = lg.stops[1];
                stop_offsets.y = stop.offset.get() as f32;
                stop_colors.y =
                    Color::pack_to_4bytes2(stop.color.red, stop.color.green, stop.color.blue);
                stop_alphas.y = stop.opacity.get() as f32;
                stop_count += 1;

                // if it's the last one
                if offset_count == 2 && stop_offsets.y != 1.0 {
                    stop_offsets.z = 1.0;
                    stop_colors.z =
                        Color::pack_to_4bytes2(stop.color.red, stop.color.green, stop.color.blue);
                    stop_alphas.z = stop.opacity.get() as f32;
                    stop_count += 1;
                }
            }
            //#3
            if offset_count > 2 {
                let stop = lg.stops[2];
                stop_offsets.z = stop.offset.get() as f32;
                stop_colors.z =
                    Color::pack_to_4bytes2(stop.color.red, stop.color.green, stop.color.blue);
                stop_alphas.z = stop.opacity.get() as f32;
                stop_count += 1;
            }

            //#4
            if offset_count > 3 {
                let stop = lg.stops[3];
                stop_offsets.w = stop.offset.get() as f32;
                stop_colors.w =
                    Color::pack_to_4bytes2(stop.color.red, stop.color.green, stop.color.blue);
                stop_alphas.w = stop.opacity.get() as f32;
                stop_count += 1;
            }

            GpuLinearGrad {
                bbox: lg.bbox,
                offsets: stop_offsets,
                colors: stop_colors,
                alphas: stop_alphas,
                stop_count,
            }
        }
        None => GpuLinearGrad::default(),
    };

    (fill_color, linear)
}

fn pack_paths_msdfs(paths_data: &Vec<PathData>) -> MsdfData {
    use rect_packer::DensePacker;

    // TODO: pack it with a better algorithm
    let mut packer_size = 64u64;

    let mut packer = DensePacker::new(packer_size as i32, packer_size as i32);

    let mut rects = Vec::new();
    for path in paths_data.iter() {
        while !packer.can_pack(path.w, path.h, false) {
            packer_size = (packer_size + 1 as u64).next_power_of_two();
            packer.resize(packer_size as i32, packer_size as i32);
        }

        let rect = packer.pack(path.w, path.h, false).unwrap();
        rects.push(rect)
    }
    let pixel_size = 3 * core::mem::size_of::<f32>();

    let mut msdf: Vec<f32> = vec![0.; (packer_size * packer_size) as usize * 3];

    let mut gpu_paths = Vec::with_capacity(paths_data.len());

    let mut current_z_index = 0.0;

    for (i, path) in paths_data.iter().enumerate() {
        let r = rects[i];

        let pixels = msdfgen::bitmapf3_pixels(&path.msdf.as_ref().unwrap());

        // TODO: speed this up
        for y in 0..path.h as usize {
            for x in 0..path.w as usize {
                let m = (r.y as usize + y) * 3 * packer_size as usize + (r.x as usize + x) * 3;

                msdf[m + 0] = pixels[y * path.w as usize * 3 + x * 3 + 0];
                msdf[m + 1] = pixels[y * path.w as usize * 3 + x * 3 + 1];
                msdf[m + 2] = pixels[y * path.w as usize * 3 + x * 3 + 2];
            }
        }

        save_png(packer_size as u32, packer_size as u32, msdf.as_slice());

        let uv_data = {
            let offx = (r.x + path.border) as f32 / packer_size as f32;
            let offy = (r.y + path.border) as f32 / packer_size as f32;
            let sx = (path.w - path.border * 2) as f32 / packer_size as f32;
            let sy = (path.h - path.border * 2) as f32 / packer_size as f32;
            [sx, offx, sy, offy]
        };

        let (pos, scale) = {
            let w = path.bbox.x1 - path.bbox.x0;
            let h = path.bbox.y1 - path.bbox.y0;

            let scalex = w * 0.5;
            let scaley = h * 0.5;

            let pos_x = path.bbox.x0 + w * 0.5;
            let pos_y = path.bbox.y0 + h * 0.5;

            let pos = glam::vec3(pos_x, pos_y, current_z_index);
            let scale = glam::vec2(scalex, scaley);

            (pos, scale)
        };

        let (fill_color, linear_gradient) = handle_style(path);

        gpu_paths.push(GpuPath {
            uv_data,
            fill_color,
            pos,
            scale,
            linear: linear_gradient,
        });
        current_z_index += 0.01;
    }

    MsdfData {
        msdf,
        msdf_w: packer_size as i32,
        msdf_h: packer_size as i32,
        gpu_paths,
    }
}

fn svg_to_paths() -> SvgData {
    let svg = fs::read_to_string("input.svg").expect("couldn't read svg file");

    println!("svg :{:?} \n \n", svg);

    let tree = usvg::Tree::from_str(&svg, &usvg::Options::default()).unwrap();
    println!("tree.view_box :{:?}\n\n", tree.view_box);
    println!("svg image size:{:?}\n\n", tree.size);

    #[derive(Debug, Clone, Copy)]
    struct Point2 {
        x: f64,
        y: f64,
    }

    #[derive(Debug, Clone, Copy)]
    struct Line2 {
        start: Point2,
        end: Point2,
    }
    #[derive(Debug, Clone)]
    struct Contour {
        lines: Vec<Line2>,
    }

    // to [0-1]
    let normalize = |v: f64, min: f64, max: f64| -> f64 { (v - min) / (max - min) };

    let bitmap_size = 92;
    let border = 1;
    let pxRange = 8.0;

    let mut paths_data: Vec<PathData> = Vec::new();
    let mut svg_bbox = BBox {
        x0: f32::MAX,
        x1: f32::MIN,
        y0: f32::MAX,
        y1: f32::MIN,
    };

    let s = tree.root.children().into_iter().count();

    for node in tree.root.children() {
        println!("node :{:?}\n\n", node);

        match *node.borrow() {
            usvg::NodeKind::Path(ref path) => {
                // shape data

                let mut bitmap_w = bitmap_size;
                let mut bitmap_h = bitmap_size;

                // let mut shape = msdfgen::new_shape();
                // let mut current_contour = msdfgen::new_contour(shape.pin_mut());
                let mut current_point = Point2 { x: 0.0, y: 0.0 };
                let mut start_point = None;
                //

                let points = path.data.points();
                let mut points_index = 0;

                let bbox = if let Some(bbox) = path.data.bbox() {
                    bbox
                } else {
                    println!("warn!!! path doesn't have bbox");
                    continue;
                };

                let bboxw = bbox.width();
                let bboxh = bbox.height();

                if bboxw > bboxh {
                    let scaley = bboxh / bboxw;
                    bitmap_h = (bitmap_h as f64 * scaley).round() as i32;
                } else if bboxw < bboxh {
                    let scalex = bboxw / bboxh;
                    bitmap_w = (bitmap_w as f64 * scalex).round() as i32;
                }

                let miny = bbox.y();
                let maxy = bbox.y() + bboxh;
                let minx = bbox.x();
                let maxx = bbox.x() + bboxw;

                svg_bbox.x0 = svg_bbox.x0.min(minx as f32);
                svg_bbox.x1 = svg_bbox.x1.max(maxx as f32);
                svg_bbox.y0 = svg_bbox.y0.min(miny as f32);
                svg_bbox.y1 = svg_bbox.y1.max(maxy as f32);

                let mut contours = Vec::new();
                let mut contour = Contour { lines: vec![] };

                use kurbo::{flatten, BezPath, CubicBez, Point, Rect, Shape, Vec2};
                let mut init_path = BezPath::new();

                for cmd in path.data.commands() {
                    // println!("cmd : {:?}", cmd);
                    match cmd {
                        usvg::PathCommand::MoveTo => {
                            current_point = Point2 {
                                x: points[points_index + 0],
                                y: points[points_index + 1],
                            };
                            // println!("{:?} ,{:?}", current_point.x, current_point.y);
                            current_point.x =
                                normalize(current_point.x, minx, maxx) * (bitmap_w as f64) as f64;
                            current_point.y =
                                normalize(current_point.y, miny, maxy) * (bitmap_h as f64) as f64;

                            init_path.move_to(Point::new(
                                current_point.x as f64,
                                current_point.y as f64,
                            ));

                            points_index += 2;
                            if start_point.is_none() {
                                start_point = Some(current_point);
                            }
                        }
                        usvg::PathCommand::LineTo => {
                            // if current_point.x == points[points_index + 0]
                            //     && current_point.y == points[points_index + 1]
                            // {
                            //     let kk = 1;
                            //     continue;
                            // }

                            // let p0 =
                            //     msdfgen::new_point2(current_point.x as f64, current_point.y as f64);
                            let start = current_point;

                            current_point = Point2 {
                                x: points[points_index + 0],
                                y: points[points_index + 1],
                            };

                            // println!("{:?} ,{:?}", current_point.x, current_point.y);
                            current_point.x =
                                normalize(current_point.x, minx, maxx) * (bitmap_w) as f64;
                            current_point.y =
                                normalize(current_point.y, miny, maxy) * (bitmap_h as f64) as f64;
                            points_index += 2;

                            // if !init_path.contains(Point::new(
                            //     current_point.x as f64,
                            //     current_point.y as f64,
                            // )) {
                            init_path.line_to(Point::new(
                                current_point.x as f64,
                                current_point.y as f64,
                            ));
                            // }

                            // contour.lines.push(Line2 {
                            //     start,
                            //     end: current_point,
                            // });

                            // let p1 =
                            //     msdfgen::new_point2(current_point.x as f64, current_point.y as f64);

                            // msdfgen::add_linear_segment(
                            //     current_contour.as_mut(),
                            //     p0.as_ref().unwrap(),
                            //     p1.as_ref().unwrap(),
                            // );

                            // contour.lines.push(Line::from(
                            //     point(current_point.x as f32, current_point.y as f32),
                            //     point(current_point.x as f32, current_point.y as f32),
                            // ));
                        }
                        usvg::PathCommand::CurveTo => {
                            debug_assert!(points.len() > points_index + 5);
                            use lyon::math::point;
                            use lyon::path::Path;

                            let p0 = point(current_point.x as f32, current_point.y as f32);
                            let mut c0 = Point2 {
                                x: points[points_index + 0],
                                y: points[points_index + 1],
                            };
                            // print!("c0:{:?},{:?} ", c0.x, c0.y);
                            c0.x = normalize(c0.x, minx, maxx) * (bitmap_w) as f64;
                            c0.y = normalize(c0.y, miny, maxy) * (bitmap_h as f64) as f64;
                            points_index += 2;

                            let mut c1 = Point2 {
                                x: points[points_index + 0],
                                y: points[points_index + 1],
                            };
                            // print!("c1:{:?},{:?} ", c1.x, c1.y);
                            c1.x = normalize(c1.x, minx, maxx) * (bitmap_w) as f64;
                            c1.y = normalize(c1.y, miny, maxy) * (bitmap_h as f64) as f64;
                            points_index += 2;

                            current_point = Point2 {
                                x: points[points_index + 0],
                                y: points[points_index + 1],
                            };

                            // println!("p2:{:?},{:?} ", current_point.x, current_point.y);
                            current_point.x =
                                normalize(current_point.x, minx, maxx) * (bitmap_w) as f64;
                            current_point.y =
                                normalize(current_point.y, miny, maxy) * (bitmap_h as f64) as f64;
                            points_index += 2;

                            // let bez = lyon::geom::CubicBezierSegment {
                            //     from: p0,
                            //     ctrl1: point(c0.x as f32, c0.y as f32),
                            //     ctrl2: point(c1.x as f32, c1.y as f32),
                            //     to: point(current_point.x as f32, current_point.y as f32),
                            // };

                            // let lines: Vec<Point2D<f32, UnknownUnit>> =
                            //     bez.flattened(5.0).collect();

                            // path.move_to(Point::new(p0.x as f64, p0.y as f64));
                            init_path.curve_to(
                                Point::new(c0.x, c0.y),
                                Point::new(c1.x, c1.y),
                                Point::new(current_point.x, current_point.y),
                            );

                            // for line in lines.windows(2) {
                            //     let start = line[0];
                            //     let end = line.get(1).unwrap(); //.unwrap_or(&start);

                            //     contour.lines.push(Line2 {
                            //         start: Point2 {
                            //             x: start.x as f64,
                            //             y: start.y as f64,
                            //         },

                            //         end: Point2 {
                            //             x: end.x as f64,
                            //             y: end.y as f64,
                            //         },
                            //     });
                            // }

                            // let last = lines.last().unwrap();
                            // current_point = Point2 {
                            //     x: last.x as f64,
                            //     y: last.y as f64,
                            // };
                            // current_point.x =
                            //     normalize(current_point.x, minx, maxx) * (bitmap_w) as f64;
                            // current_point.y =
                            //     normalize(current_point.y, miny, maxy) * (bitmap_h as f64) as f64;

                            // let path = builder.build();

                            // let p0 =
                            //     msdfgen::new_point2(current_point.x as f64, current_point.y as f64);

                            // let p1 =
                            //     msdfgen::new_point2(current_point.x as f64, current_point.y as f64);

                            // let c0 = msdfgen::new_point2(c0.x as f64, c0.y as f64);
                            // let c1 = msdfgen::new_point2(c1.x as f64, c1.y as f64);

                            // msdfgen::add_cubic_segment(
                            //     current_contour.as_mut(),
                            //     p0.as_ref().unwrap(),
                            //     c0.as_ref().unwrap(),
                            //     c1.as_ref().unwrap(),
                            //     p1.as_ref().unwrap(),
                            // );
                        }
                        usvg::PathCommand::ClosePath => {
                            init_path.close_path();

                            let mut start1 = Point::default();
                            init_path.flatten(0.025, |p| match p {
                                kurbo::PathEl::MoveTo(p) => {
                                    start1 = p;
                                }
                                kurbo::PathEl::LineTo(end) => {
                                    contour.lines.push(Line2 {
                                        start: Point2 {
                                            x: start1.x as f64,
                                            y: start1.y as f64,
                                        },

                                        end: Point2 {
                                            x: end.x as f64,
                                            y: end.y as f64,
                                        },
                                    });

                                    start1 = end;
                                }
                                kurbo::PathEl::QuadTo(_, _) => todo!(),
                                kurbo::PathEl::CurveTo(_, _, _) => todo!(),
                                kurbo::PathEl::ClosePath => {
                                    let s = contour.lines[0].start;

                                    if s.x != start1.x && s.y != start1.y {
                                        contour.lines.push(Line2 {
                                            start: Point2 {
                                                x: start1.x as f64,
                                                y: start1.y as f64,
                                            },
                                            end: s,
                                        });
                                    }
                                }
                            });

                            contours.push(contour);

                            contour = Contour { lines: vec![] };

                            init_path = BezPath::new();

                            // if current_point.x == p1.x && current_point.y == p1.y {
                            //     // start a new contour
                            //     current_contour = msdfgen::new_contour(shape.pin_mut());
                            //     start_point = None;
                            //     continue;
                            // }

                            // let p0 =
                            //     msdfgen::new_point2(current_point.x as f64, current_point.y as f64);

                            // let p1 = msdfgen::new_point2(p1.x as f64, p1.y as f64);

                            // msdfgen::add_linear_segment(
                            //     current_contour.as_mut(),
                            //     p0.as_ref().unwrap(),
                            //     p1.as_ref().unwrap(),
                            // );

                            // start a new contour
                            // current_contour = msdfgen::new_contour(shape.pin_mut());
                            start_point = None;
                        }
                    }
                }

                // styling
                // println!("path.fill : {:?}", path.fill);
                let (fill_color, linear) = match &path.fill {
                    Some(fill) => match fill.paint {
                        usvg::Paint::Color(color) => (
                            Some(Color {
                                r: color.red,
                                g: color.green,
                                b: color.blue,
                                a: fill.opacity.get() as f32,
                            }),
                            None,
                        ),
                        usvg::Paint::LinearGradient(ref lg) => {
                            if lg.stops.len() == 1 {
                                let c = lg.stops[0].color;
                                (
                                    Some(Color {
                                        r: c.red,
                                        g: c.green,
                                        b: c.blue,
                                        a: lg.stops[0].opacity.get() as f32,
                                    }),
                                    None,
                                )
                            } else {
                                let mat = glam::mat3(
                                    glam::vec3(lg.transform.a as f32, lg.transform.b as f32, 0.),
                                    glam::vec3(lg.transform.c as f32, lg.transform.d as f32, 0.),
                                    glam::vec3(lg.transform.e as f32, lg.transform.f as f32, 1.),
                                );
                                let start =
                                    mat.transform_vector2(glam::vec2(lg.x1 as f32, lg.y1 as f32));

                                let end =
                                    mat.transform_vector2(glam::vec2(lg.x2 as f32, lg.y2 as f32));

                                let linear_gradient_bbox =
                                    glam::vec4(start.x, end.x, start.y, end.y);

                                (
                                    None,
                                    Some(LinearGrad {
                                        bbox: linear_gradient_bbox,
                                        stops: lg.stops.clone(),
                                    }),
                                )
                            }
                        }
                        usvg::Paint::RadialGradient(_) => todo!(),
                        usvg::Paint::Pattern(_) => todo!(),
                    },
                    None => (None, None),
                };

                if !init_path.is_empty() {
                    let mut start1 = Point::default();
                    init_path.flatten(0.025, |p| match p {
                        kurbo::PathEl::MoveTo(p) => {
                            start1 = p;
                        }
                        kurbo::PathEl::LineTo(end) => {
                            contour.lines.push(Line2 {
                                start: Point2 {
                                    x: start1.x as f64,
                                    y: start1.y as f64,
                                },

                                end: Point2 {
                                    x: end.x as f64,
                                    y: end.y as f64,
                                },
                            });

                            start1 = end;
                        }
                        kurbo::PathEl::QuadTo(_, _) => todo!(),
                        kurbo::PathEl::CurveTo(_, _, _) => todo!(),
                        kurbo::PathEl::ClosePath => {
                            let s = contour.lines[0].start;

                            if s.x != start1.x && s.y != start1.y {
                                contour.lines.push(Line2 {
                                    start: Point2 {
                                        x: start1.x as f64,
                                        y: start1.y as f64,
                                    },
                                    end: s,
                                });
                            }
                        }
                    });
                    contours.push(contour);
                }

                use geo::orient::{Direction, Orient};
                use geo::polygon;
                use geo_booleanop::boolean::BooleanOp;

                let mut shapes = Vec::new();
                shapes.push(msdfgen::new_shape());

                let mut current_shape_index = 0;

                let mut lines = vec![];

                for contour in contours.iter() {
                    ///
                    ///
                    ///
                    ///
                    for line1 in contour.lines.iter() {
                        for line2 in contour.lines.iter() {
                            if (line1.start.x != line2.start.x && line1.start.y != line2.start.y)
                                && (line1.end.x != line2.end.x && line1.end.y != line2.end.y)
                            {
                                let seg = kurbo::PathSeg::Line(kurbo::Line::new(
                                    (line1.start.x, line1.start.y),
                                    (line1.end.x, line1.end.y),
                                ));
                                let line = kurbo::Line::new(
                                    (line2.start.x, line2.start.y),
                                    (line2.end.x, line2.end.y),
                                );
                                let intersection = seg.intersect_line(line);

                                if !intersection.is_empty() {
                                    println!("asd :{:?}", intersection);

                                    let inter = intersection[0].segment_t;
                                } else {
                                    let l = kurbo::Line::new(
                                        (line1.start.x, line1.start.y),
                                        (line1.end.x, line1.end.y),
                                    );
                                    if !lines.contains(&l) {
                                        lines.push(l);
                                    }

                                    let l = kurbo::Line::new(
                                        (line2.start.x, line2.start.y),
                                        (line2.end.x, line2.end.y),
                                    );
                                    if !lines.contains(&l) {
                                        lines.push(l);
                                    }
                                }
                            } else {
                                lines.push(kurbo::Line::new(
                                    (line1.start.x, line1.start.y),
                                    (line1.end.x, line1.end.y),
                                ))
                            }
                        }
                    }

                    let mut current_contour =
                        msdfgen::new_contour(shapes[current_shape_index].pin_mut());

                    println!("contour lines {:?}", contour.lines);
                    // for poly in inter.iter() {
                    for line in contour.lines.iter() {
                        let start = line.start;
                        let end = line.end;
                        let p0 = msdfgen::new_point2(start.x as f64, start.y as f64);

                        let p1 = msdfgen::new_point2(end.x as f64, end.y as f64);

                        msdfgen::add_linear_segment(
                            current_contour.as_mut(),
                            p1.as_ref().unwrap(),
                            p0.as_ref().unwrap(),
                        );
                    }
                }

                // generate msdf
                for shape in shapes.iter_mut() {
                    shape.as_mut().unwrap().normalize();

                    msdfgen::edge_coloring_simple(shape.pin_mut(), 3.0);

                    let w = bitmap_w + border * 2;
                    let h = bitmap_h + border * 2;
                    let mut bitmap = msdfgen::new_bitmapf3(w, h);

                    // println!(
                    //     "created bitmap , w:{:},h:{:}",
                    //     bitmap.width(),
                    //     bitmap.height()
                    // );

                    // let start = Instant::now();

                    msdfgen::generate_msdf(
                        bitmap.pin_mut(),
                        shape.as_ref().unwrap(),
                        pxRange,
                        1.0,
                        1.0,
                        border as f64,
                        border as f64,
                    );

                    msdfgen::shape_guess_orientation(shape.as_ref().unwrap(), bitmap.pin_mut());

                    // let duration = start.elapsed();
                    // println!("generated msdf: {:?}", duration);
                    // save_msdf(&bitmap);
                    paths_data.push(PathData {
                        msdf: bitmap,
                        w,
                        h,
                        border,
                        fill_color: fill_color.clone(),
                        bbox: BBox {
                            x0: minx as f32,
                            x1: maxx as f32,
                            y0: miny as f32,
                            y1: maxy as f32,
                        },
                        linear: linear.clone(),
                    });
                }
            }

            _ => println!("node :{:?}\n\n", node),
        }
    }

    // println!("paths count :{:?}", paths_data.len());

    // map path bbox to [-1 - 1] relative to svg bbox

    fn map_range(from_range: (f64, f64), to_range: (f64, f64), s: f64) -> f64 {
        to_range.0 + (s - from_range.0) * (to_range.1 - to_range.0) / (from_range.1 - from_range.0)
    }

    let from_x = (svg_bbox.x0 as f64, svg_bbox.x1 as f64);

    let from_y = (svg_bbox.y0 as f64, svg_bbox.y1 as f64);

    let to = (-1.0, 1.0);

    for path in paths_data.iter_mut() {
        let scale_x = (svg_bbox.x1 - svg_bbox.x0) / (svg_bbox.y1 - svg_bbox.y0);

        path.bbox.x0 = map_range(from_x, to, path.bbox.x0 as f64) as f32 * scale_x;
        path.bbox.x1 = map_range(from_x, to, path.bbox.x1 as f64) as f32 * scale_x;
        path.bbox.y0 = map_range(from_y, to, path.bbox.y0 as f64) as f32;
        path.bbox.y1 = map_range(from_y, to, path.bbox.y1 as f64) as f32;
    }

    //

    SvgData {
        paths: paths_data,
        size: tree.size,
        svg_bbox,
    }
}

fn save_png(w: u32, h: u32, pixels: &[f32]) {
    use image::RgbImage;

    let mut raw_pixels = Vec::with_capacity(pixels.len());

    for p in pixels {
        let v = (*p * 256.0).clamp(0.0, 255.0);
        raw_pixels.push(v as u8)
    }

    let rgb = RgbImage::from_raw(w, h, raw_pixels).unwrap();
    rgb.save("output.png").unwrap();
}

fn save_msdf(bitmap: &UniquePtr<Bitmapf3>) {
    let start = Instant::now();
    let pixels = msdfgen::bitmapf3_pixels(&bitmap.as_ref().unwrap());
    save_png(bitmap.width() as u32, bitmap.height() as u32, pixels);
    let duration = start.elapsed();
    println!("saved png : {:?}", duration);
}

struct ProgramGL {
    program: NativeProgram,
}

impl ProgramGL {
    fn new(
        gl: &Context,
        vertex_shader_source: &str,
        fragment_shader_source: &str,
        shader_version: &str,
    ) -> ProgramGL {
        unsafe {
            let shader_sources = [
                (glow::VERTEX_SHADER, vertex_shader_source),
                (glow::FRAGMENT_SHADER, fragment_shader_source),
            ];

            let program = gl.create_program().expect("Cannot create program");

            let mut shaders = Vec::with_capacity(shader_sources.len());

            for (shader_type, shader_source) in shader_sources.iter() {
                let shader = gl
                    .create_shader(*shader_type)
                    .expect("Cannot create shader");
                gl.shader_source(shader, &format!("{}\n{}", shader_version, shader_source));
                gl.compile_shader(shader);
                if !gl.get_shader_compile_status(shader) {
                    panic!("{}", gl.get_shader_info_log(shader));
                }
                gl.attach_shader(program, shader);
                shaders.push(shader);
            }

            gl.link_program(program);
            if !gl.get_program_link_status(program) {
                panic!("{}", gl.get_program_info_log(program));
            }

            for shader in shaders {
                gl.detach_shader(program, shader);
                gl.delete_shader(shader);
            }

            ProgramGL { program }
        }
    }

    pub unsafe fn get_attrib_location(&self, gl: &Context, attrib: &str) -> u32 {
        gl.get_attrib_location(self.program, attrib).unwrap()
    }

    pub unsafe fn get_uniform_location(&self, gl: &Context, name: &str) -> NativeUniformLocation {
        gl.get_uniform_location(self.program, name).unwrap()
    }

    pub unsafe fn Use(&self, gl: &Context) {
        gl.use_program(Some(self.program));
    }
}

pub struct VertexArray {
    pub vao: NativeVertexArray,
}
impl VertexArray {
    pub unsafe fn new(gl: &Context) -> Self {
        let vao = gl.create_vertex_array().unwrap();
        Self { vao }
    }

    pub unsafe fn bind(&self, gl: &Context) {
        gl.bind_vertex_array(Some(self.vao));
    }
    pub unsafe fn set_attribute(
        &self,
        gl: &Context,
        attrib_pos: u32,
        components_count: i32,
        data_type: u32,
        stride: usize,
        offset: usize,
    ) {
        self.bind(gl);
        gl.vertex_attrib_pointer_f32(
            attrib_pos,
            components_count,
            data_type,
            false,
            stride as i32,
            offset as i32,
        );
        gl.enable_vertex_attrib_array(attrib_pos);
    }
    pub unsafe fn set_attribute_i32(
        &self,
        gl: &Context,
        attrib_pos: u32,
        components_count: i32,
        data_type: u32,
        stride: usize,
        offset: usize,
    ) {
        self.bind(gl);
        gl.vertex_attrib_pointer_i32(
            attrib_pos,
            components_count,
            data_type,
            stride as i32,
            offset as i32,
        );
        gl.enable_vertex_attrib_array(attrib_pos);
    }
}

fn create_texture(gl: &Context, pixels: &[u8], width: i32, height: i32) -> glow::NativeTexture {
    unsafe {
        let tex = gl.create_texture().unwrap();
        gl.bind_texture(glow::TEXTURE_2D, Some(tex));

        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::REPEAT as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::REPEAT as i32);
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_MIN_FILTER,
            glow::LINEAR as i32,
        );
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_MAG_FILTER,
            glow::LINEAR as i32,
        );

        gl.tex_image_2d(
            glow::TEXTURE_2D,
            0,
            glow::RGB as i32,
            width,
            height,
            0,
            glow::RGB,
            glow::FLOAT,
            Some(pixels),
        );
        tex
    }
    // glGenerateMipmap(GL_TEXTURE_2D);
}

#[repr(C, packed)]
struct Vertex {
    pos: [f32; 3],
    uv: [f32; 2],
}

#[rustfmt::skip]
const VERTICES: [Vertex; 4] = [
    Vertex{pos : [-1.0, -1.0,0.0], uv: [0.0, 1.0]},
    Vertex{pos :[ 1.0, -1.0,0.0],  uv:[1.0, 1.0]},
    Vertex{pos:[ 1.0,  1.0,0.0], uv: [1.0, 0.0]},
    Vertex{pos:[-1.0,  1.0,0.0], uv: [0.0, 0.0]},
];

#[rustfmt::skip]
const INDICES: [i32; 6] = [
    0, 1, 2,
    2, 3, 0
];

fn main() {
    unsafe {
        let win_width = 1024.0;
        let win_height = 1024.0;
        let (gl, shader_version, window, event_loop) = {
            let event_loop = glutin::event_loop::EventLoop::new();
            let window_builder = glutin::window::WindowBuilder::new()
                .with_title("Hello triangle!")
                .with_inner_size(glutin::dpi::LogicalSize::new(win_width, win_height));

            let window = {
                glutin::ContextBuilder::new()
                    .with_vsync(false)
                    .build_windowed(window_builder, &event_loop)
                    .unwrap()
                    .make_current()
                    .unwrap()
            };
            let gl =
                { glow::Context::from_loader_function(|s| window.get_proc_address(s) as *const _) };

            gl.debug_message_callback(|source: u32, ty: u32, id: u32, severity: u32, msg: &str| {
                if ty == glow::DEBUG_TYPE_ERROR || ty == glow::DEBUG_TYPE_PERFORMANCE {
                    println!("opengl error/warn :{:?}", msg);
                }
            });

            // glfwSwapInterval(0);
            gl.enable(glow::DEBUG_OUTPUT);
            // println!("error : {:#X}", gl.get_error());
            // panic!();
            (gl, "#version 410", window, event_loop)
        };

        let mut svg_data = svg_to_paths();

        println!("paths count :{:?}", svg_data.paths.len());

        let mut msdf_data = pack_paths_msdfs(&mut svg_data.paths);

        // let mut duplicate = Vec::new();
        // for i in 0..1000 {
        //     duplicate.extend_from_slice(msdf_data.gpu_paths.as_slice());
        // }
        // msdf_data.gpu_paths = duplicate;

        // debugging only
        {
            let ss2 = msdf_data.msdf_w * msdf_data.msdf_h * 3;
            let s: &[f32] = slice::from_raw_parts(msdf_data.msdf.as_ptr().cast(), ss2 as usize);

            save_png(msdf_data.msdf_w as u32, msdf_data.msdf_h as u32, s);
        }

        let (vertex_shader_source, fragment_shader_source) = {
            let vs = fs::read_to_string("./svg-renderer/shaders/simple.vs")
                .expect("couldn't read svg file");
            let fs_shader = fs::read_to_string("./svg-renderer/shaders/simple.fs")
                .expect("couldn't read svg file");

            (vs, fs_shader)
        };

        let program = ProgramGL::new(
            &gl,
            vertex_shader_source.as_str(),
            fragment_shader_source.as_str(),
            shader_version,
        );
        program.Use(&gl);
        // per-vertex
        let pos_attrib = program.get_attrib_location(&gl, "position");
        let uv_attrib = program.get_attrib_location(&gl, "uv");

        // uniforms
        let tex_loc = program.get_uniform_location(&gl, "texture0");

        // let pxRange_loc = program.get_uniform_location(&gl, "pxRange");

        let mut pos = glam::vec3(0.0, 0.0, 0.0);
        let mut scale = glam::vec3(0.5, 0.5, 1.0);

        let update_cam = |gl: &Context, program: &ProgramGL, pos: &Vec3, scale: &Vec3| {
            let vp_loc = program.get_uniform_location(&gl, "vp");
            let proj = glam::Mat4::orthographic_rh(-1.0, 1.0, 1.0, -1.0, 0.0, 1.0);
            let cam = glam::Mat4::from_translation(*pos) * glam::Mat4::from_scale(*scale);
            let vp = (cam * proj).to_cols_array();
            gl.uniform_matrix_4_f32_slice(Some(&vp_loc), false, &vp);
        };
        update_cam(&gl, &program, &pos, &scale);

        let vao = VertexArray::new(&gl);
        vao.bind(&gl);

        let vertex_buffer = gl.create_buffer().unwrap();
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(vertex_buffer));
        let vertex_data: &[u8] =
            slice::from_raw_parts(VERTICES.as_ptr().cast(), std::mem::size_of_val(&VERTICES));
        gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, vertex_data, glow::STATIC_DRAW);

        let index_buffer = gl.create_buffer().unwrap();
        gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(index_buffer));
        let index_data: &[u8] =
            slice::from_raw_parts(INDICES.as_ptr().cast(), std::mem::size_of_val(&INDICES));
        gl.buffer_data_u8_slice(glow::ELEMENT_ARRAY_BUFFER, index_data, glow::STATIC_DRAW);

        // set vertex attrib
        vao.set_attribute(
            &gl,
            pos_attrib,
            3,
            glow::FLOAT,
            std::mem::size_of::<Vertex>(),
            offset_of!(Vertex, pos),
        );
        vao.set_attribute(
            &gl,
            uv_attrib,
            2,
            glow::FLOAT,
            std::mem::size_of::<Vertex>(),
            offset_of!(Vertex, uv),
        );

        // set instance attrib
        //
        let pos_attrib = program.get_attrib_location(&gl, "a_pos");
        let scale_attrib = program.get_attrib_location(&gl, "a_scale");
        let uv_data_attrib = program.get_attrib_location(&gl, "a_uv_data");
        let fill_color_attrib = program.get_attrib_location(&gl, "a_fill_color");
        // linear gradient data
        let a_linear_gradient_bbox = program.get_attrib_location(&gl, "a_linear_gradient_bbox");
        let a_stop_offsets = program.get_attrib_location(&gl, "a_stop_offsets");
        let a_stop_colors = program.get_attrib_location(&gl, "a_stop_colors");
        let a_stop_alphas = program.get_attrib_location(&gl, "a_stop_alphas");
        let a_stop_count = program.get_attrib_location(&gl, "a_stop_count");

        {
            let gpu_paths_buffer = gl.create_buffer().unwrap();
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(gpu_paths_buffer));
            let gpu_paths_data: &[u8] = slice::from_raw_parts(
                msdf_data.gpu_paths.as_ptr().cast(),
                std::mem::size_of_val(msdf_data.gpu_paths.as_slice()),
            );
            gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, gpu_paths_data, glow::STATIC_DRAW);

            // "a_pos"
            vao.set_attribute(
                &gl,
                pos_attrib,
                3,
                glow::FLOAT,
                std::mem::size_of::<GpuPath>(),
                offset_of!(GpuPath, pos),
            );
            gl.vertex_attrib_divisor(pos_attrib, 1);

            // "a_scale"
            vao.set_attribute(
                &gl,
                scale_attrib,
                2,
                glow::FLOAT,
                std::mem::size_of::<GpuPath>(),
                offset_of!(GpuPath, scale),
            );
            gl.vertex_attrib_divisor(scale_attrib, 1);

            // "uv_data"
            vao.set_attribute(
                &gl,
                uv_data_attrib,
                4,
                glow::FLOAT,
                std::mem::size_of::<GpuPath>(),
                offset_of!(GpuPath, uv_data),
            );
            gl.vertex_attrib_divisor(uv_data_attrib, 1);

            // "fill_color"
            vao.set_attribute(
                &gl,
                fill_color_attrib,
                2,
                glow::FLOAT,
                std::mem::size_of::<GpuPath>(),
                offset_of!(GpuPath, fill_color),
            );
            gl.vertex_attrib_divisor(fill_color_attrib, 1);

            // linear gradient
            {
                // "a_linear_gradient_bbox"
                vao.set_attribute(
                    &gl,
                    a_linear_gradient_bbox,
                    4,
                    glow::FLOAT,
                    std::mem::size_of::<GpuPath>(),
                    offset_of!(GpuPath, linear) + offset_of!(GpuLinearGrad, bbox),
                );
                gl.vertex_attrib_divisor(a_linear_gradient_bbox, 1);

                // "a_stop_offsets"
                vao.set_attribute(
                    &gl,
                    a_stop_offsets,
                    4,
                    glow::FLOAT,
                    std::mem::size_of::<GpuPath>(),
                    offset_of!(GpuPath, linear) + offset_of!(GpuLinearGrad, offsets),
                );
                gl.vertex_attrib_divisor(a_stop_offsets, 1);

                // "a_stop_colors"
                vao.set_attribute(
                    &gl,
                    a_stop_colors,
                    4,
                    glow::FLOAT,
                    std::mem::size_of::<GpuPath>(),
                    offset_of!(GpuPath, linear) + offset_of!(GpuLinearGrad, colors),
                );
                gl.vertex_attrib_divisor(a_stop_colors, 1);

                // "a_stop_alphas"
                vao.set_attribute(
                    &gl,
                    a_stop_alphas,
                    4,
                    glow::FLOAT,
                    std::mem::size_of::<GpuPath>(),
                    offset_of!(GpuPath, linear) + offset_of!(GpuLinearGrad, alphas),
                );
                gl.vertex_attrib_divisor(a_stop_alphas, 1);

                // "a_stop_count"
                vao.set_attribute_i32(
                    &gl,
                    a_stop_count,
                    1,
                    glow::INT,
                    std::mem::size_of::<GpuPath>(),
                    offset_of!(GpuPath, linear) + offset_of!(GpuLinearGrad, stop_count),
                );
                gl.vertex_attrib_divisor(a_stop_count, 1);
            }
        }
        // set texture
        let pixels: &[u8] = unsafe {
            slice::from_raw_parts(
                msdf_data.msdf.as_ptr().cast(),
                msdf_data.msdf.len() * core::mem::size_of::<f32>(),
            )
        };
        let texture = create_texture(&gl, pixels, msdf_data.msdf_w, msdf_data.msdf_h);
        gl.bind_texture(glow::TEXTURE_2D, Some(texture));
        gl.active_texture(glow::TEXTURE0);

        // set uniforms
        gl.uniform_1_i32(Some(&tex_loc), 0);

        // general stuff
        gl.clear_color(0.1, 0.2, 0.3, 1.0);
        gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);
        gl.enable(glow::BLEND);
        gl.enable(glow::DEPTH_TEST);
        gl.depth_func(glow::LEQUAL);
        // gl.disable(glow::CULL_FACE);

        //timing
        // let start_query = gl.create_query().unwrap();

        event_loop.run(move |event, _, control_flow| {
            *control_flow = ControlFlow::Poll;
            match event {
                Event::LoopDestroyed => {
                    return;
                }
                Event::MainEventsCleared => {
                    window.window().request_redraw();
                }
                Event::RedrawRequested(_) => {
                    let start_time = Instant::now();
                    // gl.begin_query(glow::TIMESTAMP, start_query);
                    gl.clear(glow::COLOR_BUFFER_BIT | glow::DEPTH_BUFFER_BIT);
                    // gl::DrawElements(gl::TRIANGLES, 6, gl::UNSIGNED_INT, ptr::null());
                    program.Use(&gl);
                    vao.bind(&gl);

                    gl.draw_elements_instanced_base_vertex_base_instance(
                        glow::TRIANGLES,
                        INDICES.len() as i32,
                        glow::UNSIGNED_INT,
                        0,
                        msdf_data.gpu_paths.len() as i32,
                        0,
                        0,
                    );
                    // }

                    {
                        // gl.end_query(glow::TIMESTAMP);

                        // let mut done =
                        //     gl.get_query_parameter_u32(start_query, glow::QUERY_RESULT_AVAILABLE);
                        // while done == 0 {
                        //     done = gl
                        //         .get_query_parameter_u32(start_query, glow::QUERY_RESULT_AVAILABLE)
                        // }
                        // let gpu_time = gl.get_query_parameter_u32(start_query, glow::QUERY_RESULT);
                        // println!("gputime :{:?} ms ", gpu_time);
                    }

                    window.swap_buffers().unwrap();

                    {
                        let end_time = start_time.elapsed();
                        println!("cpu time: {:? }", end_time);
                        println!("instance count : {:? }", msdf_data.gpu_paths.len() as i32);
                    }
                }
                Event::WindowEvent { ref event, .. } => match event {
                    WindowEvent::Resized(physical_size) => {
                        window.resize(*physical_size);
                    }
                    WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                    WindowEvent::KeyboardInput { input, .. } => {
                        if input
                            .virtual_keycode
                            .eq(&Some(glutin::event::VirtualKeyCode::Escape))
                            && input.state == ElementState::Pressed
                        {
                            *control_flow = ControlFlow::Exit;
                            return;
                        }

                        let speed = 0.01;
                        let scale_speed = 0.1;
                        if input
                            .virtual_keycode
                            .eq(&Some(glutin::event::VirtualKeyCode::A))
                            && input.state == ElementState::Pressed
                        {
                            pos.x -= speed;
                            update_cam(&gl, &program, &pos, &scale);
                        }

                        if input
                            .virtual_keycode
                            .eq(&Some(glutin::event::VirtualKeyCode::D))
                            && input.state == ElementState::Pressed
                        {
                            pos.x += speed;
                            update_cam(&gl, &program, &pos, &scale);
                        }

                        if input
                            .virtual_keycode
                            .eq(&Some(glutin::event::VirtualKeyCode::W))
                            && input.state == ElementState::Pressed
                        {
                            pos.y -= speed;
                            update_cam(&gl, &program, &pos, &scale);
                        }

                        if input
                            .virtual_keycode
                            .eq(&Some(glutin::event::VirtualKeyCode::S))
                            && input.state == ElementState::Pressed
                        {
                            pos.y += speed;
                            update_cam(&gl, &program, &pos, &scale);
                        }

                        if input
                            .virtual_keycode
                            .eq(&Some(glutin::event::VirtualKeyCode::Q))
                            && input.state == ElementState::Pressed
                        {
                            scale.x += scale_speed;
                            scale.y += scale_speed;
                            update_cam(&gl, &program, &pos, &scale);
                        }

                        if input
                            .virtual_keycode
                            .eq(&Some(glutin::event::VirtualKeyCode::Z))
                            && input.state == ElementState::Pressed
                        {
                            scale.x -= scale_speed;
                            scale.y -= scale_speed;
                            update_cam(&gl, &program, &pos, &scale);
                        }
                    }
                    _ => (),
                },

                _ => (),
            }
        });
    }
}
