
#include "../msdfgen/core/ShapeDistanceFinder.h"
#include "../msdfgen/msdfgen.h"
#include <memory>
#include <rust/cxx.h>

using namespace msdfgen;
using Bitmapf3 = Bitmap<float, 3>;

template <int N> static void invertColor(const BitmapRef<float, N> &bitmap) {
  const float *end = bitmap.pixels + N * bitmap.width * bitmap.height;
  for (float *p = bitmap.pixels; p < end; ++p)
    *p = 1.f - *p;
}

std::unique_ptr<Shape> new_shape() { return std::make_unique<Shape>(); };

void shape_guess_orientation(const Shape &shape, Bitmapf3 &msdf) {
  Shape::Bounds bounds = {};
  bounds = shape.getBounds();

  Point2 p(bounds.l - (bounds.r - bounds.l) - 1,
           bounds.b - (bounds.t - bounds.b) - 1);
  double distance = SimpleTrueShapeDistanceFinder::oneShotDistance(shape, p);
  if (distance > 0.0f) {
    invertColor<3>((BitmapRef<float, 3>)msdf);
  }
}

Contour &new_contour(Shape &shape) {
  Contour &contour = shape.addContour();
  return contour;
};

std::unique_ptr<Point2> new_point2(double x, double y) {
  return std::make_unique<Point2>(Point2{x, y});
};

void add_linear_segment(Contour &contour, const Point2 &p0, const Point2 &p1) {
  contour.addEdge(new LinearSegment(p0, p1));
};

void add_quadratic_segment(Contour &contour, const Point2 &p0, const Point2 &p1,
                           const Point2 &c0) {
  contour.addEdge(new QuadraticSegment(p0, p1, c0));
};

void add_cubic_segment(Contour &contour, const Point2 &p0, const Point2 &p1,
                       const Point2 &c0, const Point2 &c1) {
  contour.addEdge(new CubicSegment(p0, p1, c0, c1));
};

void edge_coloring_simple(Shape &shape, double angleThreshold) {
  edgeColoringSimple(shape, angleThreshold);
}

// bitmap functions

std::unique_ptr<Bitmapf3> new_bitmapf3(int w, int h) {
  return std::make_unique<Bitmapf3>(Bitmapf3{w, h});
};

void generate_msdf(Bitmapf3 &output, const Shape &shape, double range,
                   double scalex, double scaley, double translatex,
                   double translatey) {

  generateMSDF((BitmapRef<float, 3>)output, shape, range,
               Vector2(scalex, scaley), Vector2(translatex, translatey));
};

rust::Slice<const float> bitmapf3_pixels(const Bitmapf3 &output) {

  size_t size = output.width() * output.height() * 3;
  rust::Slice<const float> slice{(const float *)output, size};

  return slice;
};
