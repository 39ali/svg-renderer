
#include <iostream>

// TODO this should be defined by cmake but it's not :
#define MSDFGEN_PUBLIC
#include "SimpleBMP/simplebmp/simplebmp.h"
#include "msdfgen/core/arithmetics.hpp"
#include "msdfgen/msdfgen.h"


using namespace msdfgen;

int main() {
  Shape shape;

  Contour &contour = shape.addContour();
  Point2 p0(0.0, 0.0);
  Point2 p1(15.0, 0.0);
  Point2 p2(15.0, 15.0);
  Point2 p3(0.0, 15.0);
  contour.addEdge(new LinearSegment(p0, p1));
  contour.addEdge(new LinearSegment(p1, p2));
  contour.addEdge(new LinearSegment(p2, p3));
  contour.addEdge(new LinearSegment(p3, p0));

  shape.normalize();
  //                      max. angle
  edgeColoringSimple(shape, 3.0);
  //           image width, height

  if (1) {

    Bitmap<float, 3> msdf(1024, 1024);
    //                     range, scale, translation
    generateMSDF(msdf, shape, 4.0, 1.0, Vector2(4.0, 4.0));

    // save bmp
    const int w = msdf.width();
    const int h = msdf.height();
    SimpleBMP bmp(w, h);

    for (int i = 0; i < w; i++) {
      for (int j = 0; j < h; j++) {

        float *c = msdf(i, j);
        unsigned char r = clamp(256.f * (*(c++)), 255.f);
        unsigned char g = clamp(256.f * (*(c++)), 255.f);
        unsigned char b = clamp(256.f * (*(c++)), 255.f);
        bmp.setPixel(i, j, r, g, b);
      }
    }

    bmp.save("output.bmp");
  } else {

    Bitmap<float, 1> msdf(1024, 1024);
    //                     range, scale, translation
    generateSDF(msdf, shape, 4.0, 1.0, Vector2(4.0, 4.0));

    // save bmp
    const int w = msdf.width();
    const int h = msdf.height();
    SimpleBMP bmp(w, h);

    for (int i = 0; i < w; i++) {
      for (int j = 0; j < h; j++) {

        float *c = msdf(i, j);
        unsigned char r = clamp(256.f * (*(c++)), 255.f);
        // unsigned char g = *(c++) * 255.0;
        // unsigned char b = *(c++) * 255.0;
        bmp.setPixel(i, j, r, r, r);
      }
    }

    bmp.save("output.bmp");
  }

  return 0;
}