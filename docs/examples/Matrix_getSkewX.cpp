// Copyright 2019 Google LLC.
// Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
#include "tools/fiddle/examples.h"
REG_FIDDLE(Matrix_getSkewX, 256, 256, true, 0) {
void draw(SkCanvas* canvas) {
    SkMatrix matrix;
    matrix.setSkew(42, 24);
    SkDebugf("matrix.getSkewX() %c= 42\n", matrix.getSkewX() == 42 ? '=' : '!');
}
}  // END FIDDLE
