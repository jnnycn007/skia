// Copyright 2019 Google LLC.
// Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
#include "tools/fiddle/examples.h"
REG_FIDDLE(Paint_move_operator, 256, 256, true, 0) {
void draw(SkCanvas* canvas) {
    SkPaint paint1, paint2;
    paint1.setColor(SK_ColorRED);
    paint2 = std::move(paint1);
    SkDebugf("SK_ColorRED %c= paint2.getColor()\n", SK_ColorRED == paint2.getColor() ? '=' : '!');
}
}  // END FIDDLE
