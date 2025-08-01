// Copyright 2019 Google LLC.
// Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
#include "tools/fiddle/examples.h"
REG_FIDDLE(Image_encodeToData_2, 256, 256, false, 3) {
void draw(SkCanvas* canvas) {
    canvas->scale(4, 4);
    SkIRect subset = {136, 32, 200, 96};
    // This prevents re-encoding the image's pixels if the image itself was created from
    // something like an encoded PNG.
    sk_sp<SkData> data = image->refEncodedData();
    if (!data) {
        data = SkPngEncoder::Encode(nullptr, image.get(), {});
    }
    sk_sp<SkImage> eye = SkImages::DeferredFromEncodedData(data)->makeSubset(nullptr, subset, {});
    canvas->drawImage(eye, 0, 0);
}
}  // END FIDDLE
