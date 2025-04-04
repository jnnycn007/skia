/*
 * Copyright 2016 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include "bench/Benchmark.h"
#include "include/core/SkBitmap.h"
#include "include/core/SkColorType.h"
#include "include/core/SkStream.h"
#include "include/encode/SkJpegEncoder.h"
#include "include/encode/SkPngEncoder.h"
#include "include/encode/SkWebpEncoder.h"
#include "tools/DecodeUtils.h"

// Like other Benchmark subclasses, Encoder benchmarks are run by:
// nanobench --match ^Encode_
//
// There is no corresponding DecodeBench class. Decoder benchmarks are run by:
// nanobench --benchType skcodec --images your_images_directory

class EncodeBench : public Benchmark {
public:
    using Encoder = bool (*)(SkWStream*, const SkPixmap&);
    EncodeBench(const char* filename, Encoder encoder, const char* encoderName, SkColorType colorType)
        : fSourceFilename(filename)
        , fEncoder(encoder)
        , fName(SkStringPrintf("Encode_%s_%s_%d", filename, encoderName, static_cast<int>(colorType)))
        , fColorType(colorType) {}

    bool isSuitableFor(Backend backend) override { return backend == Backend::kNonRendering; }

    const char* onGetName() override { return fName.c_str(); }

    void onDelayedSetup() override {
        SkAssertResult(
          ToolUtils::GetResourceAsBitmapWithColortype(fSourceFilename, &fBitmap, fColorType)
        );
    }

    void onDraw(int loops, SkCanvas*) override {
        while (loops-- > 0) {
            SkPixmap pixmap;
            SkAssertResult(fBitmap.peekPixels(&pixmap));
            SkNullWStream dst;
            SkAssertResult(fEncoder(&dst, pixmap));
            SkASSERT(dst.bytesWritten() > 0);
        }
    }

private:
    const char* fSourceFilename;
    Encoder     fEncoder;
    SkString    fName;
    SkBitmap    fBitmap;
    SkColorType fColorType;
};

static bool encode_jpeg(SkWStream* dst, const SkPixmap& src) {
    SkJpegEncoder::Options opts;
    opts.fQuality = 90;
    return SkJpegEncoder::Encode(dst, src, opts);
}

static bool encode_webp_lossy(SkWStream* dst, const SkPixmap& src) {
    SkWebpEncoder::Options opts;
    opts.fCompression = SkWebpEncoder::Compression::kLossy;
    opts.fQuality = 90;
    return SkWebpEncoder::Encode(dst, src, opts);
}

static bool encode_webp_lossless(SkWStream* dst, const SkPixmap& src) {
    SkWebpEncoder::Options opts;
    opts.fCompression = SkWebpEncoder::Compression::kLossless;
    opts.fQuality = 90;
    return SkWebpEncoder::Encode(dst, src, opts);
}

static bool encode_png(SkWStream* dst,
                       const SkPixmap& src,
                       SkPngEncoder::FilterFlag filters,
                       int zlibLevel) {
    SkPngEncoder::Options opts;
    opts.fFilterFlags = filters;
    opts.fZLibLevel = zlibLevel;
    return SkPngEncoder::Encode(dst, src, opts);
}

#define PNG(FLAG, ZLIBLEVEL) [](SkWStream* d, const SkPixmap& s) { \
           return encode_png(d, s, SkPngEncoder::FilterFlag::FLAG, ZLIBLEVEL); }

static const char* srcs[2] = {"images/mandrill_512.png", "images/color_wheel.jpg"};

// The Android Photos app uses a quality of 90 on JPEG encodes
DEF_BENCH(return new EncodeBench(srcs[0], &encode_jpeg, "JPEG", kRGBA_8888_SkColorType));
DEF_BENCH(return new EncodeBench(srcs[1], &encode_jpeg, "JPEG", kRGBA_8888_SkColorType));

// TODO: What is the appropriate quality to use to benchmark WEBP encodes?
DEF_BENCH(return new EncodeBench(srcs[0], encode_webp_lossy, "WEBP", kRGBA_8888_SkColorType));
DEF_BENCH(return new EncodeBench(srcs[1], encode_webp_lossy, "WEBP", kRGBA_8888_SkColorType));

DEF_BENCH(return new EncodeBench(srcs[0], encode_webp_lossless, "WEBP_LL", kRGBA_8888_SkColorType));
DEF_BENCH(return new EncodeBench(srcs[1], encode_webp_lossless, "WEBP_LL", kRGBA_8888_SkColorType));

DEF_BENCH(return new EncodeBench(srcs[0], PNG(kAll, 6), "PNG", kRGBA_8888_SkColorType));
DEF_BENCH(return new EncodeBench(srcs[0], PNG(kAll, 3), "PNG_3", kRGBA_8888_SkColorType));
DEF_BENCH(return new EncodeBench(srcs[0], PNG(kAll, 1), "PNG_1", kRGBA_8888_SkColorType));

DEF_BENCH(return new EncodeBench(srcs[0], PNG(kSub, 6), "PNG_6s", kRGBA_8888_SkColorType));
DEF_BENCH(return new EncodeBench(srcs[0], PNG(kSub, 3), "PNG_3s", kRGBA_8888_SkColorType));
DEF_BENCH(return new EncodeBench(srcs[0], PNG(kSub, 1), "PNG_1s", kRGBA_8888_SkColorType));

DEF_BENCH(return new EncodeBench(srcs[0], PNG(kNone, 6), "PNG_6n", kRGBA_8888_SkColorType));
DEF_BENCH(return new EncodeBench(srcs[0], PNG(kNone, 3), "PNG_3n", kRGBA_8888_SkColorType));
DEF_BENCH(return new EncodeBench(srcs[0], PNG(kNone, 1), "PNG_1n", kRGBA_8888_SkColorType));

DEF_BENCH(return new EncodeBench(srcs[1], PNG(kAll, 6), "PNG", kRGBA_8888_SkColorType));
DEF_BENCH(return new EncodeBench(srcs[1], PNG(kAll, 3), "PNG_3", kRGBA_8888_SkColorType));
DEF_BENCH(return new EncodeBench(srcs[1], PNG(kAll, 1), "PNG_1", kRGBA_8888_SkColorType));

DEF_BENCH(return new EncodeBench(srcs[1], PNG(kSub, 6), "PNG_6s", kRGBA_8888_SkColorType));
DEF_BENCH(return new EncodeBench(srcs[1], PNG(kSub, 3), "PNG_3s", kRGBA_8888_SkColorType));
DEF_BENCH(return new EncodeBench(srcs[1], PNG(kSub, 1), "PNG_1s", kRGBA_8888_SkColorType));

DEF_BENCH(return new EncodeBench(srcs[1], PNG(kNone, 6), "PNG_6n", kRGBA_8888_SkColorType));
DEF_BENCH(return new EncodeBench(srcs[1], PNG(kNone, 3), "PNG_3n", kRGBA_8888_SkColorType));
DEF_BENCH(return new EncodeBench(srcs[1], PNG(kNone, 1), "PNG_1n", kRGBA_8888_SkColorType));

DEF_BENCH(return new EncodeBench(srcs[0], PNG(kAll, 6), "PNG", kRGBA_F16_SkColorType));
DEF_BENCH(return new EncodeBench(srcs[1], PNG(kAll, 6), "PNG", kRGBA_F16_SkColorType));

DEF_BENCH(return new EncodeBench(srcs[0], PNG(kAll, 6), "PNG", kRGB_565_SkColorType));
DEF_BENCH(return new EncodeBench(srcs[1], PNG(kAll, 6), "PNG", kRGB_565_SkColorType));

#undef PNG
