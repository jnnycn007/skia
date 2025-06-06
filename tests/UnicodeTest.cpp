/*
 * Copyright 2011 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include "include/core/SkFont.h"
#include "include/core/SkFontTypes.h"
#include "src/base/SkUTF.h"
#include "src/core/SkFontPriv.h"
#include "tests/Test.h"
#include "tools/fonts/FontToolUtils.h"

#include <cstdint>
#include <cstring>
#include <string>

// Simple test to ensure that when we call textToGlyphs, we get the same
// result (for the same text) when using UTF8, UTF16, UTF32.
// TODO: make the text more complex (i.e. incorporate chars>7bits)
DEF_TEST(Unicode_textencodings, reporter) {
    const char text8[] = "ABCDEFGabcdefg0123456789";
    uint16_t text16[sizeof(text8)];
    int32_t  text32[sizeof(text8)];
    size_t len8 = strlen(text8);
    size_t len16 = len8 * 2;
    size_t len32 = len8 * 4;

    // expand our 8bit chars to 16 and 32
    for (size_t i = 0; i < len8; ++i) {
        text32[i] = text16[i] = text8[i];
    }

    SkGlyphID glyphs8[sizeof(text8)];
    SkGlyphID glyphs16[sizeof(text8)];
    SkGlyphID glyphs32[sizeof(text8)];

    SkFont font = ToolUtils::DefaultFont();

    int count8  = font.textToGlyphs(text8,  len8,  SkTextEncoding::kUTF8,  glyphs8);
    int count16 = font.textToGlyphs(text16, len16, SkTextEncoding::kUTF16, glyphs16);
    int count32 = font.textToGlyphs(text32, len32, SkTextEncoding::kUTF32, glyphs32);

    REPORTER_ASSERT(reporter, (int)len8 == count8);
    REPORTER_ASSERT(reporter, (int)len8 == count16);
    REPORTER_ASSERT(reporter, (int)len8 == count32);

    REPORTER_ASSERT(reporter, !memcmp(glyphs8, glyphs16, count8 * sizeof(SkGlyphID)));
    REPORTER_ASSERT(reporter, !memcmp(glyphs8, glyphs32, count8 * sizeof(SkGlyphID)));
}

DEF_TEST(glyphs_to_unichars, reporter) {
    SkFont font = ToolUtils::DefaultFont();

    const int N = 52;
    SkUnichar uni[N];
    for (int i = 0; i < 26; ++i) {
        uni[i +  0] = i + 'A';
        uni[i + 26] = i + 'a';
    }
    SkGlyphID glyphs[N];
    font.textToGlyphs(uni, sizeof(uni), SkTextEncoding::kUTF32, glyphs);

    SkUnichar uni2[N];
    SkFontPriv::GlyphsToUnichars(font, glyphs, N, uni2);
    REPORTER_ASSERT(reporter, memcmp(uni, uni2, sizeof(uni)) == 0);
}

