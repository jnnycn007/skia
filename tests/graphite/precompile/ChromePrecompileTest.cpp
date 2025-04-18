/*
 * Copyright 2024 Google LLC
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include "tests/Test.h"

#if defined(SK_GRAPHITE)

#include "include/gpu/graphite/Context.h"
#include "src/gpu/graphite/ContextPriv.h"
#include "src/gpu/graphite/ContextUtils.h"
#include "src/gpu/graphite/GraphicsPipelineDesc.h"
#include "src/gpu/graphite/PrecompileContextPriv.h"
#include "src/gpu/graphite/RenderPassDesc.h"
#include "src/gpu/graphite/RendererProvider.h"
#include "tools/graphite/UniqueKeyUtils.h"

#include <cstring>
#include <set>

// Print out a final report that includes missed cases in 'kCases'
//#define FINAL_REPORT

// Print out the cases (in 'kCases') that are covered by each 'kPrecompileCases' case
// Also lists the utilization of each 'kPrecompileCases' case
//#define PRINT_COVERAGE

// Print out all the generated labels and whether they were found in 'kCases'.
// This is usually used along with the 'kChosenCase' variable.
//#define PRINT_GENERATED_LABELS

/*** From here to the matching banner can be cut and pasted into Chrome's graphite_precompile.cc **/
#include "include/gpu/graphite/PrecompileContext.h"
#include "include/gpu/graphite/precompile/PaintOptions.h"
#include "include/gpu/graphite/precompile/Precompile.h"
#include "include/gpu/graphite/precompile/PrecompileColorFilter.h"
#include "include/gpu/graphite/precompile/PrecompileShader.h"

namespace {

using ::skgpu::graphite::DepthStencilFlags;
using ::skgpu::graphite::DrawTypeFlags;
using ::skgpu::graphite::PaintOptions;
using ::skgpu::graphite::RenderPassProperties;

// "SolidColor SrcOver"
PaintOptions solid_srcover() {
    PaintOptions paintOptions;
    paintOptions.setBlendModes({ SkBlendMode::kSrcOver });
    return paintOptions;
}

// "SolidColor SrcOver"
// "SolidColor Src"
// "SolidColor Clear"
PaintOptions solid_clear_src_srcover() {
    PaintOptions paintOptions;
    paintOptions.setBlendModes({ SkBlendMode::kClear,
                                 SkBlendMode::kSrc,
                                 SkBlendMode::kSrcOver });
    return paintOptions;
}

// "LocalMatrix [ Compose [ HardwareImage(0) ColorSpaceTransformPremul ] ] SrcOver"
// "LocalMatrix [ Compose [ Image(0) ColorSpaceTransformPremul ] ] SrcOver"
PaintOptions image_premul_srcover() {
    SkColorInfo ci { kRGBA_8888_SkColorType, kPremul_SkAlphaType, nullptr };
    PaintOptions paintOptions;
    paintOptions.setShaders({ skgpu::graphite::PrecompileShaders::Image({ &ci, 1 }) });
    paintOptions.setBlendModes({ SkBlendMode::kSrcOver });
    return paintOptions;
}

// LocalMatrix [ Compose [ HWYUVImage ColorSpaceTransformSRGB ] ] SrcOver" },
// LocalMatrix [ Compose [ YUVImage ColorSpaceTransformSRGB ] ] SrcOver" },
PaintOptions yuv_image_srgb_srcover() {
    SkColorInfo ci { kRGBA_8888_SkColorType,
                     kPremul_SkAlphaType,
                     SkColorSpace::MakeRGB(SkNamedTransferFn::kSRGB, SkNamedGamut::kAdobeRGB) };

    PaintOptions paintOptions;
    paintOptions.setShaders({ skgpu::graphite::PrecompileShaders::YUVImage(
            { &ci, 1 },
            /* includeCubic= */ false) });   // using cubic sampling w/ YUV images is rare
    paintOptions.setBlendModes({ SkBlendMode::kSrcOver });
    return paintOptions;
}

// "LocalMatrix [ Compose [ HardwareImage(0) ColorSpaceTransformPremul ] ] Src"
// "LocalMatrix [ Compose [ HardwareImage(0) ColorSpaceTransformPremul ] ] SrcOver"
// "LocalMatrix [ Compose [ Image(0) ColorSpaceTransformPremul ] ] SrcOver"
PaintOptions image_premul_src_srcover() {
    SkColorInfo ci { kRGBA_8888_SkColorType, kPremul_SkAlphaType, nullptr };
    PaintOptions paintOptions;
    paintOptions.setShaders({ skgpu::graphite::PrecompileShaders::Image({ &ci, 1 }) });
    paintOptions.setBlendModes({ SkBlendMode::kSrc,
                                 SkBlendMode::kSrcOver });
    return paintOptions;
}

// "LocalMatrix [ Compose [ HardwareImage(0) ColorSpaceTransformSRGB ] ] Src"
PaintOptions image_srgb_src() {
    SkColorInfo ci { kRGBA_8888_SkColorType,
                     kPremul_SkAlphaType,
                     SkColorSpace::MakeRGB(SkNamedTransferFn::kSRGB,
                                           SkNamedGamut::kAdobeRGB) };
    PaintOptions paintOptions;
    paintOptions.setShaders({ skgpu::graphite::PrecompileShaders::Image({ &ci, 1 }) });
    paintOptions.setBlendModes({ SkBlendMode::kSrc });
    return paintOptions;
}

// "Compose [ SolidColor BlendCompose [ SolidColor Passthrough PorterDuffBlender ] ] SrcOver"
PaintOptions blend_porter_duff_cf_srcover() {
    PaintOptions paintOptions;
    // kSrcOver will trigger the PorterDuffBlender
    paintOptions.setColorFilters(
            { skgpu::graphite::PrecompileColorFilters::Blend({ SkBlendMode::kSrcOver }) });
    paintOptions.setBlendModes({ SkBlendMode::kSrcOver });

    return paintOptions;
}

// "RP(color: Dawn(f=R8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: a000)",
// Single sampled R w/ just depth
const RenderPassProperties kR_1_D { DepthStencilFlags::kDepth,
                                    kAlpha_8_SkColorType,
                                    /* fDstCS= */ nullptr,
                                    /* fRequiresMSAA= */ false };

// "RP(color: Dawn(f=R8,s=4), resolve: Dawn(f=R8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: a000)",
// "RP(color: Dawn(f=R8,s=4) w/ msaa load, resolve: Dawn(f=R8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: a000)",
// MSAA R w/ depth and stencil
const RenderPassProperties kR_4_DS { DepthStencilFlags::kDepthStencil,
                                     kAlpha_8_SkColorType,
                                     /* fDstCS= */ nullptr,
                                     /* fRequiresMSAA= */ true };

// "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba)"
// Single sampled BGRA w/ just depth
const RenderPassProperties kBGRA_1_D { DepthStencilFlags::kDepth,
                                       kBGRA_8888_SkColorType,
                                       /* fDstCS= */ nullptr,
                                       /* fRequiresMSAA= */ false };

// "RP(color: Dawn(f=BGRA8,s=4), resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D16,s=4), samples: 4, swizzle: rgba)"
// "RP(color: Dawn(f=BGRA8,s=4) w/ msaa load, resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D16,s=4), samples: 4, swizzle: rgba)"
// MSAA BGRA w/ just depth
const RenderPassProperties kBGRA_4_D { DepthStencilFlags::kDepth,
                                       kBGRA_8888_SkColorType,
                                       /* fDstCS= */ nullptr,
                                       /* fRequiresMSAA= */ true };

// "RP(color: Dawn(f=BGRA8,s=4), resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba)"
// "RP(color: Dawn(f=BGRA8,s=4) w/ msaa load, resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba)",
// MSAA BGRA w/ depth and stencil
const RenderPassProperties kBGRA_4_DS { DepthStencilFlags::kDepthStencil,
                                        kBGRA_8888_SkColorType,
                                        /* fDstCS= */ nullptr,
                                        /* fRequiresMSAA= */ true };

// The same as kBGRA_1_D but w/ an SRGB colorSpace
const RenderPassProperties kBGRA_1_D_SRGB { DepthStencilFlags::kDepth,
                                            kBGRA_8888_SkColorType,
                                            SkColorSpace::MakeSRGB(),
                                            /* fRequiresMSAA= */ false };

// The same as kBGRA_4_DS but w/ an SRGB colorSpace
const RenderPassProperties kBGRA_4_DS_SRGB { DepthStencilFlags::kDepthStencil,
                                             kBGRA_8888_SkColorType,
                                             SkColorSpace::MakeSRGB(),
                                             /* fRequiresMSAA= */ true };

// These settings cover 176 of the 202 cases in 'kCases'.
const struct PrecompileSettings {
    PaintOptions fPaintOptions;
    DrawTypeFlags fDrawTypeFlags = DrawTypeFlags::kNone;
    RenderPassProperties fRenderPassProps;
} kPrecompileCases[] = {
    { solid_srcover(),                DrawTypeFlags::kSimpleShape,     kR_1_D },

    { solid_srcover(),                DrawTypeFlags::kNonSimpleShape,  kR_4_DS },

    { solid_srcover(),                DrawTypeFlags::kBitmapText_Mask, kBGRA_1_D },
    { blend_porter_duff_cf_srcover(), DrawTypeFlags::kNonSimpleShape,  kBGRA_1_D },
    { image_premul_src_srcover(),     DrawTypeFlags::kSimpleShape,     kBGRA_1_D },
    { solid_clear_src_srcover(),      DrawTypeFlags::kSimpleShape,     kBGRA_1_D },

    { solid_srcover(),                DrawTypeFlags::kBitmapText_Mask, kBGRA_4_D },
    { solid_srcover(),                DrawTypeFlags::kNonSimpleShape,  kBGRA_4_D },

    { solid_srcover(),                DrawTypeFlags::kBitmapText_Mask, kBGRA_4_DS },
    { solid_srcover(),                DrawTypeFlags::kCircularArc,     kBGRA_4_DS },
    { solid_srcover(),                DrawTypeFlags::kNonSimpleShape,  kBGRA_4_DS },
    { image_premul_srcover(),         DrawTypeFlags::kSimpleShape,     kBGRA_4_DS },
    { solid_clear_src_srcover(),      DrawTypeFlags::kSimpleShape,     kBGRA_4_DS },

    { image_srgb_src(),               DrawTypeFlags::kSimpleShape,     kBGRA_1_D_SRGB },
    { yuv_image_srgb_srcover(),       DrawTypeFlags::kSimpleShape,     kBGRA_1_D_SRGB },

    // These two are interesting but have < 40% utility
    // { yuv_image_srgb_srcover(),      DrawTypeFlags::kSimpleShape,     kBGRA_4_DS_SRGB },
    // { solid_srcover(),               DrawTypeFlags::kSimpleShape,     kBGRA_4_D },
};

/*********** Here ends the part that can be pasted into Chrome's graphite_precompile.cc ***********/

// This helper maps from the RenderPass string in the Pipeline label to the
// RenderPassProperties needed by the Precompile system
// TODO(robertphillips): converting this to a more piecemeal approach might better illuminate
// the mapping between the string and the RenderPassProperties
[[maybe_unused]] RenderPassProperties get_render_pass_properties(const char* str) {
    static const struct {
        const char* fStr;
        RenderPassProperties fRenderPassProperties;
    } kRenderPassPropertiesMapping[] = {
        { "RP(color: Dawn(f=R8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: a000)",
          kR_1_D },

        // This RPP (kR_4_DS) can generate two strings when Caps::loadOpAffectsMSAAPipelines.
        { "RP(color: Dawn(f=R8,s=4), resolve: Dawn(f=R8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: a000)",
          kR_4_DS },
        { "RP(color: Dawn(f=R8,s=4) w/ msaa load, resolve: Dawn(f=R8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: a000)",
          kR_4_DS },

        { "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba)",
          kBGRA_1_D },

        // This RPP (kBGRA_4_D) can generate two strings when Caps::loadOpAffectsMSAAPipelines.
        { "RP(color: Dawn(f=BGRA8,s=4), resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D16,s=4), samples: 4, swizzle: rgba)",
          kBGRA_4_D },
        { "RP(color: Dawn(f=BGRA8,s=4) w/ msaa load, resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D16,s=4), samples: 4, swizzle: rgba)",
          kBGRA_4_D },

        // This RPP (kBGRA_4_DS) can generate two strings when Caps::loadOpAffectsMSAAPipelines.
        { "RP(color: Dawn(f=BGRA8,s=4), resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba)",
           kBGRA_4_DS },
        { "RP(color: Dawn(f=BGRA8,s=4) w/ msaa load, resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba)",
           kBGRA_4_DS },
    };

    for (const auto& rppm : kRenderPassPropertiesMapping) {
        if (strstr(str, rppm.fStr)) {
            return rppm.fRenderPassProperties;
        }
    }

    SkAssertResult(0);
    return {};
}

// This helper maps from the RenderStep's name in the Pipeline label to the DrawTypeFlag that
// resulted in its use.
[[maybe_unused]] DrawTypeFlags get_draw_type_flags(const char* str) {
    static const struct {
        const char* fStr;
        DrawTypeFlags fFlags;
    } kDrawTypeFlagsMapping[] = {
        { "BitmapTextRenderStep[Mask]",                  DrawTypeFlags::kBitmapText_Mask  },
        { "BitmapTextRenderStep[LCD]",                   DrawTypeFlags::kBitmapText_LCD   },
        { "BitmapTextRenderStep[Color]",                 DrawTypeFlags::kBitmapText_Color },

        { "SDFTextRenderStep",                           DrawTypeFlags::kSDFText      },
        { "SDFTextLCDRenderStep",                        DrawTypeFlags::kSDFText_LCD  },

        { "VerticesRenderStep[Tris]",                    DrawTypeFlags::kDrawVertices },
        { "VerticesRenderStep[TrisTexCoords]",           DrawTypeFlags::kDrawVertices },
        { "VerticesRenderStep[TrisColor]",               DrawTypeFlags::kDrawVertices },
        { "VerticesRenderStep[TrisColorTexCoords]",      DrawTypeFlags::kDrawVertices },
        { "VerticesRenderStep[Tristrips]",               DrawTypeFlags::kDrawVertices },
        { "VerticesRenderStep[TristripsTexCoords]",      DrawTypeFlags::kDrawVertices },
        { "VerticesRenderStep[TristripsColor]",          DrawTypeFlags::kDrawVertices },
        { "VerticesRenderStep[TristripsColorTexCoords]", DrawTypeFlags::kDrawVertices },

        { "CircularArcRenderStep",                       DrawTypeFlags::kCircularArc  },

        { "AnalyticRRectRenderStep",                     DrawTypeFlags::kSimpleShape  },
        { "CoverBoundsRenderStep[NonAAFill]",            DrawTypeFlags::kSimpleShape  },
        { "PerEdgeAAQuadRenderStep",                     DrawTypeFlags::kSimpleShape  },

        { "CoverageMaskRenderStep",                      DrawTypeFlags::kNonSimpleShape },
        { "CoverBoundsRenderStep[RegularCover]",         DrawTypeFlags::kNonSimpleShape },
        { "CoverBoundsRenderStep[InverseCover]",         DrawTypeFlags::kNonSimpleShape },
        { "MiddleOutFanRenderStep[EvenOdd]",             DrawTypeFlags::kNonSimpleShape },
        { "MiddleOutFanRenderStep[Winding]",             DrawTypeFlags::kNonSimpleShape },
        { "TessellateCurvesRenderStep[EvenOdd]",         DrawTypeFlags::kNonSimpleShape },
        { "TessellateCurvesRenderStep[Winding]",         DrawTypeFlags::kNonSimpleShape },
        { "TessellateStrokesRenderStep",                 DrawTypeFlags::kNonSimpleShape },
        { "TessellateWedgesRenderStep[Convex]",          DrawTypeFlags::kNonSimpleShape },
        { "TessellateWedgesRenderStep[EvenOdd]",         DrawTypeFlags::kNonSimpleShape },
        { "TessellateWedgesRenderStep[Winding]",         DrawTypeFlags::kNonSimpleShape },
    };

    for (const auto& dtfm : kDrawTypeFlagsMapping) {
        if (strstr(str, dtfm.fStr)) {
            SkAssertResult(dtfm.fFlags != DrawTypeFlags::kNone);
            return dtfm.fFlags;
        }
    }

    SkAssertResult(0);
    return DrawTypeFlags::kNone;
}


struct ChromePipeline {
    int fNumHits;         // the number of uses in the top 9 most visited web sites
    const char* fString;
};

//
// These Pipelines are candidates for inclusion in Chrome's precompile. They were generated
// by collecting all the Pipelines from 9 of the top 14 visited sites according to Wikipedia
//
static const ChromePipeline kCases[] = {
//--------
/*   0 */ { 9, "RP(color: Dawn(f=R8,s=4), resolve: Dawn(f=R8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: a000) + "
               "TessellateWedgesRenderStep[Winding] + (empty)" },
/*   1 */ { 9, "RP(color: Dawn(f=R8,s=4), resolve: Dawn(f=R8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: a000) + "
               "TessellateWedgesRenderStep[EvenOdd] + (empty)" },
/*   2 */ { 9, "RP(color: Dawn(f=R8,s=4), resolve: Dawn(f=R8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: a000) + "
               "TessellateWedgesRenderStep[Convex] + SolidColor SrcOver" },
/*   3 */ { 9, "RP(color: Dawn(f=R8,s=4), resolve: Dawn(f=R8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: a000) + "
               "TessellateStrokesRenderStep + SolidColor SrcOver" },
/*   4 */ { 9, "RP(color: Dawn(f=R8,s=4), resolve: Dawn(f=R8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: a000) + "
               "TessellateCurvesRenderStep[Winding] + (empty)" },
/*   5 */ { 9, "RP(color: Dawn(f=R8,s=4), resolve: Dawn(f=R8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: a000) + "
               "TessellateCurvesRenderStep[EvenOdd] + (empty)" },
/*   6 */ { 9, "RP(color: Dawn(f=R8,s=4), resolve: Dawn(f=R8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: a000) + "
               "MiddleOutFanRenderStep[Winding] + (empty)" },
/*   7 */ { 9, "RP(color: Dawn(f=R8,s=4), resolve: Dawn(f=R8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: a000) + "
               "MiddleOutFanRenderStep[EvenOdd] + (empty)" },
/*   8 */ { 9, "RP(color: Dawn(f=R8,s=4), resolve: Dawn(f=R8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: a000) + "
               "CoverBoundsRenderStep[RegularCover] + SolidColor SrcOver" },
/*   9 */ { 9, "RP(color: Dawn(f=R8,s=4), resolve: Dawn(f=R8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: a000) + "
               "CoverBoundsRenderStep[InverseCover] + SolidColor SrcOver" },
/*  10 */ { 9, "RP(color: Dawn(f=R8,s=4), resolve: Dawn(f=R8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: a000) + "
               "CoverageMaskRenderStep + SolidColor SrcOver" },
//--------
/*  11 */ { 9, "RP(color: Dawn(f=R8,s=4) w/ msaa load, resolve: Dawn(f=R8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: a000) + "
               "TessellateWedgesRenderStep[Winding] + (empty)" },
/*  12 */ { 9, "RP(color: Dawn(f=R8,s=4) w/ msaa load, resolve: Dawn(f=R8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: a000) + "
               "TessellateWedgesRenderStep[EvenOdd] + (empty)" },
/*  13 */ { 9, "RP(color: Dawn(f=R8,s=4) w/ msaa load, resolve: Dawn(f=R8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: a000) + "
               "TessellateWedgesRenderStep[Convex] + SolidColor SrcOver" },
/*  14 */ { 9, "RP(color: Dawn(f=R8,s=4) w/ msaa load, resolve: Dawn(f=R8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: a000) + "
               "TessellateStrokesRenderStep + SolidColor SrcOver" },
/*  15 */ { 9, "RP(color: Dawn(f=R8,s=4) w/ msaa load, resolve: Dawn(f=R8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: a000) + "
               "TessellateCurvesRenderStep[Winding] + (empty)" },
/*  16 */ { 9, "RP(color: Dawn(f=R8,s=4) w/ msaa load, resolve: Dawn(f=R8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: a000) + "
               "TessellateCurvesRenderStep[EvenOdd] + (empty)" },
/*  17 */ { 9, "RP(color: Dawn(f=R8,s=4) w/ msaa load, resolve: Dawn(f=R8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: a000) + "
               "MiddleOutFanRenderStep[Winding] + (empty)" },
/*  18 */ { 9, "RP(color: Dawn(f=R8,s=4) w/ msaa load, resolve: Dawn(f=R8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: a000) + "
               "MiddleOutFanRenderStep[EvenOdd] + (empty)" },
/*  19 */ { 9, "RP(color: Dawn(f=R8,s=4) w/ msaa load, resolve: Dawn(f=R8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: a000) + "
               "CoverBoundsRenderStep[RegularCover] + SolidColor SrcOver" },
/*  20 */ { 9, "RP(color: Dawn(f=R8,s=4) w/ msaa load, resolve: Dawn(f=R8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: a000) + "
               "CoverBoundsRenderStep[InverseCover] + SolidColor SrcOver" },
/*  21 */ { 9, "RP(color: Dawn(f=R8,s=4) w/ msaa load, resolve: Dawn(f=R8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: a000) + "
               "CoverageMaskRenderStep + SolidColor SrcOver" },
//--------
/*  22 */ { 9, "RP(color: Dawn(f=BGRA8,s=4), resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "TessellateWedgesRenderStep[Winding] + (empty)" },
/*  23 */ { 9, "RP(color: Dawn(f=BGRA8,s=4), resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "TessellateWedgesRenderStep[EvenOdd] + (empty)" },
/*  24 */ { 9, "RP(color: Dawn(f=BGRA8,s=4), resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "TessellateWedgesRenderStep[Convex] + SolidColor SrcOver" },
/*  25 */ { 9, "RP(color: Dawn(f=BGRA8,s=4), resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "TessellateStrokesRenderStep + SolidColor SrcOver" },
/*  26 */ { 9, "RP(color: Dawn(f=BGRA8,s=4), resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "TessellateCurvesRenderStep[Winding] + (empty)" },
/*  27 */ { 9, "RP(color: Dawn(f=BGRA8,s=4), resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "TessellateCurvesRenderStep[EvenOdd] + (empty)" },
/*  28 */ { 9, "RP(color: Dawn(f=BGRA8,s=4), resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "PerEdgeAAQuadRenderStep + SolidColor SrcOver" },
/*  29 */ { 9, "RP(color: Dawn(f=BGRA8,s=4), resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "PerEdgeAAQuadRenderStep + SolidColor Src" },
/*  30 */ { 9, "RP(color: Dawn(f=BGRA8,s=4), resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "PerEdgeAAQuadRenderStep + SolidColor Clear" },
/*  31 */ { 9, "RP(color: Dawn(f=BGRA8,s=4), resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "PerEdgeAAQuadRenderStep + LocalMatrix [ Compose [ ImageShaderClamp(0) ColorSpaceTransformPremul ] ] SrcOver" },
/*  32 */ { 9, "RP(color: Dawn(f=BGRA8,s=4), resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "PerEdgeAAQuadRenderStep + LocalMatrix [ Compose [ Image(0) ColorSpaceTransformPremul ] ] SrcOver" },
/*  33 */ { 9, "RP(color: Dawn(f=BGRA8,s=4), resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "PerEdgeAAQuadRenderStep + LocalMatrix [ Compose [ HardwareImage(0) ColorSpaceTransformPremul ] ] SrcOver" },
/*  34 */ { 9, "RP(color: Dawn(f=BGRA8,s=4), resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "PerEdgeAAQuadRenderStep + LocalMatrix [ Compose [ CubicImage(0) ColorSpaceTransformPremul ] ] SrcOver" },
/*  35 */ { 9, "RP(color: Dawn(f=BGRA8,s=4), resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "MiddleOutFanRenderStep[Winding] + (empty)" },
/*  36 */ { 9, "RP(color: Dawn(f=BGRA8,s=4), resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "MiddleOutFanRenderStep[EvenOdd] + (empty)" },
/*  37 */ { 9, "RP(color: Dawn(f=BGRA8,s=4), resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "CoverBoundsRenderStep[RegularCover] + SolidColor SrcOver" },
/*  38 */ { 9, "RP(color: Dawn(f=BGRA8,s=4), resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "CoverBoundsRenderStep[NonAAFill] + SolidColor SrcOver AnalyticClip" },
/*  39 */ { 9, "RP(color: Dawn(f=BGRA8,s=4), resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "CoverBoundsRenderStep[NonAAFill] + SolidColor SrcOver" },
/*  40 */ { 9, "RP(color: Dawn(f=BGRA8,s=4), resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "CoverBoundsRenderStep[NonAAFill] + SolidColor Src" },
/*  41 */ { 9, "RP(color: Dawn(f=BGRA8,s=4), resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "CoverBoundsRenderStep[NonAAFill] + SolidColor Clear" },
/*  42 */ { 9, "RP(color: Dawn(f=BGRA8,s=4), resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "CoverBoundsRenderStep[NonAAFill] + LocalMatrix [ Compose [ ImageShaderClamp(0) ColorSpaceTransformPremul ] ] SrcOver" },
/*  43 */ { 9, "RP(color: Dawn(f=BGRA8,s=4), resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "CoverBoundsRenderStep[NonAAFill] + LocalMatrix [ Compose [ Image(0) ColorSpaceTransformPremul ] ] SrcOver" },
/*  44 */ { 9, "RP(color: Dawn(f=BGRA8,s=4), resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "CoverBoundsRenderStep[NonAAFill] + LocalMatrix [ Compose [ HardwareImage(0) ColorSpaceTransformPremul ] ] SrcOver" },
/*  45 */ { 9, "RP(color: Dawn(f=BGRA8,s=4), resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "CoverBoundsRenderStep[NonAAFill] + LocalMatrix [ Compose [ CubicImage(0) ColorSpaceTransformPremul ] ] SrcOver" },
/*  46 */ { 9, "RP(color: Dawn(f=BGRA8,s=4), resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "CoverBoundsRenderStep[InverseCover] + SolidColor SrcOver" },
// Hmm - what even is this? A cover draw w/o a Shader?
/*  47 */ { 9, "RP(color: Dawn(f=BGRA8,s=4), resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "CoverBoundsRenderStep[InverseCover] + (empty)" },
/*  48 */ { 9, "RP(color: Dawn(f=BGRA8,s=4), resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "CoverageMaskRenderStep + SolidColor SrcOver" },
/*  49 */ { 9, "RP(color: Dawn(f=BGRA8,s=4), resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "CircularArcRenderStep + SolidColor SrcOver" },
/*  50 */ { 9, "RP(color: Dawn(f=BGRA8,s=4), resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "BitmapTextRenderStep[Mask] + SolidColor SrcOver" },
/*  51 */ { 9, "RP(color: Dawn(f=BGRA8,s=4), resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "AnalyticRRectRenderStep + SolidColor SrcOver" },
/*  52 */ { 9, "RP(color: Dawn(f=BGRA8,s=4), resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "AnalyticRRectRenderStep + SolidColor Src" },
/*  53 */ { 9, "RP(color: Dawn(f=BGRA8,s=4), resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "AnalyticRRectRenderStep + SolidColor Clear" },
/*  54 */ { 9, "RP(color: Dawn(f=BGRA8,s=4), resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "AnalyticRRectRenderStep + LocalMatrix [ Compose [ ImageShaderClamp(0) ColorSpaceTransformPremul ] ] SrcOver" },
/*  55 */ { 9, "RP(color: Dawn(f=BGRA8,s=4), resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "AnalyticRRectRenderStep + LocalMatrix [ Compose [ Image(0) ColorSpaceTransformPremul ] ] SrcOver" },
/*  56 */ { 9, "RP(color: Dawn(f=BGRA8,s=4), resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "AnalyticRRectRenderStep + LocalMatrix [ Compose [ HardwareImage(0) ColorSpaceTransformPremul ] ] SrcOver" },
/*  57 */ { 9, "RP(color: Dawn(f=BGRA8,s=4), resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "AnalyticRRectRenderStep + LocalMatrix [ Compose [ CubicImage(0) ColorSpaceTransformPremul ] ] SrcOver" },
/*  58 */ { 9, "RP(color: Dawn(f=BGRA8,s=4), resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D16,s=4), samples: 4, swizzle: rgba) + "
               "TessellateWedgesRenderStep[Winding] + (empty)" },
/*  59 */ { 9, "RP(color: Dawn(f=BGRA8,s=4), resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D16,s=4), samples: 4, swizzle: rgba) + "
               "TessellateWedgesRenderStep[EvenOdd] + (empty)" },
/*  60 */ { 9, "RP(color: Dawn(f=BGRA8,s=4), resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D16,s=4), samples: 4, swizzle: rgba) + "
               "TessellateWedgesRenderStep[Convex] + SolidColor SrcOver" },
/*  61 */ { 9, "RP(color: Dawn(f=BGRA8,s=4), resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D16,s=4), samples: 4, swizzle: rgba) + "
               "TessellateStrokesRenderStep + SolidColor SrcOver" },
/*  62 */ { 9, "RP(color: Dawn(f=BGRA8,s=4), resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D16,s=4), samples: 4, swizzle: rgba) + "
               "TessellateCurvesRenderStep[Winding] + (empty)" },
/*  63 */ { 9, "RP(color: Dawn(f=BGRA8,s=4), resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D16,s=4), samples: 4, swizzle: rgba) + "
               "TessellateCurvesRenderStep[EvenOdd] + (empty)" },
/*  64 */ { 9, "RP(color: Dawn(f=BGRA8,s=4), resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D16,s=4), samples: 4, swizzle: rgba) + "
               "MiddleOutFanRenderStep[Winding] + (empty)" },
/*  65 */ { 9, "RP(color: Dawn(f=BGRA8,s=4), resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D16,s=4), samples: 4, swizzle: rgba) + "
               "MiddleOutFanRenderStep[EvenOdd] + (empty)" },
/*  66 */ { 9, "RP(color: Dawn(f=BGRA8,s=4), resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D16,s=4), samples: 4, swizzle: rgba) + "
               "CoverBoundsRenderStep[RegularCover] + SolidColor SrcOver" },
/*  67 */ { 9, "RP(color: Dawn(f=BGRA8,s=4), resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D16,s=4), samples: 4, swizzle: rgba) + "
               "CoverBoundsRenderStep[NonAAFill] + LocalMatrix [ Compose [ HardwareImage(0) ColorSpaceTransformPremul ] ] SrcOver AnalyticClip" },
/*  68 */ { 9, "RP(color: Dawn(f=BGRA8,s=4), resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D16,s=4), samples: 4, swizzle: rgba) + "
               "CoverBoundsRenderStep[InverseCover] + SolidColor SrcOver" },
/*  69 */ { 9, "RP(color: Dawn(f=BGRA8,s=4), resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D16,s=4), samples: 4, swizzle: rgba) + "
               "CoverageMaskRenderStep + SolidColor SrcOver" },
/*  70 */ { 9, "RP(color: Dawn(f=BGRA8,s=4) w/ msaa load, resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "TessellateWedgesRenderStep[Winding] + (empty)" },
/*  71 */ { 9, "RP(color: Dawn(f=BGRA8,s=4) w/ msaa load, resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "TessellateWedgesRenderStep[EvenOdd] + (empty)" },
/*  72 */ { 9, "RP(color: Dawn(f=BGRA8,s=4) w/ msaa load, resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "TessellateWedgesRenderStep[Convex] + SolidColor SrcOver" },
/*  73 */ { 9, "RP(color: Dawn(f=BGRA8,s=4) w/ msaa load, resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "TessellateStrokesRenderStep + SolidColor SrcOver" },
/*  74 */ { 9, "RP(color: Dawn(f=BGRA8,s=4) w/ msaa load, resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "TessellateCurvesRenderStep[Winding] + (empty)" },
/*  75 */ { 9, "RP(color: Dawn(f=BGRA8,s=4) w/ msaa load, resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "TessellateCurvesRenderStep[EvenOdd] + (empty)" },
/*  76 */ { 9, "RP(color: Dawn(f=BGRA8,s=4) w/ msaa load, resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "PerEdgeAAQuadRenderStep + SolidColor SrcOver" },
/*  77 */ { 9, "RP(color: Dawn(f=BGRA8,s=4) w/ msaa load, resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "PerEdgeAAQuadRenderStep + SolidColor Src" },
/*  78 */ { 9, "RP(color: Dawn(f=BGRA8,s=4) w/ msaa load, resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "PerEdgeAAQuadRenderStep + SolidColor Clear" },
/*  79 */ { 9, "RP(color: Dawn(f=BGRA8,s=4) w/ msaa load, resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "PerEdgeAAQuadRenderStep + LocalMatrix [ Compose [ ImageShaderClamp(0) ColorSpaceTransformPremul ] ] SrcOver" },
/*  80 */ { 9, "RP(color: Dawn(f=BGRA8,s=4) w/ msaa load, resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "PerEdgeAAQuadRenderStep + LocalMatrix [ Compose [ Image(0) ColorSpaceTransformPremul ] ] SrcOver" },
/*  81 */ { 9, "RP(color: Dawn(f=BGRA8,s=4) w/ msaa load, resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "PerEdgeAAQuadRenderStep + LocalMatrix [ Compose [ HardwareImage(0) ColorSpaceTransformPremul ] ] SrcOver" },
/*  82 */ { 9, "RP(color: Dawn(f=BGRA8,s=4) w/ msaa load, resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "PerEdgeAAQuadRenderStep + LocalMatrix [ Compose [ CubicImage(0) ColorSpaceTransformPremul ] ] SrcOver" },
/*  83 */ { 9, "RP(color: Dawn(f=BGRA8,s=4) w/ msaa load, resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "MiddleOutFanRenderStep[Winding] + (empty)" },
/*  84 */ { 9, "RP(color: Dawn(f=BGRA8,s=4) w/ msaa load, resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "MiddleOutFanRenderStep[EvenOdd] + (empty)" },
/*  85 */ { 9, "RP(color: Dawn(f=BGRA8,s=4) w/ msaa load, resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "CoverBoundsRenderStep[RegularCover] + SolidColor SrcOver" },
/*  86 */ { 9, "RP(color: Dawn(f=BGRA8,s=4) w/ msaa load, resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "CoverBoundsRenderStep[NonAAFill] + SolidColor SrcOver AnalyticClip" },
/*  87 */ { 9, "RP(color: Dawn(f=BGRA8,s=4) w/ msaa load, resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "CoverBoundsRenderStep[NonAAFill] + SolidColor SrcOver" },
/*  88 */ { 9, "RP(color: Dawn(f=BGRA8,s=4) w/ msaa load, resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "CoverBoundsRenderStep[NonAAFill] + SolidColor Src" },
/*  89 */ { 9, "RP(color: Dawn(f=BGRA8,s=4) w/ msaa load, resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "CoverBoundsRenderStep[NonAAFill] + SolidColor Clear" },
/*  90 */ { 9, "RP(color: Dawn(f=BGRA8,s=4) w/ msaa load, resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "CoverBoundsRenderStep[NonAAFill] + LocalMatrix [ Compose [ ImageShaderClamp(0) ColorSpaceTransformPremul ] ] SrcOver" },
/*  91 */ { 9, "RP(color: Dawn(f=BGRA8,s=4) w/ msaa load, resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "CoverBoundsRenderStep[NonAAFill] + LocalMatrix [ Compose [ Image(0) ColorSpaceTransformPremul ] ] SrcOver" },
/*  92 */ { 9, "RP(color: Dawn(f=BGRA8,s=4) w/ msaa load, resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "CoverBoundsRenderStep[NonAAFill] + LocalMatrix [ Compose [ HardwareImage(0) ColorSpaceTransformPremul ] ] SrcOver" },
/*  93 */ { 9, "RP(color: Dawn(f=BGRA8,s=4) w/ msaa load, resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "CoverBoundsRenderStep[NonAAFill] + LocalMatrix [ Compose [ CubicImage(0) ColorSpaceTransformPremul ] ] SrcOver" },
/*  94 */ { 9, "RP(color: Dawn(f=BGRA8,s=4) w/ msaa load, resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "CoverBoundsRenderStep[InverseCover] + SolidColor SrcOver" },
// What even is this? A cover draw w/o a shader?
/*  95 */ { 9, "RP(color: Dawn(f=BGRA8,s=4) w/ msaa load, resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "CoverBoundsRenderStep[InverseCover] + (empty)" },
/*  96 */ { 9, "RP(color: Dawn(f=BGRA8,s=4) w/ msaa load, resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "CoverageMaskRenderStep + SolidColor SrcOver" },
/*  97 */ { 9, "RP(color: Dawn(f=BGRA8,s=4) w/ msaa load, resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "CircularArcRenderStep + SolidColor SrcOver" },
/*  98 */ { 9, "RP(color: Dawn(f=BGRA8,s=4) w/ msaa load, resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "BitmapTextRenderStep[Mask] + SolidColor SrcOver" },
/*  99 */ { 9, "RP(color: Dawn(f=BGRA8,s=4) w/ msaa load, resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "AnalyticRRectRenderStep + SolidColor SrcOver" },
/* 100 */ { 9, "RP(color: Dawn(f=BGRA8,s=4) w/ msaa load, resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "AnalyticRRectRenderStep + SolidColor Src" },
/* 101 */ { 9, "RP(color: Dawn(f=BGRA8,s=4) w/ msaa load, resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "AnalyticRRectRenderStep + SolidColor Clear" },
/* 102 */ { 9, "RP(color: Dawn(f=BGRA8,s=4) w/ msaa load, resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "AnalyticRRectRenderStep + LocalMatrix [ Compose [ ImageShaderClamp(0) ColorSpaceTransformPremul ] ] SrcOver" },
/* 103 */ { 9, "RP(color: Dawn(f=BGRA8,s=4) w/ msaa load, resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "AnalyticRRectRenderStep + LocalMatrix [ Compose [ Image(0) ColorSpaceTransformPremul ] ] SrcOver" },
/* 104 */ { 9, "RP(color: Dawn(f=BGRA8,s=4) w/ msaa load, resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "AnalyticRRectRenderStep + LocalMatrix [ Compose [ HardwareImage(0) ColorSpaceTransformPremul ] ] SrcOver" },
/* 105 */ { 9, "RP(color: Dawn(f=BGRA8,s=4) w/ msaa load, resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "AnalyticRRectRenderStep + LocalMatrix [ Compose [ CubicImage(0) ColorSpaceTransformPremul ] ] SrcOver" },
/* 106 */ { 9, "RP(color: Dawn(f=BGRA8,s=4) w/ msaa load, resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D16,s=4), samples: 4, swizzle: rgba) + "
               "TessellateWedgesRenderStep[Winding] + (empty)" },
/* 107 */ { 9, "RP(color: Dawn(f=BGRA8,s=4) w/ msaa load, resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D16,s=4), samples: 4, swizzle: rgba) + "
               "TessellateWedgesRenderStep[EvenOdd] + (empty)" },
/* 108 */ { 9, "RP(color: Dawn(f=BGRA8,s=4) w/ msaa load, resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D16,s=4), samples: 4, swizzle: rgba) + "
               "TessellateWedgesRenderStep[Convex] + SolidColor SrcOver" },
/* 109 */ { 9, "RP(color: Dawn(f=BGRA8,s=4) w/ msaa load, resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D16,s=4), samples: 4, swizzle: rgba) + "
               "TessellateStrokesRenderStep + SolidColor SrcOver" },
/* 110 */ { 9, "RP(color: Dawn(f=BGRA8,s=4) w/ msaa load, resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D16,s=4), samples: 4, swizzle: rgba) + "
               "TessellateCurvesRenderStep[Winding] + (empty)" },
/* 111 */ { 9, "RP(color: Dawn(f=BGRA8,s=4) w/ msaa load, resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D16,s=4), samples: 4, swizzle: rgba) + "
               "TessellateCurvesRenderStep[EvenOdd] + (empty)" },
/* 112 */ { 9, "RP(color: Dawn(f=BGRA8,s=4) w/ msaa load, resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D16,s=4), samples: 4, swizzle: rgba) + "
               "MiddleOutFanRenderStep[Winding] + (empty)" },
/* 113 */ { 9, "RP(color: Dawn(f=BGRA8,s=4) w/ msaa load, resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D16,s=4), samples: 4, swizzle: rgba) + "
               "MiddleOutFanRenderStep[EvenOdd] + (empty)" },
/* 114 */ { 9, "RP(color: Dawn(f=BGRA8,s=4) w/ msaa load, resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D16,s=4), samples: 4, swizzle: rgba) + "
               "CoverBoundsRenderStep[RegularCover] + SolidColor SrcOver" },
/* 115 */ { 9, "RP(color: Dawn(f=BGRA8,s=4) w/ msaa load, resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D16,s=4), samples: 4, swizzle: rgba) + "
               "CoverBoundsRenderStep[InverseCover] + SolidColor SrcOver" },
/* 116 */ { 9, "RP(color: Dawn(f=BGRA8,s=4) w/ msaa load, resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D16,s=4), samples: 4, swizzle: rgba) + "
               "CoverageMaskRenderStep + SolidColor SrcOver" },
/* 117 */ { 9, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "TessellateWedgesRenderStep[Winding] + (empty)" },
/* 118 */ { 9, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "TessellateWedgesRenderStep[EvenOdd] + (empty)" },
/* 119 */ { 9, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "TessellateWedgesRenderStep[Convex] + Compose [ SolidColor BlendCompose [ SolidColor Passthrough PorterDuffBlender ] ] SrcOver" },
/* 120 */ { 9, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "TessellateStrokesRenderStep + Compose [ SolidColor BlendCompose [ SolidColor Passthrough PorterDuffBlender ] ] SrcOver" },
/* 121 */ { 9, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "TessellateCurvesRenderStep[Winding] + (empty)" },
/* 122 */ { 9, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "TessellateCurvesRenderStep[EvenOdd] + (empty)" },
/* 123 */ { 9, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "PerEdgeAAQuadRenderStep + SolidColor SrcOver" },
/* 124 */ { 9, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "PerEdgeAAQuadRenderStep + SolidColor Src" },
/* 125 */ { 9, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "PerEdgeAAQuadRenderStep + SolidColor Clear" },
/* 126 */ { 9, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "PerEdgeAAQuadRenderStep + LocalMatrix [ Compose [ ImageShaderClamp(0) ColorSpaceTransformSRGB ] ] Src" },
/* 127 */ { 9, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "PerEdgeAAQuadRenderStep + LocalMatrix [ Compose [ ImageShaderClamp(0) ColorSpaceTransformPremul ] ] SrcOver" },
/* 128 */ { 9, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "PerEdgeAAQuadRenderStep + LocalMatrix [ Compose [ ImageShaderClamp(0) ColorSpaceTransformPremul ] ] Src" },
/* 129 */ { 9, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "PerEdgeAAQuadRenderStep + LocalMatrix [ Compose [ Image(0) ColorSpaceTransformSRGB ] ] Src" },
/* 130 */ { 9, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "PerEdgeAAQuadRenderStep + LocalMatrix [ Compose [ Image(0) ColorSpaceTransformPremul ] ] SrcOver" },
/* 131 */ { 9, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "PerEdgeAAQuadRenderStep + LocalMatrix [ Compose [ Image(0) ColorSpaceTransformPremul ] ] Src" },
/* 132 */ { 9, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "PerEdgeAAQuadRenderStep + LocalMatrix [ Compose [ HardwareImage(0) ColorSpaceTransformSRGB ] ] Src" },
/* 133 */ { 9, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "PerEdgeAAQuadRenderStep + LocalMatrix [ Compose [ HardwareImage(0) ColorSpaceTransformPremul ] ] SrcOver" },
/* 134 */ { 9, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "PerEdgeAAQuadRenderStep + LocalMatrix [ Compose [ HardwareImage(0) ColorSpaceTransformPremul ] ] Src" },
/* 135 */ { 9, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "PerEdgeAAQuadRenderStep + LocalMatrix [ Compose [ CubicImage(0) ColorSpaceTransformSRGB ] ] Src" },
/* 136 */ { 9, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "PerEdgeAAQuadRenderStep + LocalMatrix [ Compose [ CubicImage(0) ColorSpaceTransformPremul ] ] SrcOver" },
/* 137 */ { 9, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "PerEdgeAAQuadRenderStep + LocalMatrix [ Compose [ CubicImage(0) ColorSpaceTransformPremul ] ] Src" },
/* 138 */ { 9, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "MiddleOutFanRenderStep[Winding] + (empty)" },
/* 139 */ { 9, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "MiddleOutFanRenderStep[EvenOdd] + (empty)" },
/* 140 */ { 9, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "CoverBoundsRenderStep[RegularCover] + Compose [ SolidColor BlendCompose [ SolidColor Passthrough PorterDuffBlender ] ] SrcOver" },
/* 141 */ { 9, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "CoverBoundsRenderStep[NonAAFill] + SolidColor SrcOver" },
/* 142 */ { 9, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "CoverBoundsRenderStep[NonAAFill] + SolidColor Src" },
/* 143 */ { 9, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "CoverBoundsRenderStep[NonAAFill] + SolidColor Clear" },
/* 144 */ { 9, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "CoverBoundsRenderStep[NonAAFill] + LocalMatrix [ Compose [ ImageShaderClamp(0) ColorSpaceTransformSRGB ] ] Src" },
/* 145 */ { 9, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "CoverBoundsRenderStep[NonAAFill] + LocalMatrix [ Compose [ ImageShaderClamp(0) ColorSpaceTransformPremul ] ] SrcOver" },
/* 146 */ { 9, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "CoverBoundsRenderStep[NonAAFill] + LocalMatrix [ Compose [ ImageShaderClamp(0) ColorSpaceTransformPremul ] ] Src" },
/* 147 */ { 9, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "CoverBoundsRenderStep[NonAAFill] + LocalMatrix [ Compose [ Image(0) ColorSpaceTransformSRGB ] ] Src" },
/* 148 */ { 9, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "CoverBoundsRenderStep[NonAAFill] + LocalMatrix [ Compose [ Image(0) ColorSpaceTransformPremul ] ] SrcOver" },
/* 149 */ { 9, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "CoverBoundsRenderStep[NonAAFill] + LocalMatrix [ Compose [ Image(0) ColorSpaceTransformPremul ] ] Src" },
/* 150 */ { 9, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "CoverBoundsRenderStep[NonAAFill] + LocalMatrix [ Compose [ HardwareImage(0) ColorSpaceTransformSRGB ] ] Src" },
/* 151 */ { 9, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "CoverBoundsRenderStep[NonAAFill] + LocalMatrix [ Compose [ HardwareImage(0) ColorSpaceTransformPremul ] ] SrcOver AnalyticClip" },
/* 152 */ { 9, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "CoverBoundsRenderStep[NonAAFill] + LocalMatrix [ Compose [ HardwareImage(0) ColorSpaceTransformPremul ] ] SrcOver" },
/* 153 */ { 9, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "CoverBoundsRenderStep[NonAAFill] + LocalMatrix [ Compose [ HardwareImage(0) ColorSpaceTransformPremul ] ] Src" },
/* 154 */ { 9, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "CoverBoundsRenderStep[NonAAFill] + LocalMatrix [ Compose [ CubicImage(0) ColorSpaceTransformSRGB ] ] Src" },
/* 155 */ { 9, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "CoverBoundsRenderStep[NonAAFill] + LocalMatrix [ Compose [ CubicImage(0) ColorSpaceTransformPremul ] ] SrcOver" },
/* 156 */ { 9, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "CoverBoundsRenderStep[NonAAFill] + LocalMatrix [ Compose [ CubicImage(0) ColorSpaceTransformPremul ] ] Src" },
/* 157 */ { 9, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "CoverBoundsRenderStep[InverseCover] + Compose [ SolidColor BlendCompose [ SolidColor Passthrough PorterDuffBlender ] ] SrcOver" },
/* 158 */ { 9, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "CoverageMaskRenderStep + Compose [ SolidColor BlendCompose [ SolidColor Passthrough PorterDuffBlender ] ] SrcOver" },
/* 159 */ { 9, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "BitmapTextRenderStep[Mask] + SolidColor SrcOver" },
/* 160 */ { 9, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "AnalyticRRectRenderStep + SolidColor SrcOver" },
/* 161 */ { 9, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "AnalyticRRectRenderStep + SolidColor Src" },
/* 162 */ { 9, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "AnalyticRRectRenderStep + SolidColor Clear" },
/* 163 */ { 9, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "AnalyticRRectRenderStep + LocalMatrix [ Compose [ ImageShaderClamp(0) ColorSpaceTransformSRGB ] ] Src" },
/* 164 */ { 9, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "AnalyticRRectRenderStep + LocalMatrix [ Compose [ ImageShaderClamp(0) ColorSpaceTransformPremul ] ] SrcOver" },
/* 165 */ { 9, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "AnalyticRRectRenderStep + LocalMatrix [ Compose [ ImageShaderClamp(0) ColorSpaceTransformPremul ] ] Src" },
/* 166 */ { 9, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "AnalyticRRectRenderStep + LocalMatrix [ Compose [ Image(0) ColorSpaceTransformSRGB ] ] Src" },
/* 167 */ { 9, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "AnalyticRRectRenderStep + LocalMatrix [ Compose [ Image(0) ColorSpaceTransformPremul ] ] SrcOver" },
/* 168 */ { 9, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "AnalyticRRectRenderStep + LocalMatrix [ Compose [ Image(0) ColorSpaceTransformPremul ] ] Src" },
/* 169 */ { 9, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "AnalyticRRectRenderStep + LocalMatrix [ Compose [ HardwareImage(0) ColorSpaceTransformSRGB ] ] Src" },
/* 170 */ { 9, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "AnalyticRRectRenderStep + LocalMatrix [ Compose [ HardwareImage(0) ColorSpaceTransformPremul ] ] SrcOver" },
/* 171 */ { 9, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "AnalyticRRectRenderStep + LocalMatrix [ Compose [ HardwareImage(0) ColorSpaceTransformPremul ] ] Src" },
/* 172 */ { 9, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "AnalyticRRectRenderStep + LocalMatrix [ Compose [ CubicImage(0) ColorSpaceTransformSRGB ] ] Src" },
/* 173 */ { 9, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "AnalyticRRectRenderStep + LocalMatrix [ Compose [ CubicImage(0) ColorSpaceTransformPremul ] ] SrcOver" },
/* 174 */ { 9, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "AnalyticRRectRenderStep + LocalMatrix [ Compose [ CubicImage(0) ColorSpaceTransformPremul ] ] Src" },
    // AnalyticBlurRenderStep is currently internal only
    //{ 9, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
    //     "AnalyticBlurRenderStep + Compose [ SolidColor BlendCompose [ SolidColor Passthrough PorterDuffBlender ] ] SrcOver" },
    //{ 8, "RP(color: Dawn(f=BGRA8,s=4), resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
    //     "AnalyticBlurRenderStep + Compose [ SolidColor BlendCompose [ SolidColor Passthrough PorterDuffBlender ] ] SrcOver AnalyticClip" },
// LinearGradient?
/* 175 */ { 8, "RP(color: Dawn(f=BGRA8,s=4) w/ msaa load, resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "BitmapTextRenderStep[Mask] + LocalMatrix [ Compose [ LinearGradient4 ColorSpaceTransformPremul ] ] SrcOver" },
/* 176 */ { 8, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "PerEdgeAAQuadRenderStep + LocalMatrix [ Compose [ HardwareImage(0) ColorSpaceTransformPremul ] ] SrcOver AnalyticClip" },
/* 177 */ { 8, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "CoverBoundsRenderStep[NonAAFill] + LocalMatrix [ Compose [ HWYUVImage ColorSpaceTransformSRGB ] ] SrcOver" },
/* 178 */ { 7, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "PerEdgeAAQuadRenderStep + LocalMatrix [ Compose [ YUVImage ColorSpaceTransformSRGB ] ] SrcOver" },
/* 179 */ { 7, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "PerEdgeAAQuadRenderStep + LocalMatrix [ Compose [ HWYUVImage ColorSpaceTransformSRGB ] ] SrcOver" },
/* 180 */ { 7, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "CoverBoundsRenderStep[NonAAFill] + LocalMatrix [ Compose [ YUVImage ColorSpaceTransformSRGB ] ] SrcOver" },
// LinearGradient and a Dither?
/* 181 */ { 7, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "CoverBoundsRenderStep[NonAAFill] + Compose [ LocalMatrix [ Compose [ LinearGradient4 ColorSpaceTransformSRGB ] ] Dither ] SrcOver" },
    // AnalyticBlurRenderStep is currently internal only
    //{ 7, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
    //     "AnalyticBlurRenderStep + Compose [ SolidColor BlendCompose [ SolidColor Passthrough PorterDuffBlender ] ] SrcOver AnalyticClip" },
/* 182 */ { 6, "RP(color: Dawn(f=BGRA8,s=4), resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "PerEdgeAAQuadRenderStep + LocalMatrix [ Compose [ HardwareImage(0) ColorSpaceTransformPremul ] ] SrcOver AnalyticClip" },
/* 183 */ { 6, "RP(color: Dawn(f=BGRA8,s=4) w/ msaa load, resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "PerEdgeAAQuadRenderStep + LocalMatrix [ Compose [ YUVImage ColorSpaceTransformSRGB ] ] SrcOver" },
/* 184 */ { 6, "RP(color: Dawn(f=BGRA8,s=4) w/ msaa load, resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "PerEdgeAAQuadRenderStep + LocalMatrix [ Compose [ HWYUVImage ColorSpaceTransformSRGB ] ] SrcOver AnalyticClip" },
/* 185 */ { 6, "RP(color: Dawn(f=BGRA8,s=4) w/ msaa load, resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "PerEdgeAAQuadRenderStep + LocalMatrix [ Compose [ HardwareImage(0) ColorSpaceTransformPremul ] ] SrcOver AnalyticClip" },
/* 186 */ { 6, "RP(color: Dawn(f=BGRA8,s=4) w/ msaa load, resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "CoverBoundsRenderStep[NonAAFill] + LocalMatrix [ Compose [ YUVImage ColorSpaceTransformSRGB ] ] SrcOver" },
/* 187 */ { 6, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "PerEdgeAAQuadRenderStep + LocalMatrix [ Compose [ YUVImage ColorSpaceTransformSRGB ] ] SrcOver AnalyticClip" },
/* 188 */ { 6, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "PerEdgeAAQuadRenderStep + LocalMatrix [ Compose [ HWYUVImage ColorSpaceTransformSRGB ] ] SrcOver AnalyticClip" },
/* 189 */ { 6, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "CoverBoundsRenderStep[NonAAFill] + SolidColor SrcOver AnalyticClip" },
/* 190 */ { 5, "RP(color: Dawn(f=R8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: a000) + "
               "CoverBoundsRenderStep[NonAAFill] + SolidColor SrcOver" },
          // blurs need to wait
/* 191 */ { 5, "RP(color: Dawn(f=R8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: a000) + "
               "CoverBoundsRenderStep[NonAAFill] + KnownRuntimeEffect_1DBlur16 [ LocalMatrix [ Compose [ ImageShaderClamp(0) ColorSpaceTransform ] ] ] Src" },
/* 192 */ { 5, "RP(color: Dawn(f=R8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: a000) + "
               "AnalyticRRectRenderStep + SolidColor SrcOver" },
// LinearGradient
/* 193 */ { 5, "RP(color: Dawn(f=BGRA8,s=4), resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "CoverBoundsRenderStep[NonAAFill] + Compose [ LocalMatrix [ Compose [ LinearGradient4 ColorSpaceTransformSRGB ] ] Dither ] SrcOver" },
/* 194 */ { 5, "RP(color: Dawn(f=BGRA8,s=4), resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "CoverageMaskRenderStep + Compose [ SolidColor BlendCompose [ SolidColor Passthrough PorterDuffBlender ] ] SrcOver AnalyticClip" },
// LinearGradient
/* 195 */ { 5, "RP(color: Dawn(f=BGRA8,s=4), resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "BitmapTextRenderStep[Mask] + LocalMatrix [ Compose [ LinearGradient4 ColorSpaceTransformPremul ] ] SrcOver" },
// LinearGradient + Dither
/* 196 */ { 5, "RP(color: Dawn(f=BGRA8,s=4) w/ msaa load, resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
               "CoverBoundsRenderStep[NonAAFill] + Compose [ LocalMatrix [ Compose [ LinearGradient4 ColorSpaceTransformSRGB ] ] Dither ] SrcOver" },
// AnalyticBlurRenderStep is currently internal only
//{ 5, "RP(color: Dawn(f=BGRA8,s=4) w/ msaa load, resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D24_S8,s=4), samples: 4, swizzle: rgba) + "
// "AnalyticBlurRenderStep + Compose [ SolidColor BlendCompose [ SolidColor Passthrough PorterDuffBlender ] ] SrcOver AnalyticClip" },
/* 197 */ { 5, "RP(color: Dawn(f=BGRA8,s=4) w/ msaa load, resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D16,s=4), samples: 4, swizzle: rgba) + "
               "CoverBoundsRenderStep[NonAAFill] + SolidColor SrcOver" },
/* 198 */ { 5, "RP(color: Dawn(f=BGRA8,s=4) w/ msaa load, resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D16,s=4), samples: 4, swizzle: rgba) + "
               "BitmapTextRenderStep[Mask] + SolidColor SrcOver" },
// DstIn?
/* 199 */ { 5, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "PerEdgeAAQuadRenderStep + LocalMatrix [ Compose [ HardwareImage(0) ColorSpaceTransformPremul ] ] DstIn" },
// Luma
/* 200 */ { 5, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "PerEdgeAAQuadRenderStep + Compose [ LocalMatrix [ Compose [ HardwareImage(0) ColorSpaceTransformPremul ] ] KnownRuntimeEffect_Luma ] SrcOver" },
// Alpha-only
/* 201 */ { 5, "RP(color: Dawn(f=BGRA8,s=1), resolve: {}, ds: Dawn(f=D16,s=1), samples: 1, swizzle: rgba) + "
               "CoverBoundsRenderStep[NonAAFill] + BlendCompose [ LocalMatrix [ Compose [ ImageShaderClamp(0) ColorSpaceTransformPremul ] ] AlphaOnlyPaintColor SrcIn ] SrcOver" },

// This label is created by hand. It is a copy of #198 but w/o the "w/ msaa load" string. The issue
// is that, apparently, Dawn Mac on Intel doesn't require the "w/ msaa load" string and that is
// messing up the every-PrecompileSettings-covers-something-in-kCases assert.
/* 202 */ { 0, "RP(color: Dawn(f=BGRA8,s=4), resolve: Dawn(f=BGRA8,s=1), ds: Dawn(f=D16,s=4), samples: 4, swizzle: rgba) + "
               "BitmapTextRenderStep[Mask] + SolidColor SrcOver" },
    };

[[maybe_unused]] void find_duplicates(SkSpan<const ChromePipeline> cases) {
    for (size_t i = 0; i < std::size(cases); ++i) {
        for (size_t j = i+1; j < std::size(cases); ++j) {
            if (!strcmp(cases[i].fString, cases[j].fString)) {
                SkDebugf("Duplicate %zu && %zu\n", i, j);
            }
        }
    }
}

std::string rm_whitespace(const std::string& s) {
    auto start = s.find_first_not_of(' ');
    auto end = s.find_last_not_of(' ');
    return s.substr(start, (end - start) + 1);
}

// Precompile with the provided PrecompileSettings then verify that:
//   1) some case in 'kCases' is covered
//   2) more than 40% of the generated Pipelines are in kCases
void run_test(skgpu::graphite::PrecompileContext* precompileContext,
              skiatest::Reporter* reporter,
              const PrecompileSettings& settings,
              int precompileSettingsIndex,
              std::vector<bool>* casesThatAreMatched) {
    using namespace skgpu::graphite;

    precompileContext->priv().globalCache()->resetGraphicsPipelines();

    Precompile(precompileContext,
               settings.fPaintOptions,
               settings.fDrawTypeFlags,
               { &settings.fRenderPassProps, 1 });

    std::set<std::string> generatedLabels;

    {
        const RendererProvider* rendererProvider = precompileContext->priv().rendererProvider();
        const ShaderCodeDictionary* dict = precompileContext->priv().shaderCodeDictionary();

        std::vector<skgpu::UniqueKey> generatedKeys;

        UniqueKeyUtils::FetchUniqueKeys(precompileContext, &generatedKeys);

        for (const skgpu::UniqueKey& key : generatedKeys) {
            GraphicsPipelineDesc pipelineDesc;
            RenderPassDesc renderPassDesc;
            UniqueKeyUtils::ExtractKeyDescs(precompileContext, key, &pipelineDesc, &renderPassDesc);

            const RenderStep* renderStep = rendererProvider->lookup(pipelineDesc.renderStepID());
            std::string tmp = GetPipelineLabel(dict, renderPassDesc, renderStep,
                                               pipelineDesc.paintParamsID());
            generatedLabels.insert(rm_whitespace(tmp));
        }
    }

    std::vector<bool> localMatches;
    std::vector<size_t> matchesInCases;

    for (const std::string& g : generatedLabels) {
        bool didThisLabelMatch = false;
        for (size_t j = 0; j < std::size(kCases); ++j) {
            const char* testStr = kCases[j].fString;
            if (!strcmp(g.c_str(), testStr)) {

#if defined(SK_DEBUG)
                DrawTypeFlags expectedFlags = get_draw_type_flags(testStr);
                SkASSERT(expectedFlags == settings.fDrawTypeFlags);
                RenderPassProperties expectedRPP = get_render_pass_properties(testStr);
                if (strstr(testStr, "ColorSpaceTransformSRGB")) {
                    expectedRPP.fDstCS = SkColorSpace::MakeSRGB();
                }
                SkASSERT(expectedRPP == settings.fRenderPassProps);
#endif

                didThisLabelMatch = true;
                matchesInCases.push_back(j);
                (*casesThatAreMatched)[j] = true;
            }
        }

        localMatches.push_back(didThisLabelMatch);
    }

    REPORTER_ASSERT(reporter, matchesInCases.size() >= 1,   // This tests requirement 1, above
                    "%d: num matches: %zu", precompileSettingsIndex, matchesInCases.size());
    float utilization = ((float) matchesInCases.size())/generatedLabels.size();
    REPORTER_ASSERT(reporter, utilization >= 0.4f,         // This tests requirement 2, above
                    "%d: utilization: %f", precompileSettingsIndex, utilization);

#if defined(PRINT_COVERAGE)
    // This block will print out all the cases in 'kCases' that the given PrecompileSettings
    // covered.
    sort(matchesInCases.begin(), matchesInCases.end());
    SkDebugf("precompile case %d handles %zu/%zu cases (%.2f utilization): ",
             precompileSettingsIndex, matchesInCases.size(), generatedLabels.size(), utilization);
    for (size_t h : matchesInCases) {
        SkDebugf("%zu ", h);
    }
    SkDebugf("\n");
#endif

#if defined(PRINT_GENERATED_LABELS)
    // This block will print out all the labels from the given PrecompileSettings marked with
    // whether they were found in 'kCases'. This is useful for analyzing the set of Pipelines
    // generated by a single PrecompileSettings and is usually used along with 'kChosenCase'.
    SkASSERT(localMatches.size() == generatedLabels.size());

    int index = 0;
    for (const std::string& g : generatedLabels) {
        SkDebugf("%c %d: %s\n", localMatches[index] ? 'h' : ' ', index, g.c_str());
        ++index;
    }
#endif
}

[[maybe_unused]] bool skip(const char* str) {
    if (strstr(str, "AnalyticClip")) {  // we have to think about this a bit more
        return true;
    }
    if (strstr(str, "KnownRuntimeEffect_1DBlur16")) {  // we have to revise how we do blurring
        return true;
    }
    if (strstr(str, "LinearGradient4")) {  // this seems too specialized
        return true;
    }
    if (strstr(str, "KnownRuntimeEffect_Luma")) {  // this also seems too specialized
        return true;
    }

    return false;
}

// The pipeline strings were created using the Dawn Metal backend so that is the only viable
// comparison
bool is_dawn_metal_context_type(skgpu::ContextType type) {
    return type == skgpu::ContextType::kDawn_Metal;
}

} // anonymous namespace

// This test verifies that for each case in 'kPrecompileCases':
//    1) it covers some pipeline(s) in 'kCases'
//    2) more than 40% of the generated Precompile Pipelines are used (i.e., that over-generation
//        isn't too out of control).
// Optionally, it can also:
//    FINAL_REPORT:   Print out a final report that includes missed cases in 'kCases'
//    PRINT_COVERAGE: list the cases (in 'kCases') that are covered by each 'kPrecompileCases' case
//    PRINT_GENERATED_LABELS: list the Pipeline labels for a specific 'kPrecompileCases' case
// Also of note, the "skip" method documents the Pipelines we're intentionally skipping and why.
DEF_GRAPHITE_TEST_FOR_CONTEXTS(ChromePrecompileTest, is_dawn_metal_context_type,
                               reporter, context, /* testContext */, CtsEnforcement::kNever) {
    using namespace skgpu::graphite;

    std::unique_ptr<PrecompileContext> precompileContext = context->makePrecompileContext();
    const skgpu::graphite::Caps* caps = precompileContext->priv().caps();

    TextureInfo textureInfo = caps->getDefaultSampledTextureInfo(kBGRA_8888_SkColorType,
                                                                 skgpu::Mipmapped::kNo,
                                                                 skgpu::Protected::kNo,
                                                                 skgpu::Renderable::kYes);

    TextureInfo msaaTex = caps->getDefaultMSAATextureInfo(textureInfo, Discardable::kYes);

    if (msaaTex.numSamples() <= 1) {
        // The following pipelines rely on having MSAA
        return;
    }

#ifdef SK_ENABLE_VELLO_SHADERS
    if (caps->computeSupport()) {
        // The following pipelines rely on not utilizing Vello
        return;
    }
#endif

    std::vector<bool> casesThatAreMatched(std::size(kCases), false);

    static const size_t kChosenCase = -1;  // only test this entry in 'kPrecompileCases'
    for (size_t i = 0; i < std::size(kPrecompileCases); ++i) {
        if (kChosenCase != -1 && kChosenCase != i) {
            continue;
        }

        run_test(precompileContext.get(), reporter,
                 kPrecompileCases[i], i, &casesThatAreMatched);
    }

#if defined(FINAL_REPORT)
    // This block prints out a final report. This includes a list of the cases in 'kCases' that
    // were not covered by the PaintOptions.
    int numCovered = 0, numNotCovered = 0, numIntentionallySkipped = 0;
    SkDebugf("not covered: ");
    for (size_t i = 0; i < std::size(kCases); ++i) {
        if (!casesThatAreMatched[i]) {
            if (skip(kCases[i].fString)) {
                ++numIntentionallySkipped;
            } else {
                SkDebugf("%zu, ", i);
                ++numNotCovered;
            }
        } else {
            ++numCovered;
        }
    }
    SkDebugf("\n");
    SkDebugf("covered %d notCovered %d skipped %d total %zu\n",
             numCovered, numNotCovered, numIntentionallySkipped,
             std::size(kCases));
#endif
}

#endif // SK_GRAPHITE
