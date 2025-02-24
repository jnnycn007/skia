/*
 * Copyright 2021 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef skgpu_graphite_MtlGraphiteTypesPriv_DEFINED
#define skgpu_graphite_MtlGraphiteTypesPriv_DEFINED

#include "include/core/SkString.h"
#include "include/gpu/graphite/GraphiteTypes.h"
#include "include/gpu/graphite/TextureInfo.h"
#include "include/gpu/graphite/mtl/MtlGraphiteTypes.h"
#include "include/ports/SkCFObject.h"

class SkStream;
class SkWStream;

///////////////////////////////////////////////////////////////////////////////

#include <TargetConditionals.h>

// We're using the MSL version as shorthand for the Metal SDK version here
#if defined(SK_BUILD_FOR_MAC)
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 130000
#define SKGPU_GRAPHITE_METAL_SDK_VERSION 300
#elif __MAC_OS_X_VERSION_MAX_ALLOWED >= 120000
#define SKGPU_GRAPHITE_METAL_SDK_VERSION 240
#elif __MAC_OS_X_VERSION_MAX_ALLOWED >= 110000
#define SKGPU_GRAPHITE_METAL_SDK_VERSION 230
#else
#error Must use at least 11.00 SDK to build Metal backend for MacOS
#endif
#else
#if __IPHONE_OS_VERSION_MAX_ALLOWED >= 160000 || __TV_OS_VERSION_MAX_ALLOWED >= 160000
#define SKGPU_GRAPHITE_METAL_SDK_VERSION 300
#elif __IPHONE_OS_VERSION_MAX_ALLOWED >= 150000 || __TV_OS_VERSION_MAX_ALLOWED >= 150000
#define SKGPU_GRAPHITE_METAL_SDK_VERSION 240
#elif __IPHONE_OS_VERSION_MAX_ALLOWED >= 140000 || __TV_OS_VERSION_MAX_ALLOWED >= 140000
#define SKGPU_GRAPHITE_METAL_SDK_VERSION 230
#else
#error Must use at least 14.00 SDK to build Metal backend for iOS
#endif
#endif

#import <Metal/Metal.h>

namespace skgpu {
class ShaderErrorHandler;
}

namespace skgpu::graphite {

class MtlSharedContext;

sk_cfp<id<MTLLibrary>> MtlCompileShaderLibrary(const MtlSharedContext* sharedContext,
                                               std::string_view label,
                                               std::string_view msl,
                                               ShaderErrorHandler* errorHandler);

struct MtlTextureSpec {
    MtlTextureSpec()
            : fFormat(MTLPixelFormatInvalid)
            , fUsage(MTLTextureUsageUnknown)
            , fStorageMode(MTLStorageModeShared)
            , fFramebufferOnly(false) {}
    MtlTextureSpec(const MtlTextureInfo& info)
            : fFormat(info.fFormat)
            , fUsage(info.fUsage)
            , fStorageMode(info.fStorageMode)
            , fFramebufferOnly(info.fFramebufferOnly) {}

    bool operator==(const MtlTextureSpec& that) const {
        return fFormat == that.fFormat && fUsage == that.fUsage &&
               fStorageMode == that.fStorageMode && fFramebufferOnly == that.fFramebufferOnly;
    }

    bool isCompatible(const MtlTextureSpec& that) const {
        // The usages may match or the usage passed in may be a superset of the usage stored within.
        return fFormat == that.fFormat && fStorageMode == that.fStorageMode &&
               fFramebufferOnly == that.fFramebufferOnly && (fUsage & that.fUsage) == fUsage;
    }

    SkString toString() const {
        return SkStringPrintf("format=%u,usage=0x%04X,storageMode=%u,framebufferOnly=%d",
                              (uint32_t)fFormat,
                              (uint32_t)fUsage,
                              (uint32_t)fStorageMode,
                              fFramebufferOnly);
    }

    bool serialize(SkWStream*) const;
    static bool Deserialize(SkStream*, MtlTextureSpec* out);

    MTLPixelFormat fFormat;
    MTLTextureUsage fUsage;
    MTLStorageMode fStorageMode;
    bool fFramebufferOnly;
};

MtlTextureInfo MtlTextureSpecToTextureInfo(const MtlTextureSpec& mtlSpec,
                                           uint32_t sampleCount,
                                           Mipmapped mipmapped);

namespace TextureInfos {
MtlTextureSpec GetMtlTextureSpec(const TextureInfo&);
MTLPixelFormat GetMTLPixelFormat(const TextureInfo&);
MTLTextureUsage GetMTLTextureUsage(const TextureInfo&);
bool GetMtlFramebufferOnly(const TextureInfo&);
}  // namespace TextureInfos

}  // namespace skgpu::graphite

#endif  // skgpu_graphite_MtlGraphiteTypesPriv_DEFINED
