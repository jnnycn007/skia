/*
 * Copyright 2018 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef GrVkTypesPriv_DEFINED
#define GrVkTypesPriv_DEFINED

#include "include/gpu/ganesh/vk/GrVkTypes.h"
#include "include/gpu/vk/VulkanTypes.h"
#include "include/private/gpu/vk/SkiaVulkan.h"

#include <cstdint>

namespace skgpu {
class MutableTextureState;
enum class Protected : bool;
}

GrVkImageInfo GrVkImageInfoWithMutableState(const GrVkImageInfo&,
                                            const skgpu::MutableTextureState*);

struct GrVkImageSpec {
    GrVkImageSpec()
            : fImageTiling(VK_IMAGE_TILING_OPTIMAL)
            , fFormat(VK_FORMAT_UNDEFINED)
            , fImageUsageFlags(0)
            , fSharingMode(VK_SHARING_MODE_EXCLUSIVE) {}

    GrVkImageSpec(const GrVkSurfaceInfo& info)
            : fImageTiling(info.fImageTiling)
            , fFormat(info.fFormat)
            , fImageUsageFlags(info.fImageUsageFlags)
            , fYcbcrConversionInfo(info.fYcbcrConversionInfo)
            , fSharingMode(info.fSharingMode) {}

    VkImageTiling fImageTiling;
    VkFormat fFormat;
    VkImageUsageFlags fImageUsageFlags;
    skgpu::VulkanYcbcrConversionInfo fYcbcrConversionInfo;
    VkSharingMode fSharingMode;
};

GrVkSurfaceInfo GrVkImageSpecToSurfaceInfo(const GrVkImageSpec& vkSpec,
                                           uint32_t sampleCount,
                                           uint32_t levelCount,
                                           skgpu::Protected isProtected);

#endif
