/*
 * Copyright 2016 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef SkSVGClipPath_DEFINED
#define SkSVGClipPath_DEFINED

#include "include/core/SkPath.h"
#include "include/core/SkRefCnt.h"
#include "include/private/base/SkAPI.h"
#include "modules/svg/include/SkSVGHiddenContainer.h"
#include "modules/svg/include/SkSVGNode.h"
#include "modules/svg/include/SkSVGTypes.h"

class SkSVGRenderContext;

class SK_API SkSVGClipPath final : public SkSVGHiddenContainer {
public:
    static sk_sp<SkSVGClipPath> Make() {
        return sk_sp<SkSVGClipPath>(new SkSVGClipPath());
    }

    SVG_ATTR(ClipPathUnits, SkSVGObjectBoundingBoxUnits,
             SkSVGObjectBoundingBoxUnits(SkSVGObjectBoundingBoxUnits::Type::kUserSpaceOnUse))

private:
    friend class SkSVGRenderContext;

    SkSVGClipPath();

    bool parseAndSetAttribute(const char*, const char*) override;

    SkPath resolveClip(const SkSVGRenderContext&) const;

    using INHERITED = SkSVGHiddenContainer;
};

#endif // SkSVGClipPath_DEFINED
