/*
 * Copyright 2011 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include "src/effects/colorfilters/SkMatrixColorFilter.h"

#include "include/core/SkColorFilter.h"
#include "include/core/SkRefCnt.h"
#include "include/core/SkScalar.h"
#include "include/effects/SkColorMatrix.h"
#include "include/private/base/SkAssert.h"
#include "include/private/base/SkFloatingPoint.h"
#include "src/core/SkEffectPriv.h"
#include "src/core/SkPicturePriv.h"
#include "src/core/SkRasterPipeline.h"
#include "src/core/SkRasterPipelineOpList.h"
#include "src/core/SkReadBuffer.h"
#include "src/core/SkWriteBuffer.h"
#include "src/effects/colorfilters/SkColorFilterBase.h"

#include <array>
#include <cstring>

static bool is_alpha_unchanged(const float matrix[20]) {
    const float* srcA = matrix + 15;

    return SkScalarNearlyZero(srcA[0]) && SkScalarNearlyZero(srcA[1]) &&
           SkScalarNearlyZero(srcA[2]) && SkScalarNearlyEqual(srcA[3], 1) &&
           SkScalarNearlyZero(srcA[4]);
}

SkMatrixColorFilter::SkMatrixColorFilter(const float array[20], Domain domain, Clamp clamp)
        : fAlphaIsUnchanged(is_alpha_unchanged(array)), fDomain(domain), fClamp(clamp) {
    memcpy(fMatrix, array, 20 * sizeof(float));
}

void SkMatrixColorFilter::flatten(SkWriteBuffer& buffer) const {
    SkASSERT(sizeof(fMatrix) / sizeof(float) == 20);
    buffer.writeScalarArray(fMatrix);

    // RGBA flag
    buffer.writeBool(fDomain == Domain::kRGBA);
    buffer.writeBool(fClamp == Clamp::kYes);
}

sk_sp<SkFlattenable> SkMatrixColorFilter::CreateProc(SkReadBuffer& buffer) {
    float matrix[20];
    if (!buffer.readScalarArray(matrix)) {
        return nullptr;
    }

    auto is_rgba = buffer.readBool();
    Clamp clamp = buffer.isVersionLT(SkPicturePriv::kUnclampedMatrixColorFilter)
                          ? Clamp::kYes
                          : (buffer.readBool() ? Clamp::kYes : Clamp::kNo);
    // clamp option is ignored for HSL-domain filters
    return is_rgba ? SkColorFilters::Matrix(matrix, clamp) : SkColorFilters::HSLAMatrix(matrix);
}

bool SkMatrixColorFilter::onAsAColorMatrix(float matrix[20]) const {
    if (matrix) {
        memcpy(matrix, fMatrix, 20 * sizeof(float));
    }
    return true;
}

bool SkMatrixColorFilter::appendStages(const SkStageRec& rec, bool shaderIsOpaque) const {
    const bool willStayOpaque = shaderIsOpaque && fAlphaIsUnchanged,
               hsla = fDomain == Domain::kHSLA,
               clamp = fClamp == Clamp::kYes;

    SkRasterPipeline* p = rec.fPipeline;
    if (!shaderIsOpaque) {
        p->append(SkRasterPipelineOp::unpremul);
    }
    if (hsla) {
        p->append(SkRasterPipelineOp::rgb_to_hsl);
    }
    if (true) {
        p->append(SkRasterPipelineOp::matrix_4x5, fMatrix);
    }
    if (hsla) {
        p->append(SkRasterPipelineOp::hsl_to_rgb);
    }
    if (clamp) {
        p->append(SkRasterPipelineOp::clamp_01);
    } else {
        // We still need to clamp alpha, regardless
        p->append(SkRasterPipelineOp::clamp_a_01);
    }
    if (!willStayOpaque) {
        p->append(SkRasterPipelineOp::premul);
    }
    return true;
}

///////////////////////////////////////////////////////////////////////////////

static sk_sp<SkColorFilter> MakeMatrix(const float array[20],
                                       SkMatrixColorFilter::Domain domain,
                                       SkColorFilters::Clamp clamp) {
    if (!SkIsFinite(array, 20)) {
        return nullptr;
    }
    return sk_make_sp<SkMatrixColorFilter>(array, domain, clamp);
}

sk_sp<SkColorFilter> SkColorFilters::Matrix(const float array[20], Clamp clamp) {
    return MakeMatrix(array, SkMatrixColorFilter::Domain::kRGBA, clamp);
}

sk_sp<SkColorFilter> SkColorFilters::Matrix(const SkColorMatrix& cm, Clamp clamp) {
    return MakeMatrix(cm.fMat.data(), SkMatrixColorFilter::Domain::kRGBA, clamp);
}

sk_sp<SkColorFilter> SkColorFilters::HSLAMatrix(const float array[20]) {
    return MakeMatrix(array, SkMatrixColorFilter::Domain::kHSLA, Clamp::kYes);
}

sk_sp<SkColorFilter> SkColorFilters::HSLAMatrix(const SkColorMatrix& cm) {
    return MakeMatrix(cm.fMat.data(), SkMatrixColorFilter::Domain::kHSLA, Clamp::kYes);
}

void SkRegisterMatrixColorFilterFlattenable() {
    SK_REGISTER_FLATTENABLE(SkMatrixColorFilter);
    // Previous name
    SkFlattenable::Register("SkColorFilter_Matrix", SkMatrixColorFilter::CreateProc);
}
