//
// RT64
//

#include <algorithm>
#include <cassert>
#include <memory.h>
#include <stdio.h>

#include "common/rt64_common.h"
#include "gbi/rt64_f3d.h"

#include "rt64_vi.h"

namespace RT64 {
    // VI
    
    hlslpp::float4 VI::viewRectangle() const {
        return { 0.0f, 0.0f, 1.0f, 1.0f };
    }

    hlslpp::float4 VI::cropRectangle() const {
        // [wcw fix] Crop a few SD pixels from every edge of the scanout, like a CRT's
        // overscan did on real hardware. WCW leaves garbage it never draws near the
        // framebuffer edges: rows 0-3 at the top (the game parks the VI origin at
        // fb+1 row to hide row 0; a CRT's bezel hid the rest) and a ~16x8 px block at
        // the bottom-left (fb rows ~232-239; hardware showed only rows 1..237 of the
        // fb — VI v=37..511 — and the bezel covered the remainder). Measured on the
        // 480-wide match mode, 2026-07-05. Units: standard SD pixels (320x240 space);
        // env WCW_CROP overrides as a single value or "L,T,R,B". The renderer turns
        // this into the present scissor, so cropped edges show the border color.
        static const hlslpp::float4 cropSD = [] {
            float l = 4.0f, t = 4.0f, r = 4.0f, b = 8.0f;
            const char *env = getenv("WCW_CROP");
            if (env != nullptr) {
                float v[4];
                const int n = sscanf(env, "%f,%f,%f,%f", &v[0], &v[1], &v[2], &v[3]);
                if (n == 4) { l = v[0]; t = v[1]; r = v[2]; b = v[3]; }
                else if (n == 1) { l = t = r = b = v[0]; }
            }
            return hlslpp::float4(l, t, r, b);
        }();
        const float fl = cropSD.x / 320.0f, ft = cropSD.y / 240.0f;
        const float fr = cropSD.z / 320.0f, fb = cropSD.w / 240.0f;
        return { fl, ft, 1.0f - fl - fr, 1.0f - ft - fb };
    }

    float VI::gamma() const {
        const float GammaCorrection = 1.0f / 2.2f;
        return status.gammaEnable ? GammaCorrection : 1.0f;
    }

    bool VI::compatibleWith(const VI &vi) const {
        return
            (width == vi.width) &&
            (hRegion.hStart == vi.hRegion.hStart) &&
            (hRegion.hEnd == vi.hRegion.hEnd) &&
            (vRegion.vStart == vi.vRegion.vStart) &&
            (vRegion.vEnd == vi.vRegion.vEnd) &&
            (xTransform.xScale == vi.xTransform.xScale) &&
            (xTransform.xOffset == vi.xTransform.xOffset) &&
            (yTransform.yScale == vi.yTransform.yScale) &&
            (yTransform.yOffset == vi.yTransform.yOffset);
    }

    bool VI::visible() const {
        return (status.type != VI_STATUS_TYPE_BLANK) && (hRegion.hStart > 0);
    }

    bool VI::operator!=(const VI &rhs) const {
        return
            (status.word != rhs.status.word) ||
            (origin != rhs.origin) ||
            (width != rhs.width) ||
            (intr != rhs.intr) ||
            (vCurrentLine != rhs.vCurrentLine) ||
            (burst.word != rhs.burst.word) ||
            (vSync != rhs.vSync) ||
            (hSync.word != rhs.hSync.word) ||
            (leap.word != rhs.leap.word) ||
            (hRegion.word != rhs.hRegion.word) ||
            (vRegion.word != rhs.vRegion.word) ||
            (vBurst.word != rhs.vBurst.word) ||
            (xTransform.word != rhs.xTransform.word) ||
            (yTransform.word != rhs.yTransform.word);
    }

    uint8_t VI::fbSiz() const {
        switch (status.type) {
        case VI_STATUS_TYPE_16_BIT:
            return G_IM_SIZ_16b;
        case VI_STATUS_TYPE_32_BIT:
            return G_IM_SIZ_32b;
        case VI_STATUS_TYPE_BLANK:
        default:
            return 0;
        }
    }

    uint32_t VI::fbAddress() const {
        uint8_t siz = fbSiz();

        // Estimate the origin is off by one or two rows.
        // ([wcw] note: for WCW this subtraction is REQUIRED — the game sets origin =
        // fb + 1 row, and without the subtraction the framebuffer lookup misses and
        // every present falls back to the slow native-res scratch path. The garbage
        // row it re-exposes at the top is handled by cropRectangle() instead.)
        if (siz >= G_IM_SIZ_16b) {
            const bool interlacedStep = status.serrate && (vCurrentLine & 0x1);
            const uint32_t rowBytes = width * (1U << (siz - 1));
            const uint32_t rowCount = interlacedStep ? 2 : 1;
            const uint32_t rowOffset = rowBytes * rowCount;
            if (origin >= rowOffset) {
                return origin - rowOffset;
            }
        }

        return origin;
    }

    hlslpp::uint2 VI::fbSize() const {
        hlslpp::uint2 size = { width, 0 };
        
        // In interlaced without deflickering, the stride of the framebuffer is usually double of 
        // what its actual row size is. We detect for such a case and return half the width.
        if (status.serrate) {
            const float estimatedWidth = (hRegion.hEnd - hRegion.hStart) / xScaleFloat();
            const float interlacedTolerance = 1.875f;
            if (estimatedWidth < (width / interlacedTolerance)) {
                size.x = width / 2;
            }
        }

        // We can make a close estimate of the height the framebuffer will use by using the width
        // that was just fixed to eliminate interlacing.
        size.y = lround(float(vRegion.vEnd - vRegion.vStart) / (2.0f * yScaleFloat() * (float(size.x) / float(width))));

        if ((size.x > 0) && (size.y > 0)) {
            // Most of the time, the height is missing a few rows because the framebuffer is offset
            // at the origin and an extra row is left at the end to account for filtering.
            // We add two extra rows to whatever result we get and try to get the closest clean
            // multiplier of the specified Division factor.
            const uint32_t ExtraRows = 2;
            const uint32_t Divisor = 4;
            size.y += ExtraRows;
            size.y = lround(float(size.y) / Divisor) * Divisor;
            return size;
        } else {
            return hlslpp::uint2(0, 0);
        }
    }

    float VI::xScaleFloat() const {
        return (1024.0f / xTransform.xScale);
    }

    float VI::xOffsetFloat() const {
        return xTransform.xOffset / 1024.0f;
    }

    float VI::yScaleFloat() const {
        return (1024.0f / yTransform.yScale);
    }

    float VI::yOffsetFloat() const {
        return yTransform.yOffset / 1024.0f;
    }

    // VIHistory

    VIHistory::VIHistory() {
        historyCursor = 0;
        factorCursor = 0;
        history.fill({});
        factors.fill(0);
    }

    void VIHistory::pushVI(const VI &vi, uint32_t fbWidth) {
        historyCursor = (historyCursor + 1) % history.size();
        Present &entry = history[historyCursor];
        entry.vi = vi;
        entry.fbWidth = fbWidth;
    }

    void VIHistory::pushFactor(uint32_t factor) {
        factorCursor = (factorCursor + 1) % factors.size();
        factors[factorCursor] = factor;
    }

    uint32_t VIHistory::logicalRateFromFactors() {
        // [wcw fix] Tolerate a single-sample outlier instead of requiring strict unanimous
        // agreement across all history slots: real gameplay VI timing has occasional one-tick
        // jitter (camera cuts, hit effects), which previously zeroed out rate detection for a
        // beat and made interpolation flicker on/off instead of running consistently. Accept a
        // 2-of-3 majority match instead of requiring all 3 to agree.
        for (size_t i = 0; i < factors.size(); i++) {
            if (factors[i] == 0) {
                continue;
            }

            uint32_t matches = 0;
            for (size_t j = 0; j < factors.size(); j++) {
                if (factors[j] == factors[i]) {
                    matches++;
                }
            }

            if (matches >= 2) {
                const uint32_t FullRate = 60; // TODO: PAL support.
                return FullRate / factors[i];
            }
        }

        return 0;
    }

    const VIHistory::Present &VIHistory::top() const {
        return history[historyCursor];
    }
};