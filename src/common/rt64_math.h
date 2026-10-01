//
// RT64
//

#pragma once

#include <chrono>
#include "rt64_common.h"

#include <atomic>

namespace RT64 {
    // [wcw fix] Set once frame matching has started (60Hz target above the game's rate). The 60Hz changes
    // for this game apply only while it is set, so the renderer behaves as stock RT64 until then.
    extern std::atomic<bool> wcwInterpActive;
    // [wcw fix] Number of layers in the last completed multi-layer frame (a "run").
    extern std::atomic<uint32_t> wcwLastRunLength;
    // [wcw fix] Per rotating target set (run serial % 3): the run whose real frame it holds, and that run
    // once it has finished rendering.
    extern std::atomic<uint64_t> wcwSlotRealSerial[3];
    extern std::atomic<uint64_t> wcwSlotRealComplete[3];
    // [wcw fix] Per game colour buffer (4 slots keyed by (address >> 10) & 3): the run whose blend it holds,
    // and that run once it has finished rendering.
    extern std::atomic<uint32_t> wcwAddrSlotAddr[4];
    extern std::atomic<uint64_t> wcwAddrSlotSerial[4];
    extern std::atomic<uint64_t> wcwAddrSlotComplete[4];
    // [wcw fix] Bound for the frame-pacing waits between the render and present threads (two refreshes).
    inline std::chrono::milliseconds wcwCoherentWaitBound() { return std::chrono::milliseconds(34); }

    inline constexpr uint32_t swappedOffset(uint32_t i) {
        return i ^ 3;
    }

    float sqr(float x);
    bool matrixIsNaN(const hlslpp::float4x4 &m);
    hlslpp::float4x4 matrixScale(float scale);
    hlslpp::float4x4 matrixScale(const hlslpp::float3& scale);
    void matrixDecomposeViewProj(const hlslpp::float4x4 &vp, hlslpp::float4x4 &v, hlslpp::float4x4 &p);
    hlslpp::float4x4 matrixTranslation(const hlslpp::float3 &t);
    hlslpp::float3x3 matrixRotationX(float rad);
    hlslpp::float3x3 matrixRotationY(float rad);
    hlslpp::float3x3 matrixRotationZ(float rad);
    hlslpp::float3x3 extract3x3(const hlslpp::float4x4 &m);
    hlslpp::float3x3 rotationFrom3x3(const hlslpp::float3x3 &m);
    float traceFrom3x3(const hlslpp::float3x3 &m);
    float matrixDifference(const hlslpp::float4x4 &a, const hlslpp::float4x4 &b);
    bool isMatrixAffine(const hlslpp::float4x4 &m);
    bool isMatrixIdentity(const hlslpp::float4x4 &m);
    bool isMatrixViewProj(const hlslpp::float4x4 &m);
    hlslpp::float4x4 lerpMatrix(const hlslpp::float4x4 &a, const hlslpp::float4x4 &b, float t);
    hlslpp::float4x4 lerpMatrix3x3(const hlslpp::float4x4 &a, const hlslpp::float4x4 &b, float t);
    hlslpp::float4x4 lerpMatrixComponents(const hlslpp::float4x4 &a, const hlslpp::float4x4 &b, bool linear, bool angular, bool perspective, float t);
    float nearPlaneFromProj(const hlslpp::float4x4 &m);
    float farPlaneFromProj(const hlslpp::float4x4 &m);
    float fovFromProj(const hlslpp::float4x4 &m);
    float pseudoRandom(uint32_t &s);
    hlslpp::float2 barycentricCoordinates(const hlslpp::float2 p, const hlslpp::float2 a, const hlslpp::float2 b, const hlslpp::float2 c);
    bool decomposeMatrix(const hlslpp::float4x4& mtx, hlslpp::quaternion& rotation, hlslpp::float3& scale, hlslpp::float3& skew,
        hlslpp::float3& translation, hlslpp::float4& perspective, bool &coordinateFlip);
    hlslpp::float4x4 recomposeMatrix(const hlslpp::quaternion& rotation, const hlslpp::float3& scale, const hlslpp::float3& skew,
        const hlslpp::float3& translation, const hlslpp::float4& perspective);
    
    struct DecomposedTransform {
        hlslpp::quaternion rotation;
        hlslpp::float3 scale;
        hlslpp::float3 skew;
        hlslpp::float3 translation;
        hlslpp::float4 perspective;
        bool coordinateFlip = false;
        bool valid = false;

        DecomposedTransform() = default;
        DecomposedTransform(const hlslpp::float4x4& mtx);
    };

    DecomposedTransform lerpTransforms(const DecomposedTransform& a, const DecomposedTransform& b, float weight,
        bool lerpTranslation, bool lerpRotation, bool lerpScale, bool lerpSkew, bool lerpPerpsective, bool useSlerp);
};