//
// RT64
//

#pragma once

#include <array>

#include "common/rt64_enhancement_configuration.h"
#include "common/rt64_profiling_timer.h"
#include "common/rt64_user_configuration.h"
#include "render/rt64_framebuffer_renderer.h"
#include "render/rt64_projection_processor.h"
#include "render/rt64_raster_shader_cache.h"
#include "render/rt64_tile_processor.h"
#include "render/rt64_look_at_processor.h"
#include "render/rt64_transform_processor.h"

#include "rt64_shared_queue_resources.h"
#include "rt64_workload.h"

#if RT_ENABLED
#   include "render/rt64_raytracing_shader_cache.h"
#endif

// [wcw fix] Grown from 4 (keep in step with PRESENT_QUEUE_SIZE).
#define WORKLOAD_QUEUE_SIZE 8
// [wcw fix] Workload slots after the ring, never written by the producer: snapshots of recent layers so the
// next frame can still pair with them after their ring slot is recycled.
#define WCW_SNAPSHOT_SLOTS 16

namespace RT64 {
    struct PresentQueue;

    struct WorkloadQueue {
        struct External {
            RenderDevice *device = nullptr;
            RenderWorker *workloadGraphicsWorker = nullptr;
            BufferUploader *workloadExtrasUploader = nullptr;
            BufferUploader *workloadVelocityUploader = nullptr;
            BufferUploader *workloadTilesUploader = nullptr;
            PresentQueue *presentQueue = nullptr;
            SharedQueueResources *sharedResources = nullptr;
            RasterShaderCache *rasterShaderCache = nullptr;
            TextureCache *textureCache = nullptr;
            const ShaderLibrary *shaderLibrary = nullptr;
            UserConfiguration::GraphicsAPI createdGraphicsAPI = UserConfiguration::GraphicsAPI::OptionCount;
#       if RT_ENABLED
            const RenderTexture *blueNoiseTexture = nullptr;
            RaytracingShaderCache *rtShaderCache = nullptr;
#       endif
        };

        struct WorkloadConfiguration {
            hlslpp::float2 resolutionScale = 1.0f;
            uint32_t downsampleMultiplier = 1;
            bool raytracingEnabled = false;
            float aspectRatioSource = 1.0f;
            float aspectRatioTarget = 1.0f;
            float aspectRatioScale = 1.0f;
            float extAspectPercentage = 1.0f;
            uint32_t targetRate = 0;
            bool postBlendNoise = false;
            bool postBlendNoiseNegative = false;
        };

        External ext;
        std::array<Workload, WORKLOAD_QUEUE_SIZE + WCW_SNAPSHOT_SLOTS> workloads;
        // [wcw fix] Latest wrestler-layer snapshot: its GameFrame, workload slot and present.
        uint32_t wcwSnapFrameIndex = UINT32_MAX;
        uint32_t wcwSnapWorkloadIndex = UINT32_MAX;
        uint64_t wcwSnapPresentId = 0;
        // [wcw fix] One entry per snapshot slot: the GameFrame that points at it, the run it came from, its
        // position in that run and its transform count.
        struct WcwLayerSnap {
            uint32_t frameIndex = UINT32_MAX;
            uint64_t runSerial = 0;
            uint32_t runPos = 0;
            uint32_t xf = 0;
        };
        WcwLayerSnap wcwLayerSnaps[WCW_SNAPSHOT_SLOTS];
        uint32_t wcwSnapCounter = 0;
        int threadCursor;
        int writeCursor;
        int barrierCursor;
        std::mutex cursorMutex;
        std::condition_variable cursorCondition;
        uint64_t workloadId;
        uint64_t lastPresentId;
        std::mutex workloadIdMutex;
        std::condition_variable workloadIdCondition;
        std::thread *renderThread = nullptr;
        std::thread *idleThread = nullptr;
        bool idleActive = false;
        std::mutex idleMutex;
        std::condition_variable idleCondition;
        std::mutex workerMutex;
        std::mutex threadMutex;
        std::atomic<bool> threadsRunning = false;
        std::atomic<bool> rtEnabled = false;
        std::atomic<bool> ubershadersOnly = false;
        std::atomic<bool> ubershadersVisible = false;
        std::unique_ptr<FramebufferRenderer> framebufferRenderer;
        std::unique_ptr<RenderFramebufferManager> renderFramebufferManager;
        TileProcessor tileProcessor;
        LookAtProcessor lookAtProcessor;
        TransformProcessor transformProcessor;
        ProjectionProcessor projectionProcessor;
        std::unique_ptr<RSPProcessor> rspProcessor;
        std::unique_ptr<VertexProcessor> vertexProcessor;
        std::unique_ptr<RenderTarget> dummyDepthTarget;
        std::unique_ptr<RenderQueryPool> queryPool;
        FramebufferChangePool scratchFbChangePool;
        ProfilingTimer rendererCPUProfiler = ProfilingTimer(120);
        ProfilingTimer rendererGPUProfiler = ProfilingTimer(120);
        ProfilingTimer matchingProfiler = ProfilingTimer(120);
        ProfilingTimer workloadProfiler = ProfilingTimer(120);
        // [wcw fix] Grown from 2: a layer is paired with the same layer one displayed frame (several
        // GameFrames) back.
        std::array<GameFrame, 16> gameFrames;
        uint32_t prevFrameIndex = uint32_t(gameFrames.size()) - 1;
        uint32_t curFrameIndex = 0;

        // [wcw fix] Multi-layer frame detection: AKI games compose one displayed frame from several consecutive
        // workloads targeting the same colour address. Remembers, per address, how long its last run was.
        uint32_t lastGroupColorAddress = 0;
        bool hasLastGroupColorAddress = false;
        uint32_t currentGroupRunLength = 0;
        struct RecentColorGroup { uint32_t address = 0; uint32_t runLength = 0; };
        std::array<RecentColorGroup, 8> recentColorGroups{};
        size_t recentColorGroupsNext = 0;

        WorkloadQueue();
        ~WorkloadQueue();
        void reset();
        void advanceToNextWorkload();
        void repeatLastWorkload();
        uint32_t previousWriteCursor() const;
        void waitForIdle();
        void waitForWorkloadId(uint64_t waitId);
        void setup(const External &ext);
        void updateMultisampling();
        void threadConfigurationUpdate(hlslpp::uint2 viFbSize, WorkloadConfiguration &workloadConfig);
        void threadConfigurationValidate();
        void threadRenderFrame(GameFrame &curFrame, const GameFrame &prevFrame, const WorkloadConfiguration &workloadConfig,
            const DebuggerRenderer &debuggerRenderer, const DebuggerCamera &debuggerCamera, float curFrameWeight, float prevFrameWeight,
            float deltaTimeMs, RenderTargetKey overrideTargetKey, int32_t overrideTargetFbPairIndex, RenderTarget *overrideTarget,
            uint32_t overrideTargetModifier, bool uploadVelocity, bool uploadExtras, bool interpolateTiles, bool interpolateLookAts);

        void threadAdvanceBarrier();
        void threadAdvanceWorkloadId(uint64_t newWorkloadId);
        void renderThreadLoop();
        void idleThreadLoop();
    };
};