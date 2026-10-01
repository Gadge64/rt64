//
// RT64
//

#include "rt64_present_queue.h"

#include "common/rt64_thread.h"
#include "rhi/rt64_render_hooks.h"

#include "rt64_workload_queue.h"

namespace RT64 {
    // PresentQueue

    PresentQueue::PresentQueue() {
        reset();
    }

    PresentQueue::~PresentQueue() {
        presentThreadRunning = false;
        cursorCondition.notify_all();

        if (presentThread != nullptr) {
            presentThread->join();
            delete presentThread;
        }

        presentIdCondition.notify_all();
    }

    void PresentQueue::reset() {
        threadCursor = 0;
        writeCursor = 0;
        barrierCursor = 0;
        presentId = 0;
    }

    void PresentQueue::advanceToNextPresent() {
        int nextWriteCursor = (writeCursor + 1) % presents.size();

        // Stall the thread until the barrier is lifted if we're trying to write on a present being used by the GPU.
        bool waitForBarrier;
        do {
            const std::scoped_lock lock(cursorMutex);
            waitForBarrier = (nextWriteCursor == barrierCursor);
        } while (waitForBarrier);

        // Modify the cursor and notify anything waiting on the queue.
        {
            const std::scoped_lock lock(cursorMutex);
            writeCursor = nextWriteCursor;
        }

        cursorCondition.notify_all();
    }

    void PresentQueue::repeatLastPresent() {
        {
            const std::scoped_lock lock(cursorMutex);
            threadCursor = previousWriteCursor();
        }

        cursorCondition.notify_all();
    }

    uint32_t PresentQueue::previousWriteCursor() const {
        if (writeCursor > 0) {
            return writeCursor - 1;
        }
        else {
            return uint32_t(presents.size()) - 1;
        }
    }

    void PresentQueue::waitForIdle() {
        std::unique_lock<std::mutex> threadLock(threadMutex);
    }

    void PresentQueue::waitForPresentId(uint64_t waitId) {
        std::unique_lock<std::mutex> presentLock(presentIdMutex);
        presentIdCondition.wait(presentLock, [&]() {
            return (waitId <= presentId) || !presentThreadRunning;
        });
    }

    void PresentQueue::setup(const External &ext) {
        this->ext = ext;

        viRenderer = std::make_unique<VIRenderer>();

        presentThreadRunning = true;
        presentThread = new std::thread(&PresentQueue::threadLoop, this);
    }

    void PresentQueue::threadPresent(const Present &present, bool &swapChainValid) {
        FramebufferManager &fbManager = ext.sharedResources->framebufferManager;
        RenderTargetManager &targetManager = ext.sharedResources->renderTargetManager;
        const bool usingMSAA = (targetManager.multisampling.sampleCount > 1);
        hlslpp::float2 resolutionScale;
        EnhancementConfiguration::Presentation::Mode presentationMode;
        bool removeBlackBorders;
        UserConfiguration::RefreshRate refreshRate;
        UserConfiguration::Filtering filtering;
        uint32_t viOriginalRate;
        uint32_t targetRate;
        {
            std::scoped_lock<std::mutex> configurationLock(ext.sharedResources->configurationMutex);
            resolutionScale = ext.sharedResources->resolutionScale;
            presentationMode = ext.sharedResources->enhancementConfig.presentation.mode;
            removeBlackBorders = ext.sharedResources->enhancementConfig.presentation.removeBlackBorders;
            refreshRate = ext.sharedResources->userConfig.refreshRate;
            filtering = ext.sharedResources->userConfig.filtering;
            viOriginalRate = ext.sharedResources->viOriginalRate;
            targetRate = ext.sharedResources->targetRate;
        }

        RenderTarget *colorTarget = nullptr;
        int32_t framesToPresent = 1;
        int wcw_reason = 0; // [wcw] DIAGNOSTIC present outcome: 0=vi-not-visible 1=ok 2=target-empty 3=scratch
        uint32_t wcw_addr = 0; // [wcw] DIAGNOSTIC: address of the fb actually presented
        uint64_t wcwPairBlendSerial = 0; // [wcw fix] run serial of the blend shown at i=0
        bool lockedWorkloadMutex = false;
        const uint32_t wcwPresentSetIndex = ext.sharedResources->interpolatedFramesIndex;
        InterpolatedFrameCounters &frameCounters = ext.sharedResources->interpolatedFrames[wcwPresentSetIndex];

        // TODO: There's a possible race condition interactions that can happen while the workload
        // queue is rendering extra frames and the present event is processed while it's generating
        // interpolated frames. When the framebuffer manager or the render target manager maps are
        // modified while the present queue is retrieving the framebuffer or the target. These can
        // likely be solved by locking the access to the managers during modification.
        
        // Perform any external write operations indicated by the event.
        if (!present.fbOperations.empty()) {
            // [wcw] DIAGNOSTIC: fbOperations upload RDRAM contents ONTO render targets before
            // presenting. WCW's fb RDRAM is never written back by RT64, so a WriteChanges op
            // aimed at a presented fb would splat stale/black RAM over the rendered frame.
            { static int fo = 0;
              static const bool wcwTrace = getenv("WCW_TRACE") != nullptr;
              for (const auto &op : present.fbOperations) {
                  if (wcwTrace && (fo < 40 || (fo % 60) == 0)) fprintf(stderr, "[wcw][fbop#%d] type=%d addr=0x%X\n",
                      fo, (int)op.type, (op.type == FramebufferOperation::Type::WriteChanges) ? op.writeChanges.address : 0);
                  fo++;
              } }
            const std::scoped_lock lock(screenFbChangePoolMutex);
            {
                RenderWorkerExecution workerExecution(ext.presentGraphicsWorker);
                fbManager.performOperations(ext.presentGraphicsWorker, &screenFbChangePool, nullptr, ext.shaderLibrary, nullptr,
                    present.fbOperations, targetManager, resolutionScale, 0, 0, nullptr);
            }
        }

        // Present the VI specified by the event.
        // Attempt to find the matching framebuffer for the VI based on the origin address.
        // If that fails, we look at the shared storage.
        if (present.screenVI.visible()) {
            Framebuffer *viFb = nullptr;
            if (!viewRDRAM) {
                viFb = fbManager.find(present.screenVI.fbAddress());
            }

            Framebuffer *presentFb = viFb;
            
            // Show the framebuffer the debugger has requested instead.
            if (present.debuggerFramebuffer.view) {
                Framebuffer *candidateFb = fbManager.find(present.debuggerFramebuffer.address);
                if (candidateFb != nullptr) {
                    presentFb = candidateFb;
                }
            }
            
            if ((presentFb != nullptr) && (viFb != nullptr)) {
                for (uint32_t colorAddress : ext.sharedResources->colorImageAddressVector) {
                    Framebuffer *colorFb = fbManager.find(colorAddress);
                    if (colorFb == nullptr) {
                        continue;
                    }

                    // Always default to interpolation being disabled for all modified framebuffers.
                    colorFb->interpolationEnabled = false;
                    
                    // When the skip buffering option is on, we check the video history to find if any of the framebuffers that
                    // were drawn in this frame have been previously used for presentation. This is ignored when the debugger
                    // has forced viewing a particular framebuffer.
                    if (!present.debuggerFramebuffer.view && (presentationMode == EnhancementConfiguration::Presentation::Mode::SkipBuffering)) {
                        for (size_t h = 0; h < viHistory.history.size(); h++) {
                            const VIHistory::Present &entry = viHistory.history[h];
                            if ((colorFb->addressStart == entry.vi.fbAddress()) && (colorFb->width == entry.fbWidth) && (colorFb->siz == entry.vi.fbSiz()) && entry.vi.compatibleWith(present.screenVI)) {
                                presentFb = colorFb;
                                break;
                            }
                        }
                    }

                    // [wcw fix] This game double-buffers in a match, so the buffer just drawn is never the
                    // one being presented and the check below never passes. Allow interpolation for a
                    // buffer that VI history shows was presented, without changing what is presented.
                    if (!colorFb->interpolationEnabled && !present.debuggerFramebuffer.view) {
                        for (size_t h = 0; h < viHistory.history.size(); h++) {
                            const VIHistory::Present &entry = viHistory.history[h];
                            if ((colorFb->addressStart == entry.vi.fbAddress()) && (colorFb->width == entry.fbWidth) &&
                                (colorFb->siz == entry.vi.fbSiz()) && entry.vi.compatibleWith(present.screenVI)) {
                                colorFb->interpolationEnabled = true;
                                break;
                            }
                        }
                    }

                    // Present early (or games that behave like it) will make it so that the presented image is a color image
                    // that the workload modified. We run a basic check to see if that holds true to indicate it was presented
                    // so interpolation is possible.
                    if (colorFb == presentFb) {
                        presentFb->interpolationEnabled = true;
                        break;
                    }
                }

                if (presentFb->interpolationEnabled) {
                    framesToPresent = frameCounters.count;
                }
                else {
                    lockedWorkloadMutex = true;
                    ext.sharedResources->workloadMutex.lock();
                }

                // [wcw fix] The base-frame (i=0) present samples the LIVE render target, which
                // threadRenderFrame concurrently re-renders under workloadMutex on another GPU
                // queue. Upstream only took the lock when interpolation was disabled, so the
                // present blit raced the render and showed cleared/partial content — measured
                // as 1 of every 3 presents being pure black at 20 fps (visible black flicker).
                // Take the lock in the interpolation path too; the loop below releases it right
                // after the i=0 present. Skip only MSAA multi-frame presents (their i=0 uses an
                // interpolated COPY and waits on the workload queue, which would deadlock here).
                if (!lockedWorkloadMutex && !(usingMSAA && (framesToPresent > 1))) {
                    lockedWorkloadMutex = true;
                    ext.sharedResources->workloadMutex.lock();
                }

                // [wcw fix] Note which run's blend this buffer holds, so the real frame presented next
                // can be taken from the same run.
                {
                    const uint32_t pairSl = (presentFb->addressStart >> 10) & 3;
                    wcwPairBlendSerial = (RT64::wcwAddrSlotAddr[pairSl].load(std::memory_order_relaxed) == presentFb->addressStart) ? RT64::wcwAddrSlotSerial[pairSl].load(std::memory_order_relaxed) : 0;
                }

                RenderTargetKey colorTargetKey(presentFb->addressStart, presentFb->width, presentFb->siz, Framebuffer::Type::Color);
                colorTarget = &targetManager.get(colorTargetKey, true);

                // [wcw fix] In multi-layer match frames, never show a game framebuffer whose run has not
                // finished rendering as the blend: repeat the newest complete real frame instead.
                // (Single-layer screens are excluded: their completion stamp lags while they are fine.)
                bool wcwUsingRealFrame = false;
                if ((framesToPresent > 1) && !usingMSAA && RT64::wcwInterpActive.load(std::memory_order_relaxed) && (wcwPairBlendSerial > 0) &&
                    (RT64::wcwLastRunLength.load(std::memory_order_relaxed) >= 4)) {
                    const uint32_t sl = (presentFb->addressStart >> 10) & 3;
                    const bool complete = (RT64::wcwAddrSlotComplete[sl].load(std::memory_order_relaxed) == wcwPairBlendSerial);
                    if (!complete) {
                        uint64_t bestReal = 0;
                        int bestRealK = -1;
                        for (int k = 0; k < 3; k++) {
                            const uint64_t rs = RT64::wcwSlotRealSerial[k].load(std::memory_order_relaxed);
                            if ((rs > bestReal) && (RT64::wcwSlotRealComplete[k].load(std::memory_order_relaxed) == rs)) {
                                bestReal = rs;
                                bestRealK = k;
                            }
                        }

                        auto &realTargets = (bestRealK == 0) ? ext.sharedResources->interpolatedColorTargets
                            : ((bestRealK == 1) ? ext.sharedResources->wcwInterpolatedColorTargetsB : ext.sharedResources->wcwInterpolatedColorTargetsC);
                        if ((bestRealK >= 0) && !realTargets.empty() && (realTargets[0] != nullptr)) {
                            colorTarget = realTargets[0].get();
                            wcwUsingRealFrame = true;
                        }
                    }
                }

                // [wcw] DIAGNOSTIC: report whether the present found a non-empty rendered target.
                { static int pn = 0; static const bool wcwTrace = getenv("WCW_TRACE") != nullptr; if (wcwTrace && (pn++ % 61) == 0) fprintf(stderr, "[wcw][blit#%d] addr=0x%X w=%d siz=%d empty=%d interp=%d\n",
                    pn, presentFb->addressStart, (int)presentFb->width, (int)presentFb->siz, (int)colorTarget->isEmpty(), (int)presentFb->interpolationEnabled); }
                if (!colorTarget->isEmpty()) {
                    wcw_reason = 1;
                    wcw_addr = presentFb->addressStart;
                    // If a depth framebuffer is about to be shown, convert it to color.
                    if (!wcwUsingRealFrame && presentFb->isLastWriteDifferent(Framebuffer::Type::Color)) {
                        RenderTargetKey otherColorTargetKey(presentFb->addressStart, presentFb->width, presentFb->siz, presentFb->lastWriteType);
                        RenderTarget &otherColorTarget = targetManager.get(otherColorTargetKey, true);
                        if (!otherColorTarget.isEmpty()) {
                            const FixedRect &r = presentFb->lastWriteRect;
                            RenderWorkerExecution workerExecution(ext.presentGraphicsWorker);
                            colorTarget->copyFromTarget(ext.presentGraphicsWorker, &otherColorTarget, r.left(false), r.top(false), r.width(false, true), r.height(false, true), ext.shaderLibrary);
                        }
                    }
                }
                else {
                    colorTarget = nullptr;
                    wcw_reason = 2;
                }

                if (!present.paused && (viHistory.top().vi != present.screenVI)) {
                    viHistory.pushVI(present.screenVI, viFb->width);
                }
            }
            else {
                uint32_t fbAddress = present.screenVI.fbAddress();

                // [wcw] DIAGNOSTIC: count scratch-path presents. This path RESIZES the shared
                // render target down to native size (destroying the rendered hi-res texture),
                // clears it, and uploads the RDRAM copy (black — RT64 never writes back), which
                // would wipe the rendered frame and explain the black screen.
                { static int sp = 0; static const bool wcwTrace = getenv("WCW_TRACE") != nullptr; if (wcwTrace && (sp++ % 30) == 0) fprintf(stderr, "[wcw][scratch-present#%d] fbAddr=0x%X (fb lookup MISSED)\n", sp, fbAddress); }
                wcw_reason = 3;

                // Use a scratch framebuffer to upload the RAM to the render target.
                hlslpp::uint2 fbSize = present.screenVI.fbSize();
                scratchFb.addressStart = fbAddress;
                scratchFb.width = fbSize.x;
                scratchFb.height = fbSize.y;
                scratchFb.siz = present.screenVI.fbSiz();

                lockedWorkloadMutex = true;
                ext.sharedResources->workloadMutex.lock();

                RenderTargetKey colorTargetKey(fbAddress, scratchFb.width, scratchFb.siz, Framebuffer::Type::Color);
                colorTarget = &targetManager.get(colorTargetKey, true);
                colorTarget->resize(ext.presentGraphicsWorker, scratchFb.width, scratchFb.height);
                colorTarget->resolutionScale = { 1.0f, 1.0f };
                colorTarget->downsampleMultiplier = 1;

                scratchFb.nativeTarget.resetBufferHistory();

                {
                    RenderWorkerExecution workerExecution(ext.presentGraphicsWorker);
                    colorTarget->clearColorTarget(ext.presentGraphicsWorker);
                    FramebufferChange *colorFbChange = scratchFb.readChangeFromBytes(ext.presentGraphicsWorker, scratchFbChangePool, Framebuffer::Type::Color,
                        G_IM_FMT_RGBA, present.storage.data(), 0, scratchFb.height, ext.shaderLibrary);

                    if (colorFbChange != nullptr) {
                        colorTarget->copyFromChanges(ext.presentGraphicsWorker, *colorFbChange, scratchFb.width, scratchFb.height, 0, ext.shaderLibrary);
                    }
                }

                scratchFbChangePool.reset();

                if (!present.paused && (viHistory.top().vi != present.screenVI)) {
                    viHistory.pushVI(present.screenVI, fbSize.x);
                }
            }
        }

        // Create the framebuffers if necessary.
        if (swapChainFramebuffers.empty()) {
            uint32_t textureCount = ext.swapChain->getTextureCount();
            swapChainFramebuffers.resize(textureCount);
            for (uint32_t i = 0; i < textureCount; i++) {
                const RenderTexture *swapChainTexture = ext.swapChain->getTexture(i);
                swapChainFramebuffers[i] = ext.device->createFramebuffer(RenderFramebufferDesc(&swapChainTexture, 1));
            }
        }
        
        // [wcw] DIAGNOSTIC (env WCW_PRESENT_LOG=1): if framesToPresent is 0, nothing is ever
        // blitted to the swap chain.
        { static const bool wcwPlog = getenv("WCW_PRESENT_LOG") != nullptr;
          static int fp = 0; if (((fp++ % 60) == 0) && wcwPlog) fprintf(stderr, "[wcw][present-loop#%d] framesToPresent=%d counters.count=%u avail=%u target=%p swapValid=%d\n",
            fp, framesToPresent, frameCounters.count, frameCounters.available, (void*)colorTarget, (int)swapChainValid); }

        for (int32_t i = 0; i < framesToPresent; i++) {
            uint32_t frameCountersNextPresented = 0;
            if ((framesToPresent > 1) && (usingMSAA || (i > 0))) {
                // Stall until the interpolated color target is available.
                const uint32_t targetIndex = usingMSAA ? i : (i - 1);
                std::unique_lock<std::mutex> interpolatedLock(ext.sharedResources->interpolatedMutex);
                // [wcw fix] Bounded wait: in a multi-layer frame only the last layer makes the target
                // available, and an unbounded wait here can stall VI and with it the game logic that
                // would complete the frame. On timeout this frame is dropped like a skipped one.
                const bool wcwInterpTimedOut = !ext.sharedResources->interpolatedCondition.wait_for(interpolatedLock, RT64::wcwCoherentWaitBound(), [&]() {
                    return (frameCounters.available > targetIndex) || ((frameCounters.available == targetIndex) && frameCounters.skipped);
                });

                // Do not present any more frames after this one after reaching the last available frame if the workload was skipped.
                if (wcwInterpTimedOut || ((frameCounters.available == targetIndex) && frameCounters.skipped)) {
                    framesToPresent = std::min(int(frameCounters.available), i + 1);
                    frameCountersNextPresented = frameCounters.count;
                }
                else {
                    frameCountersNextPresented = frameCounters.presented + 1;
                }

                if (i < framesToPresent) {
                    uint32_t targetIndex = usingMSAA ? i : (i - 1);
                    // [wcw fix] Counter set 1 uses the second set of interpolated targets.
                    auto &wcwTargets = ((wcwPresentSetIndex & 1) != 0) ? ext.sharedResources->wcwInterpolatedColorTargetsB : ext.sharedResources->interpolatedColorTargets;
                    colorTarget = (targetIndex < wcwTargets.size()) ? wcwTargets[targetIndex].get() : nullptr;
                    // [wcw fix] Show the real frame of the same run as the blend just shown, from its
                    // rotating slot (serial % 3), once that run has finished.
                    if (wcwPairBlendSerial > 0) {
                        const uint32_t k = uint32_t(wcwPairBlendSerial % 3);
                        if ((RT64::wcwSlotRealSerial[k].load(std::memory_order_relaxed) == wcwPairBlendSerial) &&
                            (RT64::wcwSlotRealComplete[k].load(std::memory_order_relaxed) == wcwPairBlendSerial)) {
                            auto &slotTargets = (k == 0) ? ext.sharedResources->interpolatedColorTargets
                                : ((k == 1) ? ext.sharedResources->wcwInterpolatedColorTargetsB : ext.sharedResources->wcwInterpolatedColorTargetsC);
                            if (targetIndex < slotTargets.size()) {
                                colorTarget = slotTargets[targetIndex].get();
                            }
                        }
                    }
                }
                else {
                    colorTarget = nullptr;
                }
            }
            else if (framesToPresent == 1) {
                frameCountersNextPresented = frameCounters.count;
            }

            uint32_t swapChainIndex = 0;
            const bool presentFrame = (i < framesToPresent) && swapChainValid;
            double wcwAcqMs = 0.0;
            if (presentFrame) {
                auto wcwAcq0 = std::chrono::steady_clock::now();
                swapChainValid = ext.swapChain->acquireTexture(acquiredSemaphore.get(), &swapChainIndex);
                wcwAcqMs = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - wcwAcq0).count() / 1000.0;
            }

            if (presentFrame && swapChainValid) {
                // Draw the framebuffer with the VI renderer.
                RenderTexture *swapChainTexture = ext.swapChain->getTexture(swapChainIndex);
                RenderFramebuffer *swapChainFramebuffer = swapChainFramebuffers[swapChainIndex].get();
                RenderCommandList *commandList = ext.presentGraphicsWorker->commandList.get();
                commandList->begin();
                commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(swapChainTexture, RenderTextureLayout::COLOR_WRITE));
                
                VIRenderer::RenderParams renderParams;
                if (colorTarget != nullptr) {
                    renderParams.device = ext.device;
                    renderParams.commandList = commandList;
                    renderParams.swapChain = ext.swapChain;
                    renderParams.shaderLibrary = ext.shaderLibrary;
                    renderParams.textureFormat = colorTarget->format;
                    renderParams.resolutionScale = colorTarget->resolutionScale;
                    renderParams.downsamplingScale = 1;
                    renderParams.filtering = filtering;
                    renderParams.vi = &present.screenVI;
                    renderParams.removeBlackBorders = removeBlackBorders;

                    const bool useDownsampling = (colorTarget->downsampleMultiplier > 1);
                    if (useDownsampling) {
                        colorTarget->downsampleTarget(ext.presentGraphicsWorker, ext.shaderLibrary);
                        renderParams.texture = colorTarget->downsampledTexture.get();
                        renderParams.textureWidth = colorTarget->width / colorTarget->downsampleMultiplier;
                        renderParams.textureHeight = colorTarget->height / colorTarget->downsampleMultiplier;
                        renderParams.downsamplingScale = colorTarget->downsampleMultiplier;
                    }
                    else {
                        colorTarget->resolveTarget(ext.presentGraphicsWorker, ext.shaderLibrary);
                        renderParams.texture = colorTarget->getResolvedTexture();
                        renderParams.textureWidth = colorTarget->width;
                        renderParams.textureHeight = colorTarget->height;
                    }
                }
                
                { // [wcw] DIAGNOSTIC (env WCW_PRESENT_LOG=1): per-60-presents outcome accounting.
                  // A null source texture means this present draws ONLY the clear -> a pure black
                  // frame on screen.
                    static const bool wcwPresentLog = getenv("WCW_PRESENT_LOG") != nullptr;
                    static int prCnt = 0, prBlack = 0, prReason[4] = {};
                    static double acqSum = 0.0, acqMax = 0.0;
                    if (renderParams.texture == nullptr) prBlack++;
                    prReason[wcw_reason]++;
                    acqSum += wcwAcqMs; if (wcwAcqMs > acqMax) acqMax = wcwAcqMs;
                    if (++prCnt >= 60) {
                        if (wcwPresentLog) {
                            fprintf(stderr, "[wcw][present] last %d presents: black=%d | notvis=%d ok=%d empty=%d scratch=%d | acquire avg=%.1fms max=%.1fms\n",
                                prCnt, prBlack, prReason[0], prReason[1], prReason[2], prReason[3], acqSum / prCnt, acqMax);
                        }
                        prCnt = prBlack = 0; prReason[0] = prReason[1] = prReason[2] = prReason[3] = 0;
                        acqSum = 0.0; acqMax = 0.0;
                    }
                }
                commandList->setFramebuffer(swapChainFramebuffer);
                commandList->clearColor();

                if (renderParams.texture != nullptr) {
                    commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(renderParams.texture, RenderTextureLayout::SHADER_READ));
                    viRenderer->render(renderParams);
                }

                RenderHookDraw *drawHook = GetRenderHookDraw();
                if (drawHook != nullptr) {
                    drawHook(commandList, swapChainFramebuffer);
                }

                {
                    const std::scoped_lock lock(inspectorMutex);
                    if (inspector != nullptr) {
                        inspector->draw(commandList);
                    }

                    // [wcw] DIAGNOSTIC (env WCW_PRESENT_LUM=1): copy the finished swapchain image
                    // into a readback buffer so the ACTUAL presented pixels can be verified CPU-side
                    // (ground truth below the OS — screen captures are unreliable with MPO).
                    static const bool wcwLum = getenv("WCW_PRESENT_LUM") != nullptr;
                    // BMP dump window configurable: WCW_BMP_START (default 600), WCW_BMP_COUNT
                    // (default 13), WCW_LUM_END (default 1500) — lets captures target any moment
                    // (e.g. in-game menus) instead of only early boot.
                    static const int wcwBmpStart = getenv("WCW_BMP_START") ? atoi(getenv("WCW_BMP_START")) : 600;
                    static const int wcwBmpCount = getenv("WCW_BMP_COUNT") ? atoi(getenv("WCW_BMP_COUNT")) : 13;
                    static const int wcwLumEnd = getenv("WCW_LUM_END") ? atoi(getenv("WCW_LUM_END")) : 1500;
                    static std::unique_ptr<RenderBuffer> wcwReadback;
                    static uint32_t wcwRbW = 0, wcwRbH = 0, wcwRbRow = 0;
                    static int wcwPresentN = 0;
                    bool wcwDoRead = false;
                    if (wcwLum) {
                        wcwPresentN++;
                        if (wcwPresentN >= 240 && wcwPresentN <= wcwLumEnd) {
                            uint32_t w = ext.swapChain->getWidth(), h = ext.swapChain->getHeight();
                            uint32_t row = (w + 63) & ~63u;
                            if (!wcwReadback || wcwRbW != w || wcwRbH != h) {
                                wcwReadback = ext.device->createBuffer(RenderBufferDesc::ReadbackBuffer(uint64_t(row) * h * 4));
                                wcwRbW = w; wcwRbH = h; wcwRbRow = row;
                            }
                            commandList->barriers(RenderBarrierStage::COPY, RenderTextureBarrier(swapChainTexture, RenderTextureLayout::COPY_SOURCE));
                            commandList->copyTextureRegion(
                                RenderTextureCopyLocation::PlacedFootprint(wcwReadback.get(), RenderFormat::B8G8R8A8_UNORM, wcwRbW, wcwRbH, 1, wcwRbRow, 0),
                                RenderTextureCopyLocation::Subresource(swapChainTexture));
                            wcwDoRead = true;
                        }
                    }

                    commandList->barriers(RenderBarrierStage::NONE, RenderTextureBarrier(swapChainTexture, RenderTextureLayout::PRESENT));
                    commandList->end();
                    const RenderCommandList *commandList = ext.presentGraphicsWorker->commandList.get();
                    RenderCommandSemaphore *waitSemaphore = acquiredSemaphore.get();
                    RenderCommandSemaphore *signalSemaphore = drawSemaphores[swapChainIndex].get();
                    ext.presentGraphicsWorker->commandQueue->executeCommandLists(&commandList, 1, &waitSemaphore, 1, &signalSemaphore, 1, ext.presentGraphicsWorker->commandFence.get());
                    ext.presentGraphicsWorker->wait();

                    if (wcwDoRead) {
                        const uint8_t *p = (const uint8_t *)(wcwReadback->map(0, nullptr));
                        uint64_t sum = 0; uint32_t cnt = 0;
                        for (uint32_t y = 0; y < wcwRbH; y += 8) {
                            const uint8_t *rowp = p + uint64_t(y) * wcwRbRow * 4;
                            for (uint32_t x = 0; x < wcwRbW; x += 8) {
                                const uint8_t *px = rowp + uint64_t(x) * 4;
                                sum += px[0] + px[1] + px[2];
                                cnt += 3;
                            }
                        }
                        static FILE *lumf = nullptr;
                        if (lumf == nullptr) lumf = fopen("wcw_present_lum.csv", "w");
                        if (lumf != nullptr) {
                            double msNow = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count() / 1000.0;
                            fprintf(lumf, "%d,%.1f,0x%X,0x%X,%.2f\n", wcwPresentN, msNow, present.screenVI.fbAddress(), wcw_addr, double(sum) / cnt);
                            fflush(lumf);
                        }
                        if (wcwPresentN >= wcwBmpStart && wcwPresentN < wcwBmpStart + wcwBmpCount) {
                            char name[64]; snprintf(name, sizeof(name), "wcw_present_%04d.bmp", wcwPresentN);
                            FILE *bf = fopen(name, "wb");
                            if (bf != nullptr) {
                                uint32_t imgSize = wcwRbW * wcwRbH * 4, fileSize = 54 + imgSize, off54 = 54, ihs = 40;
                                int32_t negH = -int32_t(wcwRbH);
                                uint16_t planes = 1, bpp = 32;
                                uint8_t hdr[54] = {};
                                hdr[0] = 'B'; hdr[1] = 'M';
                                memcpy(hdr + 2, &fileSize, 4); memcpy(hdr + 10, &off54, 4); memcpy(hdr + 14, &ihs, 4);
                                memcpy(hdr + 18, &wcwRbW, 4); memcpy(hdr + 22, &negH, 4);
                                memcpy(hdr + 26, &planes, 2); memcpy(hdr + 28, &bpp, 2); memcpy(hdr + 34, &imgSize, 4);
                                fwrite(hdr, 1, 54, bf);
                                for (uint32_t y = 0; y < wcwRbH; y++) fwrite(p + uint64_t(y) * wcwRbRow * 4, 1, uint64_t(wcwRbW) * 4, bf);
                                fclose(bf);
                            }
                        }
                        wcwReadback->unmap();
                    }
                }
            }

            if (lockedWorkloadMutex) {
                ext.sharedResources->workloadMutex.unlock();
                lockedWorkloadMutex = false;
            }
            
            if (frameCountersNextPresented > 0) {
                {
                    std::unique_lock<std::mutex> interpolatedLock(ext.sharedResources->interpolatedMutex);
                    frameCounters.presented = frameCountersNextPresented;
                }

                ext.sharedResources->interpolatedCondition.notify_all();
            }

            // As soon as we're done with the first render target, we notify the workload queue it can proceed.
            if (i == 0) {
                notifyPresentId(present);
            }

            if (presentFrame && swapChainValid) {
                // Wait until the approximate time the next present should be at the current intended rate.
                if ((presentTimestamp != Timestamp()) && (targetRate > 0) && (targetRate > viOriginalRate)) {
                    Timer::preciseSleepUntil(presentTimestamp + std::chrono::nanoseconds(1'000'000'000 / targetRate));
                }

                if (presentWaitEnabled) {
                    ext.swapChain->wait();
                }

                RenderCommandSemaphore *waitSemaphore = drawSemaphores[swapChainIndex].get();
                presentTimestamp = Timer::current();
                swapChainValid = ext.swapChain->present(swapChainIndex, &waitSemaphore, 1);
                presentProfiler.logAndRestart();
            }
        }
    }

    void PresentQueue::skipInterpolation() {
        {
            std::unique_lock<std::mutex> interpolatedLock(ext.sharedResources->interpolatedMutex);
            InterpolatedFrameCounters &frameCounters = ext.sharedResources->interpolatedFrames[ext.sharedResources->interpolatedFramesIndex];
            frameCounters.presented = frameCounters.count;
        }

        ext.sharedResources->interpolatedCondition.notify_all();
    }

    void PresentQueue::notifyPresentId(const Present &present) {
        {
            std::scoped_lock<std::mutex> cursorLock(presentIdMutex);
            presentId = present.presentId;
        }

        presentIdCondition.notify_all();
    }
    
    void PresentQueue::threadAdvanceBarrier() {
        std::scoped_lock<std::mutex> cursorLock(cursorMutex);
        barrierCursor = (barrierCursor + 1) % presents.size();
    }

    void PresentQueue::threadLoop() {
        Thread::setCurrentThreadName("RT64 Present");

        // Create the semaphore the acquire method will use.
        acquiredSemaphore = ext.device->createCommandSemaphore();

        // Create as many semaphores to signal as textures there are.
        while (drawSemaphores.size() < ext.swapChain->getTextureCount()) {
            drawSemaphores.emplace_back(ext.device->createCommandSemaphore());
        }

        // Since the swap chain might not need a resize right away, detect present wait.
        presentWaitEnabled = ext.device->getCapabilities().presentWait;

        int processCursor = -1;
        bool skipPresent = false;
        uint32_t displayTimingRate = UINT32_MAX;
        const bool displayTiming = ext.device->getCapabilities().displayTiming;
        bool swapChainValid = !ext.swapChain->needsResize();
        while (presentThreadRunning) {
            {
                std::unique_lock<std::mutex> cursorLock(cursorMutex);
                cursorCondition.wait(cursorLock, [&]() {
                    return (writeCursor != threadCursor) || !presentThreadRunning;
                });

                if (presentThreadRunning) {
                    processCursor = threadCursor;
                    threadCursor = (threadCursor + 1) % presents.size();
                    skipPresent = (writeCursor != threadCursor);
                    { // [wcw] DIAGNOSTIC (WCW_PRESENT_LUM=1): present-ring processing trace.
                        static const bool wcwQlog = getenv("WCW_PRESENT_LUM") != nullptr;
                        if (wcwQlog) {
                            static FILE *qf = nullptr;
                            if (qf == nullptr) qf = fopen("wcw_pring_log.csv", "w");
                            if (qf != nullptr) {
                                double ms = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count() / 1000.0;
                                fprintf(qf, "%.1f,proc,%d,%d,%d\n", ms, processCursor, writeCursor, (int)skipPresent);
                                fflush(qf);
                            }
                        }
                    }
                }
            }

            if (processCursor >= 0) {
                std::unique_lock<std::mutex> threadLock(threadMutex);
                const bool needsResize = ext.swapChain->needsResize() || !swapChainValid;
                if (needsResize) {
                    ext.presentGraphicsWorker->commandList->begin();
                    ext.presentGraphicsWorker->commandList->end();
                    ext.presentGraphicsWorker->execute();
                    ext.presentGraphicsWorker->wait();
                    swapChainValid = ext.swapChain->resize();
                    swapChainFramebuffers.clear();

                    if (swapChainValid) {
                        ext.sharedResources->setSwapChainSize(ext.swapChain->getWidth(), ext.swapChain->getHeight());
                        
                        // Texture count could've changed after resize, so new semaphores are needed.
                        while (drawSemaphores.size() < ext.swapChain->getTextureCount()) {
                            drawSemaphores.emplace_back(ext.device->createCommandSemaphore());
                        }
                    }
                }

                if (needsResize || ext.appWindow->detectWindowMoved()) {
                    ext.appWindow->detectRefreshRate();
                    ext.sharedResources->setSwapChainRate(std::min(ext.appWindow->getRefreshRate(), displayTimingRate));
                }

                if (displayTiming) {
                    uint32_t newDisplayTimingRate = ext.swapChain->getRefreshRate();
                    if (newDisplayTimingRate == 0) {
                        newDisplayTimingRate = UINT32_MAX;
                    }

                    if (newDisplayTimingRate != displayTimingRate) {
                        ext.sharedResources->setSwapChainRate(std::min(ext.appWindow->getRefreshRate(), newDisplayTimingRate));
                        displayTimingRate = newDisplayTimingRate;
                    }
                }

                skipPresent = skipPresent || ext.swapChain->isEmpty();

                Present &present = presents[processCursor];
                ext.workloadQueue->waitForWorkloadId(present.workloadId);

                if (!presentThreadRunning) {
                    continue;
                }

                if (skipPresent) {
                    skipInterpolation();
                    notifyPresentId(present);
                }
                else {
                    threadPresent(present, swapChainValid);
                }

                if (!present.paused) {
                    if (!present.fbOperations.empty()) {
                        const std::scoped_lock lock(screenFbChangePoolMutex);
                        screenFbChangePool.release(present.fbOperations.front().writeChanges.id);
                        present.fbOperations.clear();
                    }

                    threadAdvanceBarrier();
                }

                processCursor = -1;
            }
        }

        // Transition the active swap chain render target out of the present state to avoid live references to the resource.
        uint32_t swapChainIndex = 0;
        if (!ext.swapChain->isEmpty() && ext.swapChain->acquireTexture(acquiredSemaphore.get(), &swapChainIndex)) {
            RenderTexture *swapChainTexture = ext.swapChain->getTexture(swapChainIndex);
            ext.presentGraphicsWorker->commandList->begin();
            ext.presentGraphicsWorker->commandList->barriers(RenderBarrierStage::NONE, RenderTextureBarrier(swapChainTexture, RenderTextureLayout::COLOR_WRITE));
            ext.presentGraphicsWorker->commandList->end();

            const RenderCommandList *commandList = ext.presentGraphicsWorker->commandList.get();
            RenderCommandSemaphore *waitSemaphore = acquiredSemaphore.get();
            ext.presentGraphicsWorker->commandQueue->executeCommandLists(&commandList, 1, &waitSemaphore, 1, nullptr, 0, ext.presentGraphicsWorker->commandFence.get());
            ext.presentGraphicsWorker->wait();
        }
    }
};
