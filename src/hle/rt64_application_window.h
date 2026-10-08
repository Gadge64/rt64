//
// RT64
//

#pragma once

#include "common/rt64_plume.h"
#ifdef _WIN32
#include <Windows.h>
#elif defined(__APPLE__)
#include "apple/rt64_apple.h"
#include "plume_apple.h"
#endif

#include "SDL.h"
#include "SDL_events.h"
#include "SDL_system.h"
#include "SDL_syswm.h"
#include "SDL_video.h"

// Remove macros defined by x11 from a header included by SDL_syswm.h.
#undef None
#undef Status
#undef LockMask
#undef ControlMask
#undef Success
#undef Always

namespace RT64 {
    struct ApplicationWindow {
        static ApplicationWindow *HookedApplicationWindow;

        struct Listener {
            virtual bool usesWindowMessageFilter() = 0;

            // Return true if the listener should accept and handle the message.
            // Return false otherwise for the default message handler to take over.
            virtual bool sdlEventFilter(SDL_Event *event) = 0;

#       ifdef _WIN32
            virtual bool windowMessageFilter(unsigned int message, WPARAM wParam, LPARAM lParam) = 0;
#       endif
        };

        RenderWindow windowHandle = {};
#if defined(__APPLE__)
        std::unique_ptr<CocoaWindow> windowWrapper;
#endif
        Listener *listener;
        uint32_t refreshRate = 0;
        bool fullScreen = false;
        bool lastMaximizedState = false;
        int32_t windowLeft = INT32_MAX;
        int32_t windowTop = INT32_MAX;
        SDL_Window *sdlWindow = nullptr;
        SDL_EventFilter sdlEventFilterStored = nullptr;
        void *sdlEventFilterUserdata = nullptr;
        bool sdlEventFilterInstalled = false;

        // [wcw] Requested output resolution (0x0 = desktop resolution, the stock behaviour).
        // Fullscreen switches the monitor to this mode (restored on leaving fullscreen or exit);
        // windowed sizes the window's client area to it.
        uint32_t outputWidth = 0;
        uint32_t outputHeight = 0;
        uint32_t outputRefreshRate = 0; // Hz, fullscreen only. 0 = keep the desktop's.

#   ifdef _WIN32
        HHOOK windowHook = nullptr;
        HMENU windowMenu = nullptr;
        RECT lastWindowRect = {};
        bool displayModeChanged = false;
        WCHAR displayModeDevice[CCHDEVICENAME] = {};
#   endif

        ApplicationWindow();
        ~ApplicationWindow();
        void setup(RenderWindow window, Listener *listener, uint32_t threadId);
        void setup(const char *windowTitle, Listener *listener);
        void setFullScreen(bool newFullScreen);
        void setOutputResolution(uint32_t width, uint32_t height, uint32_t refreshRate = 0);
        void makeResizable();
        void detectRefreshRate();
        uint32_t getRefreshRate() const;
        bool detectWindowMoved();
        void sdlCheckFilterInstallation();
        static int sdlEventFilter(void *userdata, SDL_Event *event);

#   ifdef _WIN32
        bool changeDisplayMode(HMONITOR monitor);
        void restoreDisplayMode();
        void applyWindowedOutputSize();
        void windowMessage(UINT message, WPARAM wParam, LPARAM lParam);
        static LRESULT windowHookCallback(int nCode, WPARAM wParam, LPARAM lParam);
#   endif
    };
};
