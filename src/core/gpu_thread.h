// SPDX-FileCopyrightText: 2019-2024 Connor McLaughlin <stenzek@gmail.com>
// SPDX-License-Identifier: (GPL-3.0 OR CC-BY-NC-ND-4.0)

#pragma once

#include "gpu_types.h"

#include "common/threading.h"

#include <optional>

class Error;

enum class RenderAPI : u32;
enum class GPUVSyncMode : u8;

namespace GPUThread {
using AsyncCallType = std::function<void()>;

const Threading::ThreadHandle& GetThreadHandle();
RenderAPI GetRenderAPI();
bool IsStarted();
bool WasFullscreenUIRequested();

/// Starts Big Picture UI.
bool StartFullscreenUI(Error* error);

/// Backend control.
std::optional<GPURenderer> GetRequestedRenderer();
bool CreateGPUBackend(GPURenderer renderer, Error* error);
bool SwitchGPUBackend(GPURenderer renderer, bool force_recreate_device, Error* error);
void DestroyGPUBackend();

/// Fully stops the thread, closing in the process if needed.
void Shutdown();

/// Re-presents the current frame. Call when things like window resizes happen to re-display
/// the current frame with the correct proportions. Should only be called from the CPU thread.
void PresentCurrentFrame();

/// Handles fullscreen transitions and such.
void UpdateDisplayWindow();

/// Called when the window is resized.
void ResizeDisplayWindow(s32 width, s32 height, float scale);

void UpdateSettings();

void RunOnThread(AsyncCallType func);
void SetVSync(GPUVSyncMode mode, bool allow_present_throttle);
void SetRunIdle(bool enabled);

float GetGPUUsage();
float GetGPUAverageTime();
void SetPerformanceCounterUpdatePending();

GPUThreadCommand* AllocateCommand(GPUBackendCommandType command, u32 size);
void PushCommand(GPUThreadCommand* cmd);
void PushCommandAndWakeThread(GPUThreadCommand* cmd);
void PushCommandAndSync(GPUThreadCommand* cmd, bool spin);

// NOTE: Only called by GPUBackend
namespace Internal {
void PresentFrame(bool allow_skip_present, Common::Timer::Value present_time);
}
} // namespace GPUThread
