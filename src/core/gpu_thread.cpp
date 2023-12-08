// SPDX-FileCopyrightText: 2019-2024 Connor McLaughlin <stenzek@gmail.com>
// SPDX-License-Identifier: (GPL-3.0 OR CC-BY-NC-ND-4.0)

#include "gpu_thread.h"
#include "fullscreen_ui.h"
#include "gpu_backend.h"
#include "host.h"
#include "imgui_overlays.h"
#include "settings.h"
#include "shader_cache_version.h"
#include "system.h"

#include "util/gpu_device.h"
#include "util/imgui_manager.h"
#include "util/postprocessing.h"
#include "util/state_wrapper.h"

#include "common/align.h"
#include "common/error.h"
#include "common/log.h"
#include "common/threading.h"
#include "common/timer.h"

#include "IconsFontAwesome5.h"
#include "imgui.h"

#include <optional>

Log_SetChannel(GPUThread);

namespace GPUThread {
enum : u32
{
  COMMAND_QUEUE_SIZE = 4 * 1024 * 1024,
  THRESHOLD_TO_WAKE_GPU = 256
};

/// Starts the thread, if it hasn't already been started.
/// TODO: Persist thread
static bool Start(std::optional<GPURenderer> api, Error* error);

static void RunGPULoop();
static u32 GetPendingCommandSize();
static void WakeGPUThread();

static bool CreateDeviceOnThread(RenderAPI api, Error* error);
static void DestroyDeviceOnThread();

static void CreateGPUBackendOnThread(bool initialize_vram);
static void ChangeGPUBackendOnThread();
static void DestroyGPUBackendOnThread();

static void UpdateSettingsOnThread(const Settings& old_settings);
static void UpdateVSyncOnThread();
static void UpdatePerformanceCountersOnThread();

// TODO: Pack this crap together, don't trust LTO...

static RenderAPI s_render_api = RenderAPI::None;
static std::unique_ptr<GPUBackend> s_gpu_backend;
static std::optional<GPURenderer> s_requested_renderer;
static bool s_start_fullscreen_ui = false;
static GPUVSyncMode s_requested_vsync = GPUVSyncMode::Disabled;
static bool s_requested_allow_present_throttle = false;

static bool s_last_frame_skipped = false;

static Common::Timer::Value s_last_performance_counter_update_time = 0;
static u32 s_presents_since_last_update = 0;
static float s_accumulated_gpu_time = 0.0f;
static float s_average_gpu_time = 0.0f;
static float s_gpu_usage = 0.0f;

static Threading::KernelSemaphore m_sync_semaphore;
static Threading::Thread m_gpu_thread;
static Error s_open_error;
static std::atomic_bool s_open_flag{false};
static std::atomic_bool s_shutdown_flag{false};
static std::atomic_bool s_run_idle_flag{false};
static std::atomic_flag s_performance_counters_updated = ATOMIC_FLAG_INIT;

static Threading::WorkSema s_work_sema;

static FixedHeapArray<u8, COMMAND_QUEUE_SIZE> m_command_fifo_data;
alignas(HOST_CACHE_LINE_SIZE) static std::atomic<u32> m_command_fifo_read_ptr{0};
alignas(HOST_CACHE_LINE_SIZE) static std::atomic<u32> m_command_fifo_write_ptr{0};
} // namespace GPUThread

const Threading::ThreadHandle& GPUThread::GetThreadHandle()
{
  return m_gpu_thread;
}

RenderAPI GPUThread::GetRenderAPI()
{
  std::atomic_thread_fence(std::memory_order_acquire);
  return s_render_api;
}

bool GPUThread::IsStarted()
{
  return m_gpu_thread.Joinable();
}

bool GPUThread::WasFullscreenUIRequested()
{
  return s_start_fullscreen_ui;
}

bool GPUThread::Start(std::optional<GPURenderer> renderer, Error* error)
{
  Assert(!IsStarted());

  INFO_LOG("Starting GPU thread...");

  s_requested_renderer = renderer;
  g_gpu_settings = g_settings;

  s_last_performance_counter_update_time = Common::Timer::GetCurrentValue();
  s_presents_since_last_update = 0;
  s_average_gpu_time = 0.0f;
  s_gpu_usage = 0.0f;
  GPUBackend::ResetStatistics();

  s_shutdown_flag.store(false, std::memory_order_release);
  s_run_idle_flag.store(false, std::memory_order_release);
  m_gpu_thread.Start(&GPUThread::RunGPULoop);
  m_sync_semaphore.Wait();

  if (!s_open_flag.load(std::memory_order_acquire))
  {
    ERROR_LOG("Failed to create GPU thread.");
    if (error)
      *error = s_open_error;

    m_gpu_thread.Join();
    return false;
  }

  VERBOSE_LOG("GPU thread started.");
  return true;
}

bool GPUThread::StartFullscreenUI(Error* error)
{
  // NOTE: Racey read.
  if (FullscreenUI::IsInitialized())
    return true;

  if (IsStarted())
  {
    RunOnThread([]() {
      // TODO: Error handling.
      if (!FullscreenUI::Initialize())
        Panic("Failed");
    });

    return true;
  }

  s_start_fullscreen_ui = true;
  if (!Start(std::nullopt, error))
  {
    s_start_fullscreen_ui = false;
    return false;
  }

  return true;
}

std::optional<GPURenderer> GPUThread::GetRequestedRenderer()
{
  return s_requested_renderer;
}

bool GPUThread::CreateGPUBackend(GPURenderer renderer, Error* error)
{
  if (IsStarted())
  {
    s_requested_renderer = renderer;
    std::atomic_thread_fence(std::memory_order_release);
    PushCommandAndSync(AllocateCommand(GPUBackendCommandType::ChangeBackend, sizeof(GPUThreadCommand)), false);
    return true;
  }
  else
  {
    return Start(renderer, error);
  }
}

bool GPUThread::SwitchGPUBackend(GPURenderer renderer, bool force_recreate_device, Error* error)
{
  if (!force_recreate_device)
  {
    s_requested_renderer = renderer;
    std::atomic_thread_fence(std::memory_order_release);
    PushCommandAndSync(AllocateCommand(GPUBackendCommandType::ChangeBackend, sizeof(GPUThreadCommand)), false);
    return true;
  }

  const bool was_running_fsui = s_start_fullscreen_ui;
  Shutdown();
  s_requested_renderer = renderer;
  s_start_fullscreen_ui = was_running_fsui;
  if (!Start(renderer, error))
  {
    s_requested_renderer.reset();
    s_start_fullscreen_ui = false;
    return false;
  }

  return true;
}

void GPUThread::DestroyGPUBackend()
{
  if (!IsStarted())
    return;

  if (s_start_fullscreen_ui)
  {
    VERBOSE_LOG("Keeping GPU thread open for fullscreen UI");
    s_requested_renderer.reset();
    std::atomic_thread_fence(std::memory_order_release);
    PushCommandAndSync(AllocateCommand(GPUBackendCommandType::ChangeBackend, sizeof(GPUThreadCommand)), false);
    return;
  }

  Shutdown();
}

void GPUThread::Shutdown()
{
  if (!IsStarted())
    return;

  s_shutdown_flag.store(true, std::memory_order_release);
  s_start_fullscreen_ui = false;
  s_requested_renderer.reset();

  WakeGPUThread();
  m_gpu_thread.Join();
  INFO_LOG("GPU thread stopped.");
}

GPUThreadCommand* GPUThread::AllocateCommand(GPUBackendCommandType command, u32 size)
{
  // Ensure size is a multiple of 4 so we don't end up with an unaligned command.
  size = Common::AlignUpPow2(size, 4);

  for (;;)
  {
    u32 read_ptr = m_command_fifo_read_ptr.load(std::memory_order_acquire);
    u32 write_ptr = m_command_fifo_write_ptr.load(std::memory_order_relaxed);
    if (read_ptr > write_ptr)
    {
      u32 available_size = read_ptr - write_ptr;
      while (available_size < (size + sizeof(GPUBackendCommandType)))
      {
        WakeGPUThread();
        read_ptr = m_command_fifo_read_ptr.load(std::memory_order_acquire);
        available_size = (read_ptr > write_ptr) ? (read_ptr - write_ptr) : (COMMAND_QUEUE_SIZE - write_ptr);
      }
    }
    else
    {
      const u32 available_size = COMMAND_QUEUE_SIZE - write_ptr;
      if ((size + sizeof(GPUBackendCommand)) > available_size)
      {
        // allocate a dummy command to wrap the buffer around
        GPUBackendCommand* dummy_cmd = reinterpret_cast<GPUBackendCommand*>(&m_command_fifo_data[write_ptr]);
        dummy_cmd->type = GPUBackendCommandType::Wraparound;
        dummy_cmd->size = available_size;
        dummy_cmd->params.bits = 0;
        m_command_fifo_write_ptr.store(0, std::memory_order_release);
        continue;
      }
    }

    GPUThreadCommand* cmd = reinterpret_cast<GPUThreadCommand*>(&m_command_fifo_data[write_ptr]);
    cmd->type = command;
    cmd->size = size;
    return cmd;
  }
}

u32 GPUThread::GetPendingCommandSize()
{
  const u32 read_ptr = m_command_fifo_read_ptr.load();
  const u32 write_ptr = m_command_fifo_write_ptr.load();
  return (write_ptr >= read_ptr) ? (write_ptr - read_ptr) : (COMMAND_QUEUE_SIZE - read_ptr + write_ptr);
}

void GPUThread::PushCommand(GPUThreadCommand* cmd)
{
  const u32 new_write_ptr = m_command_fifo_write_ptr.fetch_add(cmd->size, std::memory_order_release) + cmd->size;
  DebugAssert(new_write_ptr <= COMMAND_QUEUE_SIZE);
  UNREFERENCED_VARIABLE(new_write_ptr);
  if (GetPendingCommandSize() >= THRESHOLD_TO_WAKE_GPU)
    WakeGPUThread();
}

void GPUThread::PushCommandAndWakeThread(GPUThreadCommand* cmd)
{
  const u32 new_write_ptr = m_command_fifo_write_ptr.fetch_add(cmd->size, std::memory_order_release) + cmd->size;
  DebugAssert(new_write_ptr <= COMMAND_QUEUE_SIZE);
  UNREFERENCED_VARIABLE(new_write_ptr);
  WakeGPUThread();
}

void GPUThread::PushCommandAndSync(GPUThreadCommand* cmd, bool spin)
{
  const u32 new_write_ptr = m_command_fifo_write_ptr.fetch_add(cmd->size, std::memory_order_release) + cmd->size;
  DebugAssert(new_write_ptr <= COMMAND_QUEUE_SIZE);
  UNREFERENCED_VARIABLE(new_write_ptr);
  WakeGPUThread();

  if (spin)
    s_work_sema.WaitForEmptyWithSpin();
  else
    s_work_sema.WaitForEmpty();
}

void GPUThread::WakeGPUThread()
{
  s_work_sema.NotifyOfWork();
}

void GPUThread::RunGPULoop()
{
  Threading::SetNameOfCurrentThread("GPUThread");

  if (!CreateDeviceOnThread(
        Settings::GetRenderAPIForRenderer(s_requested_renderer.value_or(g_gpu_settings.gpu_renderer)), &s_open_error))
  {
    Host::ReleaseRenderWindow();
    s_open_flag.store(false, std::memory_order_release);
    m_sync_semaphore.Post();
    return;
  }

  CreateGPUBackendOnThread(true);

  s_open_flag.store(true, std::memory_order_release);
  m_sync_semaphore.Post();

  for (;;)
  {
    u32 write_ptr = m_command_fifo_write_ptr.load(std::memory_order_acquire);
    u32 read_ptr = m_command_fifo_read_ptr.load(std::memory_order_relaxed);
    if (read_ptr == write_ptr)
    {
      if (s_shutdown_flag.load(std::memory_order_relaxed))
      {
        break;
      }
      else if (s_run_idle_flag.load(std::memory_order_relaxed))
      {
        if (!s_work_sema.CheckForWork())
        {
          Internal::PresentFrame(false, 0);
          if (!g_gpu_device->IsVSyncModeBlocking())
            g_gpu_device->ThrottlePresentation();

          continue;
        }

        // we should have something to do, since we got woken...
      }
      else
      {
        s_work_sema.WaitForWork();
        continue;
      }
    }

    write_ptr = (write_ptr < read_ptr) ? COMMAND_QUEUE_SIZE : write_ptr;
    while (read_ptr < write_ptr)
    {
      GPUThreadCommand* cmd = reinterpret_cast<GPUThreadCommand*>(&m_command_fifo_data[read_ptr]);
      DebugAssert((read_ptr + cmd->size) <= COMMAND_QUEUE_SIZE);
      read_ptr += cmd->size;

      switch (cmd->type)
      {
        case GPUBackendCommandType::Wraparound:
        {
          DebugAssert(read_ptr == COMMAND_QUEUE_SIZE);
          write_ptr = m_command_fifo_write_ptr.load(std::memory_order_acquire);
          read_ptr = 0;

          // let the CPU thread know as early as possible that we're here
          m_command_fifo_read_ptr.store(read_ptr, std::memory_order_release);
        }
        break;

        case GPUBackendCommandType::AsyncCall:
        {
          GPUThreadAsyncCallCommand* acmd = static_cast<GPUThreadAsyncCallCommand*>(cmd);
          acmd->func();
          acmd->~GPUThreadAsyncCallCommand();
        }
        break;

        case GPUBackendCommandType::ChangeBackend:
        {
          ChangeGPUBackendOnThread();
        }
        break;

        case GPUBackendCommandType::UpdateVSync:
        {
          UpdateVSyncOnThread();
        }
        break;

        default:
        {
          DebugAssert(s_gpu_backend);
          s_gpu_backend->HandleCommand(cmd);
        }
        break;
      }
    }

    m_command_fifo_read_ptr.store(read_ptr, std::memory_order_release);
  }

  DestroyGPUBackendOnThread();
  DestroyDeviceOnThread();
  Host::ReleaseRenderWindow();
}

bool GPUThread::CreateDeviceOnThread(RenderAPI api, Error* error)
{
  DebugAssert(!g_gpu_device);

  INFO_LOG("Trying to create a {} GPU device...", GPUDevice::RenderAPIToString(api));
  g_gpu_device = GPUDevice::CreateDeviceForAPI(api);

  std::optional<bool> exclusive_fullscreen_control;
  if (g_settings.display_exclusive_fullscreen_control != DisplayExclusiveFullscreenControl::Automatic)
  {
    exclusive_fullscreen_control =
      (g_settings.display_exclusive_fullscreen_control == DisplayExclusiveFullscreenControl::Allowed);
  }

  u32 disabled_features = 0;
  if (g_settings.gpu_disable_dual_source_blend)
    disabled_features |= GPUDevice::FEATURE_MASK_DUAL_SOURCE_BLEND;
  if (g_settings.gpu_disable_framebuffer_fetch)
    disabled_features |= GPUDevice::FEATURE_MASK_FRAMEBUFFER_FETCH;
  if (g_settings.gpu_disable_texture_buffers)
    disabled_features |= GPUDevice::FEATURE_MASK_TEXTURE_BUFFERS;
  if (g_settings.gpu_disable_texture_copy_to_self)
    disabled_features |= GPUDevice::FEATURE_MASK_TEXTURE_COPY_TO_SELF;

  Error create_error;
  if (!g_gpu_device ||
      !g_gpu_device->Create(
        g_gpu_settings.gpu_adapter,
        g_gpu_settings.gpu_disable_shader_cache ? std::string_view() : std::string_view(EmuFolders::Cache),
        SHADER_CACHE_VERSION, g_gpu_settings.gpu_use_debug_device, s_requested_vsync,
        s_requested_allow_present_throttle, g_gpu_settings.gpu_threaded_presentation, exclusive_fullscreen_control,
        static_cast<GPUDevice::FeatureMask>(disabled_features), &create_error))
  {
    ERROR_LOG("Failed to create GPU device: {}", create_error.GetDescription());
    if (g_gpu_device)
      g_gpu_device->Destroy();
    g_gpu_device.reset();

    Error::SetStringFmt(
      error,
      TRANSLATE_FS("System", "Failed to create render device:\n\n{0}\n\nThis may be due to your GPU not supporting the "
                             "chosen renderer ({1}), or because your graphics drivers need to be updated."),
      create_error.GetDescription(), GPUDevice::RenderAPIToString(api));

    s_render_api = RenderAPI::None;
    std::atomic_thread_fence(std::memory_order_release);
    return false;
  }

  if (!ImGuiManager::Initialize(g_settings.display_osd_scale / 100.0f, g_settings.display_show_osd_messages,
                                &create_error) ||
      (s_start_fullscreen_ui && !FullscreenUI::Initialize()))
  {
    ERROR_LOG("Failed to initialize ImGuiManager: {}", create_error.GetDescription());
    Error::SetStringFmt(error, "Failed to initialize ImGuiManager: {}", create_error.GetDescription());
    FullscreenUI::Shutdown();
    ImGuiManager::Shutdown();
    g_gpu_device->Destroy();
    g_gpu_device.reset();
    s_render_api = RenderAPI::None;
    std::atomic_thread_fence(std::memory_order_release);
    return false;
  }

  s_accumulated_gpu_time = 0.0f;
  s_presents_since_last_update = 0;
  s_render_api = g_gpu_device->GetRenderAPI();
  std::atomic_thread_fence(std::memory_order_release);

  return true;
}

void GPUThread::DestroyDeviceOnThread()
{
  if (!g_gpu_device)
    return;

  ImGuiManager::DestroyOverlayTextures();
  FullscreenUI::Shutdown();
  ImGuiManager::Shutdown();

  INFO_LOG("Destroying {} GPU device...", GPUDevice::RenderAPIToString(g_gpu_device->GetRenderAPI()));
  g_gpu_device->Destroy();
  g_gpu_device.reset();
}

void GPUThread::CreateGPUBackendOnThread(bool clear_vram)
{
  Assert(!s_gpu_backend);
  if (!s_requested_renderer.has_value())
    return;

  const bool is_hardware = (s_requested_renderer.value() != GPURenderer::Software);

  if (is_hardware)
    s_gpu_backend = GPUBackend::CreateHardwareBackend();
  else
    s_gpu_backend = GPUBackend::CreateSoftwareBackend();

  Error error;
  DebugAssert(s_gpu_backend);
  if (!s_gpu_backend->Initialize(clear_vram, &error))
  {
    ERROR_LOG("Failed to create {} renderer: {}", Settings::GetRendererName(s_requested_renderer.value()),
              error.GetDescription());

    if (is_hardware)
    {
      Host::AddIconOSDMessage(
        "GPUBackendCreationFailed", ICON_FA_PAINT_ROLLER,
        fmt::format(TRANSLATE_FS("OSDMessage", "Failed to initialize {} renderer, falling back to software renderer."),
                    Settings::GetRendererName(s_requested_renderer.value())),
        Host::OSD_CRITICAL_ERROR_DURATION);

      s_requested_renderer = GPURenderer::Software;
      s_gpu_backend = GPUBackend::CreateSoftwareBackend();
      if (!s_gpu_backend)
        Panic("Failed to initialize software backend.");
    }
  }
}

ALWAYS_INLINE_RELEASE void GPUThread::ChangeGPUBackendOnThread()
{
  std::atomic_thread_fence(std::memory_order_acquire);
  if (!s_requested_renderer.has_value())
  {
    if (s_gpu_backend)
      DestroyGPUBackendOnThread();

    return;
  }

  // Readback old VRAM for hardware renderers.
  s_gpu_backend->ReadVRAM(0, 0, VRAM_WIDTH, VRAM_HEIGHT);

  if (s_requested_renderer.value() == GPURenderer::Software)
  {
    // Just recreate the backend, software works with everything.
    DestroyGPUBackendOnThread();
    CreateGPUBackendOnThread(false);
    return;
  }

  DestroyGPUBackendOnThread();

  DebugAssert(g_gpu_device);
  const RenderAPI current_api = s_render_api;
  const RenderAPI expected_api = Settings::GetRenderAPIForRenderer(s_requested_renderer.value());
  if (!GPUDevice::IsSameRenderAPI(current_api, expected_api))
  {
    WARNING_LOG("Recreating GPU device, expecting {} got {}", GPUDevice::RenderAPIToString(expected_api),
                GPUDevice::RenderAPIToString(current_api));

    DestroyDeviceOnThread();

    // Things tend to break when you don't recreate the window, after switching APIs.
    Host::ReleaseRenderWindow();

    Error error;
    if (!CreateDeviceOnThread(expected_api, &error))
    {
      Host::AddIconOSDMessage(
        "DeviceSwitchFailed", ICON_FA_PAINT_ROLLER,
        fmt::format(TRANSLATE_FS("OSDMessage", "Failed to create {} GPU device, reverting to {}.\n{}"),
                    GPUDevice::RenderAPIToString(expected_api), GPUDevice::RenderAPIToString(current_api),
                    error.GetDescription()),
        Host::OSD_CRITICAL_ERROR_DURATION);

      Host::ReleaseRenderWindow();
      if (!CreateDeviceOnThread(current_api, &error))
        Panic("Failed to switch back to old API after creation failure");
    }
  }

  CreateGPUBackendOnThread(false);
}

void GPUThread::DestroyGPUBackendOnThread()
{
  if (!s_gpu_backend)
    return;

  VERBOSE_LOG("Shutting down GPU backend...");

  PostProcessing::Shutdown();
  s_gpu_backend.reset();
}

void GPUThread::UpdateSettingsOnThread(const Settings& old_settings)
{
  if (g_gpu_settings.display_show_gpu_usage != old_settings.display_show_gpu_usage ||
      g_gpu_settings.display_show_gpu_stats != old_settings.display_show_gpu_stats)
  {
    s_performance_counters_updated.clear(std::memory_order_relaxed);
    s_last_performance_counter_update_time = Common::Timer::GetCurrentValue();
    s_presents_since_last_update = 0;
  }

  if (g_gpu_settings.display_show_gpu_usage != old_settings.display_show_gpu_usage)
  {
    s_accumulated_gpu_time = 0.0f;
    s_average_gpu_time = 0.0f;
    s_gpu_usage = 0.0f;
  }

  if (s_gpu_backend)
    s_gpu_backend->UpdateSettings(old_settings);
}

void GPUThread::UpdateVSyncOnThread()
{
  std::atomic_thread_fence(std::memory_order_acquire);

  g_gpu_device->SetVSyncMode(s_requested_vsync, s_requested_allow_present_throttle);
}

void GPUThread::RunOnThread(AsyncCallType func)
{
  GPUThreadAsyncCallCommand* cmd = static_cast<GPUThreadAsyncCallCommand*>(
    AllocateCommand(GPUBackendCommandType::AsyncCall, sizeof(GPUThreadAsyncCallCommand)));
  new (cmd) GPUThreadAsyncCallCommand;
  cmd->func = std::move(func);
  PushCommandAndWakeThread(cmd);
}

void GPUThread::UpdateSettings()
{
  AssertMsg(IsStarted(), "GPU Thread is running");

  RunOnThread([settings = g_settings]() {
    VERBOSE_LOG("Updating GPU settings on thread...");

    Settings old_settings = std::move(g_gpu_settings);
    g_gpu_settings = std::move(settings);

    UpdateSettingsOnThread(old_settings);
  });
}

void GPUThread::ResizeDisplayWindow(s32 width, s32 height, float scale)
{
  AssertMsg(IsStarted(), "GPU Thread is running");
  RunOnThread([width, height, scale]() {
    if (!g_gpu_device)
      return;

    DEV_LOG("Display window resized to {}x{}", width, height);

    g_gpu_device->ResizeWindow(width, height, scale);
    ImGuiManager::WindowResized();

    // If we're paused, re-present the current frame at the new window size.
    if (System::IsValid())
    {
      if (System::IsPaused())
      {
        // Hackity hack, on some systems, presenting a single frame isn't enough to actually get it
        // displayed. Two seems to be good enough. Maybe something to do with direct scanout.
        PresentCurrentFrame();
        PresentCurrentFrame();
      }
    }
  });

  // TODO: The window size for GTE and stuff isn't going to be correct here.
  System::HostDisplayResized();
}

void GPUThread::UpdateDisplayWindow()
{
  AssertMsg(IsStarted(), "MTGS is running");
  RunOnThread([]() {
    if (!g_gpu_device)
      return;

    if (!g_gpu_device->UpdateWindow())
    {
      Host::ReportErrorAsync("Error", "Failed to change window after update. The log may contain more information.");
      return;
    }

    ImGuiManager::WindowResized();

    if (System::IsValid())
    {
      // Fix up vsync etc.
      System::UpdateSpeedLimiterState();

      // If we're paused, re-present the current frame at the new window size.
      if (System::IsPaused())
        PresentCurrentFrame();
    }
  });
}

void GPUThread::SetVSync(GPUVSyncMode mode, bool allow_present_throttle)
{
  Assert(IsStarted());

  if (s_requested_vsync == mode && s_requested_allow_present_throttle == allow_present_throttle)
    return;

  s_requested_vsync = mode;
  s_requested_allow_present_throttle = allow_present_throttle;
  std::atomic_thread_fence(std::memory_order_release);
  PushCommandAndWakeThread(AllocateCommand(GPUBackendCommandType::UpdateVSync, sizeof(GPUThreadCommand)));
}

void GPUThread::PresentCurrentFrame()
{
  if (s_run_idle_flag.load(std::memory_order_relaxed))
  {
    // If we're running idle, we're going to re-present anyway.
    return;
  }

  RunOnThread([]() { Internal::PresentFrame(false, 0); });
}

void GPUThread::Internal::PresentFrame(bool allow_skip_present, Common::Timer::Value present_time)
{
  // Make sure the GPU is flushed, otherwise the VB might still be mapped.
  // TODO: Make this suck less...
  if (s_gpu_backend)
    s_gpu_backend->FlushRender();

  s_presents_since_last_update++;
  if (!s_performance_counters_updated.test_and_set(std::memory_order_acq_rel))
    UpdatePerformanceCountersOnThread();

  const bool skip_present = (allow_skip_present && g_gpu_device->ShouldSkipPresentingFrame());
  const bool explicit_present = (present_time != 0 && g_gpu_device->GetFeatures().explicit_present);

  // TODO: Maybe?
  //(present_time != 0 && Common::Timer::GetCurrentValue() > present_time && !s_last_frame_skipped)));

  // acquire for IO.MousePos.
  std::atomic_thread_fence(std::memory_order_acquire);

  if (!skip_present)
  {
    FullscreenUI::Render();
    ImGuiManager::RenderTextOverlays();
    ImGuiManager::RenderOSDMessages();

    if (System::GetState() == System::State::Running)
      ImGuiManager::RenderSoftwareCursors();
  }

  // Debug windows are always rendered, otherwise mouse input breaks on skip.
  ImGuiManager::RenderOverlayWindows();
  ImGuiManager::RenderDebugWindows();

  if (s_gpu_backend && !skip_present)
    s_last_frame_skipped = !s_gpu_backend->PresentDisplay();
  else
    s_last_frame_skipped = !g_gpu_device->BeginPresent(skip_present);

  if (!s_last_frame_skipped)
  {
    g_gpu_device->RenderImGui();
    g_gpu_device->EndPresent(explicit_present);

    if (g_gpu_device->IsGPUTimingEnabled())
      s_accumulated_gpu_time += g_gpu_device->GetAndResetAccumulatedGPUTime();

    if (explicit_present)
    {
      // See note in System::Throttle().
#if !defined(__linux__) && !defined(__ANDROID__)
      Common::Timer::SleepUntil(present_time, false);
#else
      Common::Timer::SleepUntil(present_time, true);
#endif

      g_gpu_device->SubmitPresent();
    }
  }
  else
  {
    // Still need to kick ImGui or it gets cranky.
    ImGui::Render();
  }

  ImGuiManager::NewFrame();

  if (s_gpu_backend)
    s_gpu_backend->RestoreDeviceContext();
}

void GPUThread::SetRunIdle(bool enabled)
{
  s_run_idle_flag.store(enabled, std::memory_order_release);
  DEV_LOG("GPU thread now {} idle", enabled ? "running" : "NOT running");
}

float GPUThread::GetGPUUsage()
{
  return s_gpu_usage;
}

float GPUThread::GetGPUAverageTime()
{
  return s_average_gpu_time;
}

void GPUThread::SetPerformanceCounterUpdatePending()
{
  s_performance_counters_updated.clear(std::memory_order_release);
}

void GPUThread::UpdatePerformanceCountersOnThread()
{
  const Common::Timer::Value current_time = Common::Timer::GetCurrentValue();
  const u32 frames = std::exchange(s_presents_since_last_update, 0);
  const float time = static_cast<float>(Common::Timer::ConvertValueToSeconds(
    current_time - std::exchange(s_last_performance_counter_update_time, current_time)));

  if (g_gpu_device->IsGPUTimingEnabled())
  {
    s_average_gpu_time = s_accumulated_gpu_time / static_cast<float>(std::max(frames, 1u));
    s_gpu_usage = static_cast<float>(s_accumulated_gpu_time / (time * 10.0));
    s_accumulated_gpu_time = 0.0f;
  }

  if (g_settings.display_show_gpu_stats)
    GPUBackend::UpdateStatistics(frames);
}