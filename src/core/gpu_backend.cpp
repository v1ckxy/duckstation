// SPDX-FileCopyrightText: 2019-2024 Connor McLaughlin <stenzek@gmail.com>
// SPDX-License-Identifier: (GPL-3.0 OR CC-BY-NC-ND-4.0)

#include "gpu_backend.h"
#include "gpu.h"
#include "gpu_shadergen.h"
#include "gpu_thread.h"
#include "host.h"
#include "settings.h"

#include "util/gpu_device.h"
#include "util/image.h"
#include "util/imgui_manager.h"
#include "util/postprocessing.h"
#include "util/state_wrapper.h"

#include "common/align.h"
#include "common/error.h"
#include "common/file_system.h"
#include "common/gsvector_formatter.h"
#include "common/log.h"
#include "common/path.h"
#include "common/small_string.h"
#include "common/string_util.h"
#include "common/timer.h"

#include "IconsFontAwesome5.h"
#include "fmt/format.h"

#include <thread>

Log_SetChannel(GPUBackend);

namespace {
struct Counters
{
  u32 num_reads;
  u32 num_writes;
  u32 num_copies;
  u32 num_vertices;
  u32 num_primitives;

  // u32 num_read_texture_updates;
  // u32 num_ubo_updates;
};

// TODO: This is probably wrong/racey...
struct Stats : Counters
{
  size_t host_buffer_streamed;
  u32 host_num_draws;
  u32 host_num_barriers;
  u32 host_num_render_passes;
  u32 host_num_copies;
  u32 host_num_downloads;
  u32 host_num_uploads;
};
} // namespace

static bool CompressAndWriteTextureToFile(u32 width, u32 height, std::string filename, FileSystem::ManagedCFilePtr fp,
                                          u8 quality, bool clear_alpha, bool flip_y, std::vector<u32> texture_data,
                                          u32 texture_data_stride, GPUTexture::Format texture_format,
                                          bool display_osd_message, bool use_thread);
static void JoinScreenshotThreads();

// TODO: Pack state...

static std::atomic<u32> s_queued_frames;
static std::atomic_bool s_waiting_for_gpu_thread;
static Threading::KernelSemaphore s_gpu_thread_wait;

static std::tuple<u32, u32> s_last_display_source_size;

static std::deque<std::thread> s_screenshot_threads;
static std::mutex s_screenshot_threads_mutex;

static constexpr GPUTexture::Format DISPLAY_INTERNAL_POSTFX_FORMAT = GPUTexture::Format::RGBA8;

static Counters s_counters = {};
static Stats s_stats = {};

GPUBackend::GPUBackend()
{
  ResetStatistics();
}

GPUBackend::~GPUBackend()
{
  JoinScreenshotThreads();
  DestroyDeinterlaceTextures();
  g_gpu_device->RecycleTexture(std::move(m_chroma_smoothing_texture));
  g_gpu_device->SetGPUTimingEnabled(false);
}

bool GPUBackend::Initialize(bool clear_vram, Error* error)
{
  if (!CompileDisplayPipelines(true, true, g_gpu_settings.gpu_24bit_chroma_smoothing))
  {
    Error::SetStringView(error, "Failed to compile base GPU pipelines.");
    return false;
  }

  g_gpu_device->SetGPUTimingEnabled(g_gpu_settings.display_show_gpu_usage);
  return true;
}

void GPUBackend::UpdateSettings(const Settings& old_settings)
{
  FlushRender();

  if (g_settings.display_show_gpu_usage != old_settings.display_show_gpu_usage)
    g_gpu_device->SetGPUTimingEnabled(g_gpu_settings.display_show_gpu_usage);

  if (g_settings.display_show_gpu_stats != old_settings.display_show_gpu_stats)
    GPUBackend::ResetStatistics();

  if (g_gpu_settings.display_scaling != old_settings.display_scaling ||
      g_gpu_settings.display_deinterlacing_mode != old_settings.display_deinterlacing_mode ||
      g_gpu_settings.gpu_24bit_chroma_smoothing != old_settings.gpu_24bit_chroma_smoothing)
  {
    // Toss buffers on mode change.
    if (g_gpu_settings.display_deinterlacing_mode != old_settings.display_deinterlacing_mode)
      DestroyDeinterlaceTextures();

    if (!CompileDisplayPipelines(g_gpu_settings.display_scaling != old_settings.display_scaling,
                                 g_gpu_settings.display_deinterlacing_mode != old_settings.display_deinterlacing_mode,
                                 g_gpu_settings.gpu_24bit_chroma_smoothing != old_settings.gpu_24bit_chroma_smoothing))
    {
      Panic("Failed to compile display pipeline on settings change.");
    }
  }
}

void GPUBackend::RestoreDeviceContext()
{
}

GPUThreadCommand* GPUBackend::NewClearVRAMCommand()
{
  return static_cast<GPUThreadCommand*>(
    GPUThread::AllocateCommand(GPUBackendCommandType::ClearVRAM, sizeof(GPUThreadCommand)));
}

GPUBackendDoStateCommand* GPUBackend::NewDoStateCommand()
{
  return static_cast<GPUBackendDoStateCommand*>(
    GPUThread::AllocateCommand(GPUBackendCommandType::DoState, sizeof(GPUBackendDoStateCommand)));
}

GPUThreadCommand* GPUBackend::NewClearDisplayCommand()
{
  return static_cast<GPUThreadCommand*>(
    GPUThread::AllocateCommand(GPUBackendCommandType::ClearDisplay, sizeof(GPUThreadCommand)));
}

GPUBackendUpdateDisplayCommand* GPUBackend::NewUpdateDisplayCommand()
{
  return static_cast<GPUBackendUpdateDisplayCommand*>(
    GPUThread::AllocateCommand(GPUBackendCommandType::UpdateDisplay, sizeof(GPUBackendUpdateDisplayCommand)));
}

GPUThreadCommand* GPUBackend::NewClearCacheCommand()
{
  return static_cast<GPUThreadCommand*>(
    GPUThread::AllocateCommand(GPUBackendCommandType::ClearCache, sizeof(GPUThreadCommand)));
}

GPUThreadCommand* GPUBackend::NewBufferSwappedCommand()
{
  return static_cast<GPUThreadCommand*>(
    GPUThread::AllocateCommand(GPUBackendCommandType::BufferSwapped, sizeof(GPUThreadCommand)));
}

GPUThreadCommand* GPUBackend::NewFlushRenderCommand()
{
  return static_cast<GPUThreadCommand*>(
    GPUThread::AllocateCommand(GPUBackendCommandType::FlushRender, sizeof(GPUThreadCommand)));
}

GPUThreadCommand* GPUBackend::NewUpdateResolutionScaleCommand()
{
  return static_cast<GPUThreadCommand*>(
    GPUThread::AllocateCommand(GPUBackendCommandType::UpdateResolutionScale, sizeof(GPUThreadCommand)));
}

GPUBackendReadVRAMCommand* GPUBackend::NewReadVRAMCommand()
{
  return static_cast<GPUBackendReadVRAMCommand*>(
    GPUThread::AllocateCommand(GPUBackendCommandType::ReadVRAM, sizeof(GPUBackendReadVRAMCommand)));
}

GPUBackendFillVRAMCommand* GPUBackend::NewFillVRAMCommand()
{
  return static_cast<GPUBackendFillVRAMCommand*>(
    GPUThread::AllocateCommand(GPUBackendCommandType::FillVRAM, sizeof(GPUBackendFillVRAMCommand)));
}

GPUBackendUpdateVRAMCommand* GPUBackend::NewUpdateVRAMCommand(u32 num_words)
{
  const u32 size = sizeof(GPUBackendUpdateVRAMCommand) + (num_words * sizeof(u16));
  GPUBackendUpdateVRAMCommand* cmd =
    static_cast<GPUBackendUpdateVRAMCommand*>(GPUThread::AllocateCommand(GPUBackendCommandType::UpdateVRAM, size));
  return cmd;
}

GPUBackendCopyVRAMCommand* GPUBackend::NewCopyVRAMCommand()
{
  return static_cast<GPUBackendCopyVRAMCommand*>(
    GPUThread::AllocateCommand(GPUBackendCommandType::CopyVRAM, sizeof(GPUBackendCopyVRAMCommand)));
}

GPUBackendSetDrawingAreaCommand* GPUBackend::NewSetDrawingAreaCommand()
{
  return static_cast<GPUBackendSetDrawingAreaCommand*>(
    GPUThread::AllocateCommand(GPUBackendCommandType::SetDrawingArea, sizeof(GPUBackendSetDrawingAreaCommand)));
}

GPUBackendUpdateCLUTCommand* GPUBackend::NewUpdateCLUTCommand()
{
  return static_cast<GPUBackendUpdateCLUTCommand*>(
    GPUThread::AllocateCommand(GPUBackendCommandType::UpdateCLUT, sizeof(GPUBackendUpdateCLUTCommand)));
}

GPUBackendDrawPolygonCommand* GPUBackend::NewDrawPolygonCommand(u32 num_vertices)
{
  const u32 size = sizeof(GPUBackendDrawPolygonCommand) + (num_vertices * sizeof(GPUBackendDrawPolygonCommand::Vertex));
  GPUBackendDrawPolygonCommand* cmd =
    static_cast<GPUBackendDrawPolygonCommand*>(GPUThread::AllocateCommand(GPUBackendCommandType::DrawPolygon, size));
  cmd->num_vertices = Truncate8(num_vertices);
  return cmd;
}

GPUBackendDrawPrecisePolygonCommand* GPUBackend::NewDrawPrecisePolygonCommand(u32 num_vertices)
{
  const u32 size =
    sizeof(GPUBackendDrawPrecisePolygonCommand) + (num_vertices * sizeof(GPUBackendDrawPrecisePolygonCommand::Vertex));
  GPUBackendDrawPrecisePolygonCommand* cmd = static_cast<GPUBackendDrawPrecisePolygonCommand*>(
    GPUThread::AllocateCommand(GPUBackendCommandType::DrawPrecisePolygon, size));
  cmd->num_vertices = Truncate8(num_vertices);
  return cmd;
}

GPUBackendDrawRectangleCommand* GPUBackend::NewDrawRectangleCommand()
{
  return static_cast<GPUBackendDrawRectangleCommand*>(
    GPUThread::AllocateCommand(GPUBackendCommandType::DrawRectangle, sizeof(GPUBackendDrawRectangleCommand)));
}

GPUBackendDrawLineCommand* GPUBackend::NewDrawLineCommand(u32 num_vertices)
{
  const u32 size = sizeof(GPUBackendDrawLineCommand) + (num_vertices * sizeof(GPUBackendDrawLineCommand::Vertex));
  GPUBackendDrawLineCommand* cmd =
    static_cast<GPUBackendDrawLineCommand*>(GPUThread::AllocateCommand(GPUBackendCommandType::DrawLine, size));
  cmd->num_vertices = Truncate16(num_vertices);
  return cmd;
}

void GPUBackend::PushCommand(GPUThreadCommand* cmd)
{
  GPUThread::PushCommand(cmd);
}

void GPUBackend::PushCommandAndWakeThread(GPUThreadCommand* cmd)
{
  GPUThread::PushCommandAndWakeThread(cmd);
}

void GPUBackend::PushCommandAndSync(GPUThreadCommand* cmd, bool spin)
{
  GPUThread::PushCommandAndSync(cmd, spin);
}

bool GPUBackend::IsUsingHardwareBackend()
{
  return (GPUThread::GetRequestedRenderer().value_or(GPURenderer::Software) != GPURenderer::Software);
}

bool GPUBackend::BeginQueueFrame()
{
  const u32 queued_frames = s_queued_frames.fetch_add(1, std::memory_order_acq_rel) + 1;
  if (queued_frames < g_settings.gpu_max_queued_frames)
    return false;

  DEV_LOG("<-- {} queued frames, {} max, blocking CPU thread", queued_frames, g_settings.gpu_max_queued_frames);
  s_waiting_for_gpu_thread.store(true, std::memory_order_release);
  return true;
}

void GPUBackend::WaitForOneQueuedFrame()
{
  s_gpu_thread_wait.Wait();
}

bool GPUBackend::RenderScreenshotToBuffer(u32 width, u32 height, const GSVector4i draw_rect, bool postfx,
                                          std::vector<u32>* out_pixels, u32* out_stride, GPUTexture::Format* out_format)
{
  bool result;

  GPUThreadRenderScreenshotToBufferCommand* cmd =
    static_cast<GPUThreadRenderScreenshotToBufferCommand*>(GPUThread::AllocateCommand(
      GPUBackendCommandType::RenderScreenshotToBuffer, sizeof(GPUThreadRenderScreenshotToBufferCommand)));
  cmd->width = width;
  cmd->height = height;
  GSVector4i::store<false>(cmd->draw_rect, draw_rect);
  cmd->postfx = postfx;
  cmd->out_pixels = out_pixels;
  cmd->out_stride = out_stride;
  cmd->out_format = out_format;
  cmd->out_result = &result;
  PushCommandAndSync(cmd, false);

  return result;
}

std::tuple<u32, u32> GPUBackend::GetLastDisplaySourceSize()
{
  std::atomic_thread_fence(std::memory_order_acquire);
  return s_last_display_source_size;
}

void GPUBackend::HandleCommand(const GPUThreadCommand* cmd)
{
  switch (cmd->type)
  {
    case GPUBackendCommandType::ClearVRAM:
    {
      ClearVRAM();
    }
    break;

    case GPUBackendCommandType::DoState:
    {
      const GPUBackendDoStateCommand* ccmd = static_cast<const GPUBackendDoStateCommand*>(cmd);
      DoState(ccmd->host_texture, ccmd->is_reading, ccmd->update_display);
    }
    break;

    case GPUBackendCommandType::ClearDisplay:
    {
      ClearDisplay();
    }
    break;

    case GPUBackendCommandType::UpdateDisplay:
    {
      const GPUBackendUpdateDisplayCommand* ccmd = static_cast<const GPUBackendUpdateDisplayCommand*>(cmd);
      m_display_width = ccmd->display_width;
      m_display_height = ccmd->display_height;
      m_display_origin_left = ccmd->display_origin_left;
      m_display_origin_top = ccmd->display_origin_top;
      m_display_vram_width = ccmd->display_vram_width;
      m_display_vram_height = ccmd->display_vram_height;
      m_display_aspect_ratio = ccmd->display_aspect_ratio;

      UpdateDisplay(ccmd);

      if (ccmd->present_frame)
      {
        GPUThread::Internal::PresentFrame(true, ccmd->present_time);

        s_queued_frames.fetch_sub(1);

        bool expected = true;
        if (s_waiting_for_gpu_thread.compare_exchange_strong(expected, false))
        {
          DEV_LOG("--> Unblocking CPU thread");
          s_gpu_thread_wait.Post();
        }
      }
    }
    break;

    case GPUBackendCommandType::ClearCache:
    {
      ClearCache();
    }
    break;

    case GPUBackendCommandType::BufferSwapped:
    {
      OnBufferSwapped();
    }
    break;

    case GPUBackendCommandType::FlushRender:
    {
      FlushRender();
    }
    break;

    case GPUBackendCommandType::UpdateResolutionScale:
    {
      UpdateResolutionScale();
    }
    break;

    case GPUBackendCommandType::RenderScreenshotToBuffer:
    {
      HandleRenderScreenshotToBuffer(static_cast<const GPUThreadRenderScreenshotToBufferCommand*>(cmd));
    }
    break;

    case GPUBackendCommandType::ReadVRAM:
    {
      const GPUBackendReadVRAMCommand* ccmd = static_cast<const GPUBackendReadVRAMCommand*>(cmd);
      s_counters.num_reads++;
      ReadVRAM(ZeroExtend32(ccmd->x), ZeroExtend32(ccmd->y), ZeroExtend32(ccmd->width), ZeroExtend32(ccmd->height));
    }
    break;

    case GPUBackendCommandType::FillVRAM:
    {
      const GPUBackendFillVRAMCommand* ccmd = static_cast<const GPUBackendFillVRAMCommand*>(cmd);
      FillVRAM(ZeroExtend32(ccmd->x), ZeroExtend32(ccmd->y), ZeroExtend32(ccmd->width), ZeroExtend32(ccmd->height),
               ccmd->color, ccmd->params);
    }
    break;

    case GPUBackendCommandType::UpdateVRAM:
    {
      const GPUBackendUpdateVRAMCommand* ccmd = static_cast<const GPUBackendUpdateVRAMCommand*>(cmd);
      s_counters.num_writes++;
      UpdateVRAM(ZeroExtend32(ccmd->x), ZeroExtend32(ccmd->y), ZeroExtend32(ccmd->width), ZeroExtend32(ccmd->height),
                 ccmd->data, ccmd->params);
    }
    break;

    case GPUBackendCommandType::CopyVRAM:
    {
      const GPUBackendCopyVRAMCommand* ccmd = static_cast<const GPUBackendCopyVRAMCommand*>(cmd);
      s_counters.num_copies++;
      CopyVRAM(ZeroExtend32(ccmd->src_x), ZeroExtend32(ccmd->src_y), ZeroExtend32(ccmd->dst_x),
               ZeroExtend32(ccmd->dst_y), ZeroExtend32(ccmd->width), ZeroExtend32(ccmd->height), ccmd->params);
    }
    break;

    case GPUBackendCommandType::SetDrawingArea:
    {
      FlushRender();
      const GPUBackendSetDrawingAreaCommand* ccmd = static_cast<const GPUBackendSetDrawingAreaCommand*>(cmd);
      DrawingAreaChanged(ccmd->new_area, GSVector4i::load<false>(ccmd->new_clamped_area));
    }
    break;

    case GPUBackendCommandType::UpdateCLUT:
    {
      const GPUBackendUpdateCLUTCommand* ccmd = static_cast<const GPUBackendUpdateCLUTCommand*>(cmd);
      UpdateCLUT(ccmd->reg, ccmd->clut_is_8bit);
    }
    break;

    case GPUBackendCommandType::DrawPolygon:
    {
      const GPUBackendDrawPolygonCommand* ccmd = static_cast<const GPUBackendDrawPolygonCommand*>(cmd);
      s_counters.num_vertices += ccmd->num_vertices;
      s_counters.num_primitives++;
      DrawPolygon(ccmd);
    }
    break;

    case GPUBackendCommandType::DrawPrecisePolygon:
    {
      const GPUBackendDrawPolygonCommand* ccmd = static_cast<const GPUBackendDrawPolygonCommand*>(cmd);
      s_counters.num_vertices += ccmd->num_vertices;
      s_counters.num_primitives++;
      DrawPrecisePolygon(static_cast<const GPUBackendDrawPrecisePolygonCommand*>(cmd));
    }
    break;

    case GPUBackendCommandType::DrawRectangle:
    {
      const GPUBackendDrawRectangleCommand* ccmd = static_cast<const GPUBackendDrawRectangleCommand*>(cmd);
      s_counters.num_vertices++;
      s_counters.num_primitives++;
      DrawSprite(ccmd);
    }
    break;

    case GPUBackendCommandType::DrawLine:
    {
      const GPUBackendDrawLineCommand* ccmd = static_cast<const GPUBackendDrawLineCommand*>(cmd);
      s_counters.num_vertices += ccmd->num_vertices;
      s_counters.num_primitives += ccmd->num_vertices / 2;
      DrawLine(ccmd);
    }
    break;

      DefaultCaseIsUnreachable();
  }
}

void GPUBackend::FillVRAM(u32 x, u32 y, u32 width, u32 height, u32 color, GPUBackendCommandParameters params)
{
  const u16 color16 = VRAMRGBA8888ToRGBA5551(color);
  if ((x + width) <= VRAM_WIDTH && !params.interlaced_rendering)
  {
    for (u32 yoffs = 0; yoffs < height; yoffs++)
    {
      const u32 row = (y + yoffs) % VRAM_HEIGHT;
      std::fill_n(&g_vram[row * VRAM_WIDTH + x], width, color16);
    }
  }
  else if (params.interlaced_rendering)
  {
    // Hardware tests show that fills seem to break on the first two lines when the offset matches the displayed field.
    const u32 active_field = params.active_line_lsb;
    for (u32 yoffs = 0; yoffs < height; yoffs++)
    {
      const u32 row = (y + yoffs) % VRAM_HEIGHT;
      if ((row & u32(1)) == active_field)
        continue;

      u16* row_ptr = &g_vram[row * VRAM_WIDTH];
      for (u32 xoffs = 0; xoffs < width; xoffs++)
      {
        const u32 col = (x + xoffs) % VRAM_WIDTH;
        row_ptr[col] = color16;
      }
    }
  }
  else
  {
    for (u32 yoffs = 0; yoffs < height; yoffs++)
    {
      const u32 row = (y + yoffs) % VRAM_HEIGHT;
      u16* row_ptr = &g_vram[row * VRAM_WIDTH];
      for (u32 xoffs = 0; xoffs < width; xoffs++)
      {
        const u32 col = (x + xoffs) % VRAM_WIDTH;
        row_ptr[col] = color16;
      }
    }
  }
}

void GPUBackend::UpdateVRAM(u32 x, u32 y, u32 width, u32 height, const void* data, GPUBackendCommandParameters params)
{
  // Fast path when the copy is not oversized.
  if ((x + width) <= VRAM_WIDTH && (y + height) <= VRAM_HEIGHT && !params.IsMaskingEnabled())
  {
    const u16* src_ptr = static_cast<const u16*>(data);
    u16* dst_ptr = &g_vram[y * VRAM_WIDTH + x];
    for (u32 yoffs = 0; yoffs < height; yoffs++)
    {
      std::copy_n(src_ptr, width, dst_ptr);
      src_ptr += width;
      dst_ptr += VRAM_WIDTH;
    }
  }
  else
  {
    // Slow path when we need to handle wrap-around.
    // During transfer/render operations, if ((dst_pixel & mask_and) == 0) { pixel = src_pixel | mask_or }
    const u16* src_ptr = static_cast<const u16*>(data);
    const u16 mask_and = params.GetMaskAND();
    const u16 mask_or = params.GetMaskOR();

    for (u32 row = 0; row < height;)
    {
      u16* dst_row_ptr = &g_vram[((y + row++) % VRAM_HEIGHT) * VRAM_WIDTH];
      for (u32 col = 0; col < width;)
      {
        // TODO: Handle unaligned reads...
        u16* pixel_ptr = &dst_row_ptr[(x + col++) % VRAM_WIDTH];
        if (((*pixel_ptr) & mask_and) == 0)
          *pixel_ptr = *(src_ptr++) | mask_or;
      }
    }
  }
}

void GPUBackend::CopyVRAM(u32 src_x, u32 src_y, u32 dst_x, u32 dst_y, u32 width, u32 height,
                          GPUBackendCommandParameters params)
{
  // Break up oversized copies. This behavior has not been verified on console.
  if ((src_x + width) > VRAM_WIDTH || (dst_x + width) > VRAM_WIDTH)
  {
    u32 remaining_rows = height;
    u32 current_src_y = src_y;
    u32 current_dst_y = dst_y;
    while (remaining_rows > 0)
    {
      const u32 rows_to_copy =
        std::min<u32>(remaining_rows, std::min<u32>(VRAM_HEIGHT - current_src_y, VRAM_HEIGHT - current_dst_y));

      u32 remaining_columns = width;
      u32 current_src_x = src_x;
      u32 current_dst_x = dst_x;
      while (remaining_columns > 0)
      {
        const u32 columns_to_copy =
          std::min<u32>(remaining_columns, std::min<u32>(VRAM_WIDTH - current_src_x, VRAM_WIDTH - current_dst_x));
        CopyVRAM(current_src_x, current_src_y, current_dst_x, current_dst_y, columns_to_copy, rows_to_copy, params);
        current_src_x = (current_src_x + columns_to_copy) % VRAM_WIDTH;
        current_dst_x = (current_dst_x + columns_to_copy) % VRAM_WIDTH;
        remaining_columns -= columns_to_copy;
      }

      current_src_y = (current_src_y + rows_to_copy) % VRAM_HEIGHT;
      current_dst_y = (current_dst_y + rows_to_copy) % VRAM_HEIGHT;
      remaining_rows -= rows_to_copy;
    }

    return;
  }

  // This doesn't have a fast path, but do we really need one? It's not common.
  const u16 mask_and = params.GetMaskAND();
  const u16 mask_or = params.GetMaskOR();

  // Copy in reverse when src_x < dst_x, this is verified on console.
  if (src_x < dst_x || ((src_x + width - 1) % VRAM_WIDTH) < ((dst_x + width - 1) % VRAM_WIDTH))
  {
    for (u32 row = 0; row < height; row++)
    {
      const u16* src_row_ptr = &g_vram[((src_y + row) % VRAM_HEIGHT) * VRAM_WIDTH];
      u16* dst_row_ptr = &g_vram[((dst_y + row) % VRAM_HEIGHT) * VRAM_WIDTH];

      for (s32 col = static_cast<s32>(width - 1); col >= 0; col--)
      {
        const u16 src_pixel = src_row_ptr[(src_x + static_cast<u32>(col)) % VRAM_WIDTH];
        u16* dst_pixel_ptr = &dst_row_ptr[(dst_x + static_cast<u32>(col)) % VRAM_WIDTH];
        if ((*dst_pixel_ptr & mask_and) == 0)
          *dst_pixel_ptr = src_pixel | mask_or;
      }
    }
  }
  else
  {
    for (u32 row = 0; row < height; row++)
    {
      const u16* src_row_ptr = &g_vram[((src_y + row) % VRAM_HEIGHT) * VRAM_WIDTH];
      u16* dst_row_ptr = &g_vram[((dst_y + row) % VRAM_HEIGHT) * VRAM_WIDTH];

      for (u32 col = 0; col < width; col++)
      {
        const u16 src_pixel = src_row_ptr[(src_x + col) % VRAM_WIDTH];
        u16* dst_pixel_ptr = &dst_row_ptr[(dst_x + col) % VRAM_WIDTH];
        if ((*dst_pixel_ptr & mask_and) == 0)
          *dst_pixel_ptr = src_pixel | mask_or;
      }
    }
  }
}

bool GPUBackend::CompileDisplayPipelines(bool display, bool deinterlace, bool chroma_smoothing)
{
  GPUShaderGen shadergen(g_gpu_device->GetRenderAPI(), g_gpu_device->GetFeatures().dual_source_blend,
                         g_gpu_device->GetFeatures().framebuffer_fetch);

  GPUPipeline::GraphicsConfig plconfig;
  plconfig.input_layout.vertex_stride = 0;
  plconfig.primitive = GPUPipeline::Primitive::Triangles;
  plconfig.rasterization = GPUPipeline::RasterizationState::GetNoCullState();
  plconfig.depth = GPUPipeline::DepthState::GetNoTestsState();
  plconfig.blend = GPUPipeline::BlendState::GetNoBlendingState();
  plconfig.geometry_shader = nullptr;
  plconfig.depth_format = GPUTexture::Format::Unknown;
  plconfig.samples = 1;
  plconfig.per_sample_shading = false;
  plconfig.render_pass_flags = GPUPipeline::NoRenderPassFlags;

  if (display)
  {
    plconfig.layout = GPUPipeline::Layout::SingleTextureAndPushConstants;
    plconfig.SetTargetFormats(g_gpu_device->HasSurface() ? g_gpu_device->GetWindowFormat() : GPUTexture::Format::RGBA8);

    std::string vs = shadergen.GenerateDisplayVertexShader();
    std::string fs;
    switch (g_settings.display_scaling)
    {
      case DisplayScalingMode::BilinearSharp:
        fs = shadergen.GenerateDisplaySharpBilinearFragmentShader();
        break;

      case DisplayScalingMode::BilinearSmooth:
        fs = shadergen.GenerateDisplayFragmentShader(true);
        break;

      case DisplayScalingMode::Nearest:
      case DisplayScalingMode::NearestInteger:
      default:
        fs = shadergen.GenerateDisplayFragmentShader(false);
        break;
    }

    std::unique_ptr<GPUShader> vso = g_gpu_device->CreateShader(GPUShaderStage::Vertex, shadergen.GetLanguage(), vs);
    std::unique_ptr<GPUShader> fso = g_gpu_device->CreateShader(GPUShaderStage::Fragment, shadergen.GetLanguage(), fs);
    if (!vso || !fso)
      return false;
    GL_OBJECT_NAME(vso, "Display Vertex Shader");
    GL_OBJECT_NAME_FMT(fso, "Display Fragment Shader [{}]",
                       Settings::GetDisplayScalingName(g_settings.display_scaling));
    plconfig.vertex_shader = vso.get();
    plconfig.fragment_shader = fso.get();
    if (!(m_display_pipeline = g_gpu_device->CreatePipeline(plconfig)))
      return false;
    GL_OBJECT_NAME_FMT(m_display_pipeline, "Display Pipeline [{}]",
                       Settings::GetDisplayScalingName(g_settings.display_scaling));
  }

  if (deinterlace)
  {
    plconfig.SetTargetFormats(GPUTexture::Format::RGBA8);

    std::unique_ptr<GPUShader> vso = g_gpu_device->CreateShader(GPUShaderStage::Vertex, shadergen.GetLanguage(),
                                                                shadergen.GenerateScreenQuadVertexShader());
    if (!vso)
      return false;
    GL_OBJECT_NAME(vso, "Deinterlace Vertex Shader");

    std::unique_ptr<GPUShader> fso;
    if (!(fso = g_gpu_device->CreateShader(GPUShaderStage::Fragment, shadergen.GetLanguage(),
                                           shadergen.GenerateInterleavedFieldExtractFragmentShader())))
    {
      return false;
    }

    GL_OBJECT_NAME(fso, "Deinterlace Field Extract Fragment Shader");

    plconfig.layout = GPUPipeline::Layout::SingleTextureAndPushConstants;
    plconfig.vertex_shader = vso.get();
    plconfig.fragment_shader = fso.get();
    if (!(m_deinterlace_extract_pipeline = g_gpu_device->CreatePipeline(plconfig)))
      return false;

    GL_OBJECT_NAME(m_deinterlace_extract_pipeline, "Deinterlace Field Extract Pipeline");

    switch (g_settings.display_deinterlacing_mode)
    {
      case DisplayDeinterlacingMode::Disabled:
        break;

      case DisplayDeinterlacingMode::Weave:
      {
        if (!(fso = g_gpu_device->CreateShader(GPUShaderStage::Fragment, shadergen.GetLanguage(),
                                               shadergen.GenerateDeinterlaceWeaveFragmentShader())))
        {
          return false;
        }

        GL_OBJECT_NAME(fso, "Weave Deinterlace Fragment Shader");

        plconfig.layout = GPUPipeline::Layout::SingleTextureAndPushConstants;
        plconfig.vertex_shader = vso.get();
        plconfig.fragment_shader = fso.get();
        if (!(m_deinterlace_pipeline = g_gpu_device->CreatePipeline(plconfig)))
          return false;

        GL_OBJECT_NAME(m_deinterlace_pipeline, "Weave Deinterlace Pipeline");
      }
      break;

      case DisplayDeinterlacingMode::Blend:
      {
        if (!(fso = g_gpu_device->CreateShader(GPUShaderStage::Fragment, shadergen.GetLanguage(),
                                               shadergen.GenerateDeinterlaceBlendFragmentShader())))
        {
          return false;
        }

        GL_OBJECT_NAME(fso, "Blend Deinterlace Fragment Shader");

        plconfig.layout = GPUPipeline::Layout::MultiTextureAndPushConstants;
        plconfig.vertex_shader = vso.get();
        plconfig.fragment_shader = fso.get();
        if (!(m_deinterlace_pipeline = g_gpu_device->CreatePipeline(plconfig)))
          return false;

        GL_OBJECT_NAME(m_deinterlace_pipeline, "Blend Deinterlace Pipeline");
      }
      break;

      case DisplayDeinterlacingMode::Adaptive:
      {
        fso = g_gpu_device->CreateShader(GPUShaderStage::Fragment, shadergen.GetLanguage(),
                                         shadergen.GenerateFastMADReconstructFragmentShader());
        if (!fso)
          return false;

        GL_OBJECT_NAME(fso, "FastMAD Reconstruct Fragment Shader");

        plconfig.layout = GPUPipeline::Layout::MultiTextureAndPushConstants;
        plconfig.fragment_shader = fso.get();
        if (!(m_deinterlace_pipeline = g_gpu_device->CreatePipeline(plconfig)))
          return false;

        GL_OBJECT_NAME(m_deinterlace_pipeline, "FastMAD Reconstruct Pipeline");
      }
      break;

      default:
        UnreachableCode();
    }
  }

  if (chroma_smoothing)
  {
    m_chroma_smoothing_pipeline.reset();
    g_gpu_device->RecycleTexture(std::move(m_chroma_smoothing_texture));

    if (g_settings.gpu_24bit_chroma_smoothing)
    {
      plconfig.layout = GPUPipeline::Layout::SingleTextureAndPushConstants;
      plconfig.SetTargetFormats(GPUTexture::Format::RGBA8);

      std::unique_ptr<GPUShader> vso = g_gpu_device->CreateShader(GPUShaderStage::Vertex, shadergen.GetLanguage(),
                                                                  shadergen.GenerateScreenQuadVertexShader());
      std::unique_ptr<GPUShader> fso = g_gpu_device->CreateShader(GPUShaderStage::Fragment, shadergen.GetLanguage(),
                                                                  shadergen.GenerateChromaSmoothingFragmentShader());
      if (!vso || !fso)
        return false;
      GL_OBJECT_NAME(vso, "Chroma Smoothing Vertex Shader");
      GL_OBJECT_NAME(fso, "Chroma Smoothing Fragment Shader");

      plconfig.vertex_shader = vso.get();
      plconfig.fragment_shader = fso.get();
      if (!(m_chroma_smoothing_pipeline = g_gpu_device->CreatePipeline(plconfig)))
        return false;
      GL_OBJECT_NAME(m_chroma_smoothing_pipeline, "Chroma Smoothing Pipeline");
    }
  }

  return true;
}

void GPUBackend::ClearDisplay()
{
  ClearDisplayTexture();

  // Just recycle the textures, it'll get re-fetched.
  DestroyDeinterlaceTextures();
}

void GPUBackend::ClearDisplayTexture()
{
  m_display_texture = nullptr;
  m_display_texture_view_x = 0;
  m_display_texture_view_y = 0;
  m_display_texture_view_width = 0;
  m_display_texture_view_height = 0;
  s_last_display_source_size = {};
  std::atomic_thread_fence(std::memory_order_release);
}

void GPUBackend::SetDisplayTexture(GPUTexture* texture, GPUTexture* depth_texture, s32 view_x, s32 view_y,
                                   s32 view_width, s32 view_height)
{
  DebugAssert(texture);
  m_display_texture = texture;
  m_display_depth_buffer = depth_texture;
  m_display_texture_view_x = view_x;
  m_display_texture_view_y = view_y;
  m_display_texture_view_width = view_width;
  m_display_texture_view_height = view_height;
  s_last_display_source_size = {static_cast<u32>(view_width), static_cast<u32>(view_height)};
  std::atomic_thread_fence(std::memory_order_release);
}

bool GPUBackend::PresentDisplay()
{
  if (!HasDisplayTexture())
    return g_gpu_device->BeginPresent(false);

  const GSVector4i draw_rect = CalculateDrawRect(g_gpu_device->GetWindowWidth(), g_gpu_device->GetWindowHeight());
  return RenderDisplay(nullptr, draw_rect, !g_gpu_settings.debugging.show_vram);
}

bool GPUBackend::RenderDisplay(GPUTexture* target, const GSVector4i draw_rect, bool postfx)
{
  GL_SCOPE_FMT("RenderDisplay: {}", draw_rect);

  if (m_display_texture)
    m_display_texture->MakeReadyForSampling();

  // Internal post-processing.
  GPUTexture* display_texture = m_display_texture;
  s32 display_texture_view_x = m_display_texture_view_x;
  s32 display_texture_view_y = m_display_texture_view_y;
  s32 display_texture_view_width = m_display_texture_view_width;
  s32 display_texture_view_height = m_display_texture_view_height;
  if (postfx && display_texture && PostProcessing::InternalChain.IsActive() &&
      PostProcessing::InternalChain.CheckTargets(DISPLAY_INTERNAL_POSTFX_FORMAT, display_texture_view_width,
                                                 display_texture_view_height))
  {
    DebugAssert(display_texture_view_x == 0 && display_texture_view_y == 0 &&
                static_cast<s32>(display_texture->GetWidth()) == display_texture_view_width &&
                static_cast<s32>(display_texture->GetHeight()) == display_texture_view_height);

    // Now we can apply the post chain.
    GPUTexture* post_output_texture = PostProcessing::InternalChain.GetOutputTexture();
    if (PostProcessing::InternalChain.Apply(display_texture, m_display_depth_buffer, post_output_texture,
                                            GSVector4i(0, 0, display_texture_view_width, display_texture_view_height),
                                            display_texture_view_width, display_texture_view_height, m_display_width,
                                            m_display_height))
    {
      display_texture_view_x = 0;
      display_texture_view_y = 0;
      display_texture = post_output_texture;
      display_texture->MakeReadyForSampling();
    }
  }

  const GPUTexture::Format hdformat = target ? target->GetFormat() : g_gpu_device->GetWindowFormat();
  const u32 target_width = target ? target->GetWidth() : g_gpu_device->GetWindowWidth();
  const u32 target_height = target ? target->GetHeight() : g_gpu_device->GetWindowHeight();
  const bool really_postfx =
    (postfx && PostProcessing::DisplayChain.IsActive() && !g_gpu_device->GetWindowInfo().IsSurfaceless() &&
     hdformat != GPUTexture::Format::Unknown && target_width > 0 && target_height > 0 &&
     PostProcessing::DisplayChain.CheckTargets(hdformat, target_width, target_height));
  const GSVector4i real_draw_rect =
    g_gpu_device->UsesLowerLeftOrigin() ? GPUDevice::FlipToLowerLeft(draw_rect, target_height) : draw_rect;
  if (really_postfx)
  {
    g_gpu_device->ClearRenderTarget(PostProcessing::DisplayChain.GetInputTexture(), 0);
    g_gpu_device->SetRenderTarget(PostProcessing::DisplayChain.GetInputTexture());
  }
  else
  {
    if (target)
      g_gpu_device->SetRenderTarget(target);
    else if (!g_gpu_device->BeginPresent(false))
      return false;
  }

  if (display_texture)
  {
    bool texture_filter_linear = false;

    struct Uniforms
    {
      float src_rect[4];
      float src_size[4];
      float clamp_rect[4];
      float params[4];
    } uniforms;
    std::memset(uniforms.params, 0, sizeof(uniforms.params));

    switch (g_settings.display_scaling)
    {
      case DisplayScalingMode::Nearest:
      case DisplayScalingMode::NearestInteger:
        break;

      case DisplayScalingMode::BilinearSmooth:
      case DisplayScalingMode::BlinearInteger:
        texture_filter_linear = true;
        break;

      case DisplayScalingMode::BilinearSharp:
      {
        texture_filter_linear = true;
        uniforms.params[0] = std::max(
          std::floor(static_cast<float>(draw_rect.width()) / static_cast<float>(m_display_texture_view_width)), 1.0f);
        uniforms.params[1] = std::max(
          std::floor(static_cast<float>(draw_rect.height()) / static_cast<float>(m_display_texture_view_height)), 1.0f);
        uniforms.params[2] = 0.5f - 0.5f / uniforms.params[0];
        uniforms.params[3] = 0.5f - 0.5f / uniforms.params[1];
      }
      break;

      default:
        UnreachableCode();
        break;
    }

    g_gpu_device->SetPipeline(m_display_pipeline.get());
    g_gpu_device->SetTextureSampler(
      0, display_texture, texture_filter_linear ? g_gpu_device->GetLinearSampler() : g_gpu_device->GetNearestSampler());

    // For bilinear, clamp to 0.5/SIZE-0.5 to avoid bleeding from the adjacent texels in VRAM. This is because
    // 1.0 in UV space is not the bottom-right texel, but a mix of the bottom-right and wrapped/next texel.
    const float rcp_width = 1.0f / static_cast<float>(display_texture->GetWidth());
    const float rcp_height = 1.0f / static_cast<float>(display_texture->GetHeight());
    uniforms.src_rect[0] = static_cast<float>(display_texture_view_x) * rcp_width;
    uniforms.src_rect[1] = static_cast<float>(display_texture_view_y) * rcp_height;
    uniforms.src_rect[2] = static_cast<float>(display_texture_view_width) * rcp_width;
    uniforms.src_rect[3] = static_cast<float>(display_texture_view_height) * rcp_height;
    uniforms.clamp_rect[0] = (static_cast<float>(display_texture_view_x) + 0.5f) * rcp_width;
    uniforms.clamp_rect[1] = (static_cast<float>(display_texture_view_y) + 0.5f) * rcp_height;
    uniforms.clamp_rect[2] =
      (static_cast<float>(display_texture_view_x + display_texture_view_width) - 0.5f) * rcp_width;
    uniforms.clamp_rect[3] =
      (static_cast<float>(display_texture_view_y + display_texture_view_height) - 0.5f) * rcp_height;
    uniforms.src_size[0] = static_cast<float>(display_texture->GetWidth());
    uniforms.src_size[1] = static_cast<float>(display_texture->GetHeight());
    uniforms.src_size[2] = rcp_width;
    uniforms.src_size[3] = rcp_height;
    g_gpu_device->PushUniformBuffer(&uniforms, sizeof(uniforms));

    g_gpu_device->SetViewportAndScissor(real_draw_rect);
    g_gpu_device->Draw(3, 0);
  }

  if (really_postfx)
  {
    DebugAssert(!g_settings.debugging.show_vram);

    // "original size" in postfx includes padding.
    const float upscale_x =
      m_display_texture ? static_cast<float>(m_display_texture_view_width) / static_cast<float>(m_display_vram_width) :
                          1.0f;
    const float upscale_y = m_display_texture ? static_cast<float>(m_display_texture_view_height) /
                                                  static_cast<float>(m_display_vram_height) :
                                                1.0f;
    const s32 orig_width = static_cast<s32>(std::ceil(static_cast<float>(m_display_width) * upscale_x));
    const s32 orig_height = static_cast<s32>(std::ceil(static_cast<float>(m_display_height) * upscale_y));

    return PostProcessing::DisplayChain.Apply(PostProcessing::DisplayChain.GetInputTexture(), nullptr, target,
                                              real_draw_rect, orig_width, orig_height, m_display_width,
                                              m_display_height);
  }
  else
    return true;
}

void GPUBackend::DestroyDeinterlaceTextures()
{
  for (std::unique_ptr<GPUTexture>& tex : m_deinterlace_buffers)
    g_gpu_device->RecycleTexture(std::move(tex));
  g_gpu_device->RecycleTexture(std::move(m_deinterlace_texture));
  m_current_deinterlace_buffer = 0;
}

bool GPUBackend::Deinterlace(u32 field, u32 line_skip)
{
  GPUTexture* src = m_display_texture;
  const u32 x = m_display_texture_view_x;
  const u32 y = m_display_texture_view_y;
  const u32 width = m_display_texture_view_width;
  const u32 height = m_display_texture_view_height;

  switch (g_settings.display_deinterlacing_mode)
  {
    case DisplayDeinterlacingMode::Disabled:
    {
      if (line_skip == 0)
        return true;

      // Still have to extract the field.
      if (!DeinterlaceExtractField(0, src, x, y, width, height, line_skip)) [[unlikely]]
        return false;

      SetDisplayTexture(m_deinterlace_buffers[0].get(), m_display_depth_buffer, 0, 0, width, height);
      return true;
    }

    case DisplayDeinterlacingMode::Weave:
    {
      GL_SCOPE_FMT("DeinterlaceWeave({{{},{}}}, {}x{}, field={}, line_skip={})", x, y, width, height, field, line_skip);

      const u32 full_height = height * 2;
      if (!DeinterlaceSetTargetSize(width, full_height, true)) [[unlikely]]
      {
        ClearDisplayTexture();
        return false;
      }

      src->MakeReadyForSampling();

      g_gpu_device->SetRenderTarget(m_deinterlace_texture.get());
      g_gpu_device->SetPipeline(m_deinterlace_pipeline.get());
      g_gpu_device->SetTextureSampler(0, src, g_gpu_device->GetNearestSampler());
      const u32 uniforms[] = {x, y, field, line_skip};
      g_gpu_device->PushUniformBuffer(uniforms, sizeof(uniforms));
      g_gpu_device->SetViewportAndScissor(0, 0, width, full_height);
      g_gpu_device->Draw(3, 0);

      m_deinterlace_texture->MakeReadyForSampling();
      SetDisplayTexture(m_deinterlace_texture.get(), m_display_depth_buffer, 0, 0, width, full_height);
      return true;
    }

    case DisplayDeinterlacingMode::Blend:
    {
      constexpr u32 NUM_BLEND_BUFFERS = 2;

      GL_SCOPE_FMT("DeinterlaceBlend({{{},{}}}, {}x{}, field={}, line_skip={})", x, y, width, height, field, line_skip);

      const u32 this_buffer = m_current_deinterlace_buffer;
      m_current_deinterlace_buffer = (m_current_deinterlace_buffer + 1u) % NUM_BLEND_BUFFERS;
      GL_INS_FMT("Current buffer: {}", this_buffer);
      if (!DeinterlaceExtractField(this_buffer, src, x, y, width, height, line_skip) ||
          !DeinterlaceSetTargetSize(width, height, false)) [[unlikely]]
      {
        ClearDisplayTexture();
        return false;
      }

      // TODO: could be implemented with alpha blending instead..

      g_gpu_device->InvalidateRenderTarget(m_deinterlace_texture.get());
      g_gpu_device->SetRenderTarget(m_deinterlace_texture.get());
      g_gpu_device->SetPipeline(m_deinterlace_pipeline.get());
      g_gpu_device->SetTextureSampler(0, m_deinterlace_buffers[this_buffer].get(), g_gpu_device->GetNearestSampler());
      g_gpu_device->SetTextureSampler(1, m_deinterlace_buffers[(this_buffer - 1) % NUM_BLEND_BUFFERS].get(),
                                      g_gpu_device->GetNearestSampler());
      g_gpu_device->SetViewportAndScissor(0, 0, width, height);
      g_gpu_device->Draw(3, 0);

      m_deinterlace_texture->MakeReadyForSampling();
      SetDisplayTexture(m_deinterlace_texture.get(), m_display_depth_buffer, 0, 0, width, height);
      return true;
    }

    case DisplayDeinterlacingMode::Adaptive:
    {
      GL_SCOPE_FMT("DeinterlaceAdaptive({{{},{}}}, {}x{}, field={}, line_skip={})", x, y, width, height, field,
                   line_skip);

      const u32 full_height = height * 2;
      const u32 this_buffer = m_current_deinterlace_buffer;
      m_current_deinterlace_buffer = (m_current_deinterlace_buffer + 1u) % DEINTERLACE_BUFFER_COUNT;
      GL_INS_FMT("Current buffer: {}", this_buffer);
      if (!DeinterlaceExtractField(this_buffer, src, x, y, width, height, line_skip) ||
          !DeinterlaceSetTargetSize(width, full_height, false)) [[unlikely]]
      {
        ClearDisplayTexture();
        return false;
      }

      g_gpu_device->SetRenderTarget(m_deinterlace_texture.get());
      g_gpu_device->SetPipeline(m_deinterlace_pipeline.get());
      g_gpu_device->SetTextureSampler(0, m_deinterlace_buffers[this_buffer].get(), g_gpu_device->GetNearestSampler());
      g_gpu_device->SetTextureSampler(1, m_deinterlace_buffers[(this_buffer - 1) % DEINTERLACE_BUFFER_COUNT].get(),
                                      g_gpu_device->GetNearestSampler());
      g_gpu_device->SetTextureSampler(2, m_deinterlace_buffers[(this_buffer - 2) % DEINTERLACE_BUFFER_COUNT].get(),
                                      g_gpu_device->GetNearestSampler());
      g_gpu_device->SetTextureSampler(3, m_deinterlace_buffers[(this_buffer - 3) % DEINTERLACE_BUFFER_COUNT].get(),
                                      g_gpu_device->GetNearestSampler());
      const u32 uniforms[] = {field, full_height};
      g_gpu_device->PushUniformBuffer(uniforms, sizeof(uniforms));
      g_gpu_device->SetViewportAndScissor(0, 0, width, full_height);
      g_gpu_device->Draw(3, 0);

      m_deinterlace_texture->MakeReadyForSampling();
      SetDisplayTexture(m_deinterlace_texture.get(), m_display_depth_buffer, 0, 0, width, full_height);
      return true;
    }

    default:
      UnreachableCode();
  }
}

bool GPUBackend::DeinterlaceExtractField(u32 dst_bufidx, GPUTexture* src, u32 x, u32 y, u32 width, u32 height,
                                         u32 line_skip)
{
  if (!m_deinterlace_buffers[dst_bufidx] || m_deinterlace_buffers[dst_bufidx]->GetWidth() != width ||
      m_deinterlace_buffers[dst_bufidx]->GetHeight() != height)
  {
    if (!g_gpu_device->ResizeTexture(&m_deinterlace_buffers[dst_bufidx], width, height, GPUTexture::Type::RenderTarget,
                                     GPUTexture::Format::RGBA8, false)) [[unlikely]]
    {
      return false;
    }

    GL_OBJECT_NAME_FMT(m_deinterlace_buffers[dst_bufidx], "Blend Deinterlace Buffer {}", dst_bufidx);
  }

  GPUTexture* dst = m_deinterlace_buffers[dst_bufidx].get();
  g_gpu_device->InvalidateRenderTarget(dst);

  // If we're not skipping lines, then we can simply copy the texture.
  if (line_skip == 0 && src->GetFormat() == dst->GetFormat())
  {
    GL_INS_FMT("DeinterlaceExtractField({{{},{}}} {}x{} line_skip={}) => copy direct", x, y, width, height, line_skip);
    g_gpu_device->CopyTextureRegion(dst, 0, 0, 0, 0, src, x, y, 0, 0, width, height);
  }
  else
  {
    GL_SCOPE_FMT("DeinterlaceExtractField({{{},{}}} {}x{} line_skip={}) => shader copy", x, y, width, height,
                 line_skip);

    // Otherwise, we need to extract every other line from the texture.
    src->MakeReadyForSampling();
    g_gpu_device->SetRenderTarget(dst);
    g_gpu_device->SetPipeline(m_deinterlace_extract_pipeline.get());
    g_gpu_device->SetTextureSampler(0, src, g_gpu_device->GetNearestSampler());
    const u32 uniforms[] = {x, y, line_skip};
    g_gpu_device->PushUniformBuffer(uniforms, sizeof(uniforms));
    g_gpu_device->SetViewportAndScissor(0, 0, width, height);
    g_gpu_device->Draw(3, 0);

    GL_POP();
  }

  dst->MakeReadyForSampling();
  return true;
}

bool GPUBackend::DeinterlaceSetTargetSize(u32 width, u32 height, bool preserve)
{
  if (!m_deinterlace_texture || m_deinterlace_texture->GetWidth() != width ||
      m_deinterlace_texture->GetHeight() != height)
  {
    if (!g_gpu_device->ResizeTexture(&m_deinterlace_texture, width, height, GPUTexture::Type::RenderTarget,
                                     GPUTexture::Format::RGBA8, preserve)) [[unlikely]]
    {
      return false;
    }

    GL_OBJECT_NAME(m_deinterlace_texture, "Deinterlace target texture");
  }

  return true;
}

bool GPUBackend::ApplyChromaSmoothing()
{
  const u32 x = m_display_texture_view_x;
  const u32 y = m_display_texture_view_y;
  const u32 width = m_display_texture_view_width;
  const u32 height = m_display_texture_view_height;
  if (!m_chroma_smoothing_texture || m_chroma_smoothing_texture->GetWidth() != width ||
      m_chroma_smoothing_texture->GetHeight() != height)
  {
    if (!g_gpu_device->ResizeTexture(&m_chroma_smoothing_texture, width, height, GPUTexture::Type::RenderTarget,
                                     GPUTexture::Format::RGBA8, false))
    {
      ClearDisplayTexture();
      return false;
    }

    GL_OBJECT_NAME(m_chroma_smoothing_texture, "Chroma smoothing texture");
  }

  GL_SCOPE_FMT("ApplyChromaSmoothing({{{},{}}}, {}x{})", x, y, width, height);

  m_display_texture->MakeReadyForSampling();
  g_gpu_device->InvalidateRenderTarget(m_chroma_smoothing_texture.get());
  g_gpu_device->SetRenderTarget(m_chroma_smoothing_texture.get());
  g_gpu_device->SetPipeline(m_chroma_smoothing_pipeline.get());
  g_gpu_device->SetTextureSampler(0, m_display_texture, g_gpu_device->GetNearestSampler());
  const u32 uniforms[] = {x, y, width - 1, height - 1};
  g_gpu_device->PushUniformBuffer(uniforms, sizeof(uniforms));
  g_gpu_device->SetViewportAndScissor(0, 0, width, height);
  g_gpu_device->Draw(3, 0);

  m_chroma_smoothing_texture->MakeReadyForSampling();
  SetDisplayTexture(m_chroma_smoothing_texture.get(), m_display_depth_buffer, 0, 0, width, height);
  return true;
}

void GPUBackend::UpdateCLUT(GPUTexturePaletteReg reg, bool clut_is_8bit)
{
}

GSVector4i GPUBackend::CalculateDrawRect(s32 window_width, s32 window_height,
                                         bool apply_aspect_ratio /* = true */) const
{
  const bool integer_scale = (g_gpu_settings.display_scaling == DisplayScalingMode::NearestInteger ||
                              g_gpu_settings.display_scaling == DisplayScalingMode::BlinearInteger);
  const bool show_vram = g_gpu_settings.debugging.show_vram;
  const float display_aspect_ratio = m_display_aspect_ratio;
  const float window_ratio = static_cast<float>(window_width) / static_cast<float>(window_height);
  const float crtc_display_width = static_cast<float>(show_vram ? VRAM_WIDTH : m_display_width);
  const float crtc_display_height = static_cast<float>(show_vram ? VRAM_HEIGHT : m_display_height);
  const float x_scale =
    apply_aspect_ratio ?
      (display_aspect_ratio / (static_cast<float>(crtc_display_width) / static_cast<float>(crtc_display_height))) :
      1.0f;
  float display_width = crtc_display_width;
  float display_height = crtc_display_height;
  float active_left = static_cast<float>(show_vram ? 0 : m_display_origin_left);
  float active_top = static_cast<float>(show_vram ? 0 : m_display_origin_top);
  float active_width = static_cast<float>(show_vram ? VRAM_WIDTH : m_display_vram_width);
  float active_height = static_cast<float>(show_vram ? VRAM_HEIGHT : m_display_vram_height);
  if (!g_gpu_settings.display_stretch_vertically)
  {
    display_width *= x_scale;
    active_left *= x_scale;
    active_width *= x_scale;
  }
  else
  {
    display_height /= x_scale;
    active_top /= x_scale;
    active_height /= x_scale;
  }

  // now fit it within the window
  float scale;
  float left_padding, top_padding;
  if ((display_width / display_height) >= window_ratio)
  {
    // align in middle vertically
    scale = static_cast<float>(window_width) / display_width;
    if (integer_scale)
    {
      scale = std::max(std::floor(scale), 1.0f);
      left_padding = std::max<float>((static_cast<float>(window_width) - display_width * scale) / 2.0f, 0.0f);
    }
    else
    {
      left_padding = 0.0f;
    }

    switch (g_gpu_settings.display_alignment)
    {
      case DisplayAlignment::RightOrBottom:
        top_padding = std::max<float>(static_cast<float>(window_height) - (display_height * scale), 0.0f);
        break;

      case DisplayAlignment::Center:
        top_padding = std::max<float>((static_cast<float>(window_height) - (display_height * scale)) / 2.0f, 0.0f);
        break;

      case DisplayAlignment::LeftOrTop:
      default:
        top_padding = 0.0f;
        break;
    }
  }
  else
  {
    // align in middle horizontally
    scale = static_cast<float>(window_height) / display_height;
    if (integer_scale)
    {
      scale = std::max(std::floor(scale), 1.0f);
      top_padding = std::max<float>((static_cast<float>(window_height) - (display_height * scale)) / 2.0f, 0.0f);
    }
    else
    {
      top_padding = 0.0f;
    }

    switch (g_gpu_settings.display_alignment)
    {
      case DisplayAlignment::RightOrBottom:
        left_padding = std::max<float>(static_cast<float>(window_width) - (display_width * scale), 0.0f);
        break;

      case DisplayAlignment::Center:
        left_padding = std::max<float>((static_cast<float>(window_width) - (display_width * scale)) / 2.0f, 0.0f);
        break;

      case DisplayAlignment::LeftOrTop:
      default:
        left_padding = 0.0f;
        break;
    }
  }

  // TODO: This should be a float rectangle. But because GL is lame, it only has integer viewports...
  const s32 left = static_cast<s32>(active_left * scale + left_padding);
  const s32 top = static_cast<s32>(active_top * scale + top_padding);
  const s32 right = left + static_cast<s32>(active_width * scale);
  const s32 bottom = top + static_cast<s32>(active_height * scale);
  return GSVector4i(left, top, right, bottom);
}

bool CompressAndWriteTextureToFile(u32 width, u32 height, std::string filename, FileSystem::ManagedCFilePtr fp,
                                   u8 quality, bool clear_alpha, bool flip_y, std::vector<u32> texture_data,
                                   u32 texture_data_stride, GPUTexture::Format texture_format, bool display_osd_message,
                                   bool use_thread)
{
  std::string osd_key;
  if (display_osd_message)
  {
    // Use a 60 second timeout to give it plenty of time to actually save.
    osd_key = fmt::format("ScreenshotSaver_{}", filename);
    Host::AddIconOSDMessage(osd_key, ICON_FA_CAMERA,
                            fmt::format(TRANSLATE_FS("GPU", "Saving screenshot to '{}'."), Path::GetFileName(filename)),
                            60.0f);
  }

  static constexpr auto proc = [](u32 width, u32 height, std::string filename, FileSystem::ManagedCFilePtr fp,
                                  u8 quality, bool clear_alpha, bool flip_y, std::vector<u32> texture_data,
                                  u32 texture_data_stride, GPUTexture::Format texture_format, std::string osd_key,
                                  bool use_thread) {
    bool result;

    const char* extension = std::strrchr(filename.c_str(), '.');
    if (extension)
    {
      if (GPUTexture::ConvertTextureDataToRGBA8(width, height, texture_data, texture_data_stride, texture_format))
      {
        if (clear_alpha)
        {
          for (u32& pixel : texture_data)
            pixel |= 0xFF000000u;
        }

        if (flip_y)
          GPUTexture::FlipTextureDataRGBA8(width, height, reinterpret_cast<u8*>(texture_data.data()),
                                           texture_data_stride);

        Assert(texture_data_stride == sizeof(u32) * width);
        RGBA8Image image(width, height, std::move(texture_data));
        if (image.SaveToFile(filename.c_str(), fp.get(), quality))
        {
          result = true;
        }
        else
        {
          ERROR_LOG("Unknown extension in filename '{}' or save error: '{}'", filename, extension);
          result = false;
        }
      }
      else
      {
        result = false;
      }
    }
    else
    {
      ERROR_LOG("Unable to determine file extension for '{}'", filename);
      result = false;
    }

    if (!osd_key.empty())
    {
      Host::AddIconOSDMessage(std::move(osd_key), ICON_FA_CAMERA,
                              fmt::format(result ? TRANSLATE_FS("GPU", "Saved screenshot to '{}'.") :
                                                   TRANSLATE_FS("GPU", "Failed to save screenshot to '{}'."),
                                          Path::GetFileName(filename),
                                          result ? Host::OSD_INFO_DURATION : Host::OSD_ERROR_DURATION));
    }

    if (use_thread)
    {
      // remove ourselves from the list, if the GS thread is waiting for us, we won't be in there
      const auto this_id = std::this_thread::get_id();
      std::unique_lock lock(s_screenshot_threads_mutex);
      for (auto it = s_screenshot_threads.begin(); it != s_screenshot_threads.end(); ++it)
      {
        if (it->get_id() == this_id)
        {
          it->detach();
          s_screenshot_threads.erase(it);
          break;
        }
      }
    }

    return result;
  };

  if (!use_thread)
  {
    return proc(width, height, std::move(filename), std::move(fp), quality, clear_alpha, flip_y,
                std::move(texture_data), texture_data_stride, texture_format, std::move(osd_key), use_thread);
  }

  std::thread thread(proc, width, height, std::move(filename), std::move(fp), quality, clear_alpha, flip_y,
                     std::move(texture_data), texture_data_stride, texture_format, std::move(osd_key), use_thread);
  std::unique_lock lock(s_screenshot_threads_mutex);
  s_screenshot_threads.push_back(std::move(thread));
  return true;
}

void JoinScreenshotThreads()
{
  std::unique_lock lock(s_screenshot_threads_mutex);
  while (!s_screenshot_threads.empty())
  {
    std::thread save_thread(std::move(s_screenshot_threads.front()));
    s_screenshot_threads.pop_front();
    lock.unlock();
    save_thread.join();
    lock.lock();
  }
}

bool GPUBackend::WriteDisplayTextureToFile(std::string filename, bool compress_on_thread /* = false */)
{
  if (!m_display_texture)
    return false;

  const u32 read_x = static_cast<u32>(m_display_texture_view_x);
  const u32 read_y = static_cast<u32>(m_display_texture_view_y);
  const u32 read_width = static_cast<u32>(m_display_texture_view_width);
  const u32 read_height = static_cast<u32>(m_display_texture_view_height);

  const u32 texture_data_stride =
    Common::AlignUpPow2(GPUTexture::GetPixelSize(m_display_texture->GetFormat()) * read_width, 4);
  std::vector<u32> texture_data((texture_data_stride * read_height) / sizeof(u32));

  std::unique_ptr<GPUDownloadTexture> dltex;
  if (g_gpu_device->GetFeatures().memory_import)
  {
    dltex =
      g_gpu_device->CreateDownloadTexture(read_width, read_height, m_display_texture->GetFormat(), texture_data.data(),
                                          texture_data.size() * sizeof(u32), texture_data_stride);
  }
  if (!dltex)
  {
    if (!(dltex = g_gpu_device->CreateDownloadTexture(read_width, read_height, m_display_texture->GetFormat())))
    {
      ERROR_LOG("Failed to create {}x{} {} download texture", read_width, read_height,
                GPUTexture::GetFormatName(m_display_texture->GetFormat()));
      return false;
    }
  }

  dltex->CopyFromTexture(0, 0, m_display_texture, read_x, read_y, read_width, read_height, 0, 0, !dltex->IsImported());
  if (!dltex->ReadTexels(0, 0, read_width, read_height, texture_data.data(), texture_data_stride))
  {
    RestoreDeviceContext();
    return false;
  }

  RestoreDeviceContext();

  Error error;
  auto fp = FileSystem::OpenManagedCFile(filename.c_str(), "wb", &error);
  if (!fp)
  {
    ERROR_LOG("Can't open file '{}': {}", Path::GetFileName(filename), error.GetDescription());
    return false;
  }

  constexpr bool clear_alpha = true;
  const bool flip_y = g_gpu_device->UsesLowerLeftOrigin();

  return CompressAndWriteTextureToFile(
    read_width, read_height, std::move(filename), std::move(fp), g_settings.display_screenshot_quality, clear_alpha,
    flip_y, std::move(texture_data), texture_data_stride, m_display_texture->GetFormat(), false, compress_on_thread);
}

void GPUBackend::HandleRenderScreenshotToBuffer(const GPUThreadRenderScreenshotToBufferCommand* cmd)
{
  const u32 width = cmd->width;
  const u32 height = cmd->height;
  const GSVector4i draw_rect = GSVector4i::load<false>(cmd->draw_rect);
  const GPUTexture::Format hdformat =
    g_gpu_device->HasSurface() ? g_gpu_device->GetWindowFormat() : GPUTexture::Format::RGBA8;

  auto render_texture =
    g_gpu_device->FetchAutoRecycleTexture(cmd->width, cmd->height, 1, 1, 1, GPUTexture::Type::RenderTarget, hdformat);
  if (!render_texture)
  {
    *cmd->out_result = false;
    return;
  }

  g_gpu_device->ClearRenderTarget(render_texture.get(), 0);

  // TODO: this should use copy shader instead.
  RenderDisplay(render_texture.get(), draw_rect, cmd->postfx);

  const u32 stride = Common::AlignUpPow2(GPUTexture::GetPixelSize(hdformat) * width, sizeof(u32));
  cmd->out_pixels->resize((height * stride) / sizeof(u32));

  std::unique_ptr<GPUDownloadTexture> dltex;
  if (g_gpu_device->GetFeatures().memory_import)
  {
    dltex = g_gpu_device->CreateDownloadTexture(width, height, hdformat, cmd->out_pixels->data(),
                                                cmd->out_pixels->size() * sizeof(u32), stride);
  }
  if (!dltex)
  {
    if (!(dltex = g_gpu_device->CreateDownloadTexture(width, height, hdformat)))
    {
      ERROR_LOG("Failed to create {}x{} download texture", width, height);
      *cmd->out_result = false;
      return;
    }
  }

  dltex->CopyFromTexture(0, 0, render_texture.get(), 0, 0, width, height, 0, 0, false);
  if (!dltex->ReadTexels(0, 0, width, height, cmd->out_pixels->data(), stride))
  {
    RestoreDeviceContext();
    *cmd->out_result = false;
    return;
  }

  *cmd->out_stride = stride;
  *cmd->out_format = hdformat;
  *cmd->out_result = true;
  RestoreDeviceContext();
}

bool GPUBackend::RenderScreenshotToFile(std::string filename, DisplayScreenshotMode mode, u8 quality,
                                        bool compress_on_thread, bool show_osd_message)
{
  u32 width = g_gpu_device->GetWindowWidth();
  u32 height = g_gpu_device->GetWindowHeight();
  GSVector4i draw_rect = CalculateDrawRect(width, height, true);

  const bool internal_resolution = (mode != DisplayScreenshotMode::ScreenResolution || g_settings.debugging.show_vram);
  if (internal_resolution && m_display_texture_view_width != 0 && m_display_texture_view_height != 0)
  {
    if (mode == DisplayScreenshotMode::InternalResolution)
    {
      const u32 draw_width = static_cast<u32>(draw_rect.width());
      const u32 draw_height = static_cast<u32>(draw_rect.height());

      // If internal res, scale the computed draw rectangle to the internal res.
      // We re-use the draw rect because it's already been AR corrected.
      const float sar =
        static_cast<float>(m_display_texture_view_width) / static_cast<float>(m_display_texture_view_height);
      const float dar = static_cast<float>(draw_width) / static_cast<float>(draw_height);
      if (sar >= dar)
      {
        // stretch height, preserve width
        const float scale = static_cast<float>(m_display_texture_view_width) / static_cast<float>(draw_width);
        width = m_display_texture_view_width;
        height = static_cast<u32>(std::round(static_cast<float>(draw_height) * scale));
      }
      else
      {
        // stretch width, preserve height
        const float scale = static_cast<float>(m_display_texture_view_height) / static_cast<float>(draw_height);
        width = static_cast<u32>(std::round(static_cast<float>(draw_width) * scale));
        height = m_display_texture_view_height;
      }

      // DX11 won't go past 16K texture size.
      const u32 max_texture_size = g_gpu_device->GetMaxTextureSize();
      if (width > max_texture_size)
      {
        height = static_cast<u32>(static_cast<float>(height) /
                                  (static_cast<float>(width) / static_cast<float>(max_texture_size)));
        width = max_texture_size;
      }
      if (height > max_texture_size)
      {
        height = max_texture_size;
        width = static_cast<u32>(static_cast<float>(width) /
                                 (static_cast<float>(height) / static_cast<float>(max_texture_size)));
      }
    }
    else // if (mode == DisplayScreenshotMode::UncorrectedInternalResolution)
    {
      width = m_display_texture_view_width;
      height = m_display_texture_view_height;
    }

    // Remove padding, it's not part of the framebuffer.
    draw_rect = GSVector4i(0, 0, static_cast<s32>(width), static_cast<s32>(height));
  }
  if (width == 0 || height == 0)
    return false;

  std::vector<u32> pixels;
  u32 pixels_stride;
  GPUTexture::Format pixels_format;
  if (!RenderScreenshotToBuffer(width, height, draw_rect, !internal_resolution, &pixels, &pixels_stride,
                                &pixels_format))
  {
    ERROR_LOG("Failed to render {}x{} screenshot", width, height);
    return false;
  }

  Error error;
  auto fp = FileSystem::OpenManagedCFile(filename.c_str(), "wb", &error);
  if (!fp)
  {
    ERROR_LOG("Can't open file '{}': {}", Path::GetFileName(filename), error.GetDescription());
    return false;
  }

  return CompressAndWriteTextureToFile(width, height, std::move(filename), std::move(fp), quality, true,
                                       g_gpu_device->UsesLowerLeftOrigin(), std::move(pixels), pixels_stride,
                                       pixels_format, show_osd_message, compress_on_thread);
}

void GPUBackend::GetStatsString(SmallStringBase& str)
{
  if (IsUsingHardwareBackend())
  {
    str.format("{} HW | {} P | {} DC | {} B | {} RP | {} RB | {} C | {} W",
               GPUDevice::RenderAPIToString(g_gpu_device->GetRenderAPI()), s_stats.num_primitives,
               s_stats.host_num_draws, s_stats.host_num_barriers, s_stats.host_num_render_passes,
               s_stats.host_num_downloads, s_stats.num_copies, s_stats.num_writes);
  }
  else
  {
    str.format("{} SW | {} P | {} R | {} C | {} W", GPUDevice::RenderAPIToString(g_gpu_device->GetRenderAPI()),
               s_stats.num_primitives, s_stats.num_reads, s_stats.num_copies, s_stats.num_writes);
  }
}

void GPUBackend::GetMemoryStatsString(SmallStringBase& str)
{
  const u32 vram_usage_mb = static_cast<u32>((g_gpu_device->GetVRAMUsage() + (1048576 - 1)) / 1048576);
  const u32 stream_kb = static_cast<u32>((s_stats.host_buffer_streamed + (1024 - 1)) / 1024);

  str.format("{} MB VRAM | {} KB STR | {} TC | {} TU", vram_usage_mb, stream_kb, s_stats.host_num_copies,
             s_stats.host_num_uploads);
}

void GPUBackend::ResetStatistics()
{
  s_counters = {};
  g_gpu_device->ResetStatistics();
}

void GPUBackend::UpdateStatistics(u32 frame_count)
{
  const GPUDevice::Statistics& stats = g_gpu_device->GetStatistics();
  const u32 round = (frame_count - 1);

#define UPDATE_COUNTER(x) s_stats.x = (s_counters.x + round) / frame_count
#define UPDATE_GPU_STAT(x) s_stats.host_##x = (stats.x + round) / frame_count

  UPDATE_COUNTER(num_reads);
  UPDATE_COUNTER(num_writes);
  UPDATE_COUNTER(num_copies);
  UPDATE_COUNTER(num_vertices);
  UPDATE_COUNTER(num_primitives);

  // UPDATE_COUNTER(num_read_texture_updates);
  // UPDATE_COUNTER(num_ubo_updates);

  UPDATE_GPU_STAT(buffer_streamed);
  UPDATE_GPU_STAT(num_draws);
  UPDATE_GPU_STAT(num_barriers);
  UPDATE_GPU_STAT(num_render_passes);
  UPDATE_GPU_STAT(num_copies);
  UPDATE_GPU_STAT(num_downloads);
  UPDATE_GPU_STAT(num_uploads);

#undef UPDATE_GPU_STAT
#undef UPDATE_COUNTER

  ResetStatistics();
}
