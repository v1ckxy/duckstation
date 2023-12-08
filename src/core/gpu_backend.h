// SPDX-FileCopyrightText: 2019-2024 Connor McLaughlin <stenzek@gmail.com>
// SPDX-License-Identifier: (GPL-3.0 OR CC-BY-NC-ND-4.0)

#pragma once

#include "gpu_types.h"

#include "util/gpu_texture.h"

#include <tuple>

class Error;
class SmallStringBase;

class GPUFramebuffer;
class GPUPipeline;

struct Settings;
class StateWrapper;

// DESIGN NOTE: Only static methods should be called on the CPU thread.
// You specifically don't have a global pointer available for this reason.

class GPUBackend
{
public:
  static GPUThreadCommand* NewClearVRAMCommand();
  static GPUBackendDoStateCommand* NewDoStateCommand();
  static GPUThreadCommand* NewClearDisplayCommand();
  static GPUBackendUpdateDisplayCommand* NewUpdateDisplayCommand();
  static GPUThreadCommand* NewClearCacheCommand();
  static GPUThreadCommand* NewBufferSwappedCommand();
  static GPUThreadCommand* NewFlushRenderCommand();
  static GPUThreadCommand* NewUpdateResolutionScaleCommand();
  static GPUBackendReadVRAMCommand* NewReadVRAMCommand();
  static GPUBackendFillVRAMCommand* NewFillVRAMCommand();
  static GPUBackendUpdateVRAMCommand* NewUpdateVRAMCommand(u32 num_words);
  static GPUBackendCopyVRAMCommand* NewCopyVRAMCommand();
  static GPUBackendSetDrawingAreaCommand* NewSetDrawingAreaCommand();
  static GPUBackendUpdateCLUTCommand* NewUpdateCLUTCommand();
  static GPUBackendDrawPolygonCommand* NewDrawPolygonCommand(u32 num_vertices);
  static GPUBackendDrawPrecisePolygonCommand* NewDrawPrecisePolygonCommand(u32 num_vertices);
  static GPUBackendDrawRectangleCommand* NewDrawRectangleCommand();
  static GPUBackendDrawLineCommand* NewDrawLineCommand(u32 num_vertices);
  static void PushCommand(GPUThreadCommand* cmd);
  static void PushCommandAndWakeThread(GPUThreadCommand* cmd);
  static void PushCommandAndSync(GPUThreadCommand* cmd, bool spin);

  static bool IsUsingHardwareBackend();

  static std::unique_ptr<GPUBackend> CreateHardwareBackend();
  static std::unique_ptr<GPUBackend> CreateSoftwareBackend();

  static bool BeginQueueFrame();
  static void WaitForOneQueuedFrame();

  static bool RenderScreenshotToBuffer(u32 width, u32 height, const GSVector4i draw_rect, bool postfx,
                                       std::vector<u32>* out_pixels, u32* out_stride, GPUTexture::Format* out_format);

  static std::tuple<u32, u32> GetLastDisplaySourceSize();

  static void GetStatsString(SmallStringBase& str);
  static void GetMemoryStatsString(SmallStringBase& str);
  static void ResetStatistics();
  static void UpdateStatistics(u32 frame_count);

public:
  GPUBackend();
  virtual ~GPUBackend();

  ALWAYS_INLINE const void* GetDisplayTextureHandle() const { return m_display_texture; }
  ALWAYS_INLINE s32 GetDisplayWidth() const { return m_display_width; }
  ALWAYS_INLINE s32 GetDisplayHeight() const { return m_display_height; }
  ALWAYS_INLINE s32 GetDisplayViewWidth() const { return m_display_texture_view_width; }
  ALWAYS_INLINE s32 GetDisplayViewHeight() const { return m_display_texture_view_height; }
  ALWAYS_INLINE float GetDisplayAspectRatio() const { return m_display_aspect_ratio; }
  ALWAYS_INLINE bool HasDisplayTexture() const { return static_cast<bool>(m_display_texture); }

  virtual bool Initialize(bool clear_vram, Error* error);

  virtual void ClearVRAM() = 0;
  virtual bool DoState(GPUTexture** host_texture, bool is_reading, bool update_display) = 0;

  virtual void ReadVRAM(u32 x, u32 y, u32 width, u32 height) = 0;
  virtual void FillVRAM(u32 x, u32 y, u32 width, u32 height, u32 color, GPUBackendCommandParameters params);
  virtual void UpdateVRAM(u32 x, u32 y, u32 width, u32 height, const void* data, GPUBackendCommandParameters params);
  virtual void CopyVRAM(u32 src_x, u32 src_y, u32 dst_x, u32 dst_y, u32 width, u32 height,
                        GPUBackendCommandParameters params);

  virtual void DrawPolygon(const GPUBackendDrawPolygonCommand* cmd) = 0;
  virtual void DrawPrecisePolygon(const GPUBackendDrawPrecisePolygonCommand* cmd) = 0;
  virtual void DrawSprite(const GPUBackendDrawRectangleCommand* cmd) = 0;
  virtual void DrawLine(const GPUBackendDrawLineCommand* cmd) = 0;

  virtual void DrawingAreaChanged(const GPUDrawingArea& new_drawing_area, const GSVector4i clamped_drawing_area) = 0;
  virtual void UpdateCLUT(GPUTexturePaletteReg reg, bool clut_is_8bit) = 0;
  virtual void ClearCache() = 0;
  virtual void OnBufferSwapped() = 0;

  virtual void UpdateDisplay(const GPUBackendUpdateDisplayCommand* cmd) = 0;

  virtual void UpdateSettings(const Settings& old_settings);

  /// Returns the effective display resolution of the GPU.
  virtual std::tuple<u32, u32> GetEffectiveDisplayResolution(bool scaled = true) const = 0;

  /// Returns the full display resolution of the GPU, including padding.
  virtual std::tuple<u32, u32> GetFullDisplayResolution(bool scaled = true) const = 0;

  /// TODO: Updates the resolution scale when it's set to automatic.
  virtual void UpdateResolutionScale() = 0;

  /// Ensures all pending draws are flushed to the host GPU.
  virtual void FlushRender() = 0;

  // Graphics API state reset/restore - call when drawing the UI etc.
  // TODO: replace with "invalidate cached state"
  virtual void RestoreDeviceContext() = 0;

  void HandleCommand(const GPUThreadCommand* cmd);

  /// Draws the current display texture, with any post-processing.
  bool PresentDisplay();

protected:
  enum : u32
  {
    DEINTERLACE_BUFFER_COUNT = 4,
  };

  /// Helper function for computing the draw rectangle in a larger window.
  GSVector4i CalculateDrawRect(s32 window_width, s32 window_height, bool apply_aspect_ratio = true) const;

  /// Helper function to save current display texture to PNG.
  bool WriteDisplayTextureToFile(std::string filename, bool compress_on_thread = false);

  /// Renders the display, optionally with postprocessing to the specified image.
  void HandleRenderScreenshotToBuffer(const GPUThreadRenderScreenshotToBufferCommand* cmd);

  /// Helper function to save screenshot to PNG.
  bool RenderScreenshotToFile(std::string filename, DisplayScreenshotMode mode, u8 quality, bool compress_on_thread,
                              bool show_osd_message);

  bool CompileDisplayPipelines(bool display, bool deinterlace, bool chroma_smoothing);

  void ClearDisplay();
  void ClearDisplayTexture();
  void SetDisplayTexture(GPUTexture* texture, GPUTexture* depth_texture, s32 view_x, s32 view_y, s32 view_width,
                         s32 view_height);

  bool RenderDisplay(GPUTexture* target, const GSVector4i draw_rect, bool postfx);

  bool Deinterlace(u32 field, u32 line_skip);
  bool DeinterlaceExtractField(u32 dst_bufidx, GPUTexture* src, u32 x, u32 y, u32 width, u32 height, u32 line_skip);
  bool DeinterlaceSetTargetSize(u32 width, u32 height, bool preserve);
  void DestroyDeinterlaceTextures();
  bool ApplyChromaSmoothing();

  s32 m_display_width = 0;
  s32 m_display_height = 0;
  s32 m_display_origin_left = 0;
  s32 m_display_origin_top = 0;
  s32 m_display_vram_width = 0;
  s32 m_display_vram_height = 0;
  float m_display_aspect_ratio = 1.0f;

  u32 m_current_deinterlace_buffer = 0;
  std::unique_ptr<GPUPipeline> m_deinterlace_pipeline;
  std::unique_ptr<GPUPipeline> m_deinterlace_extract_pipeline;
  std::array<std::unique_ptr<GPUTexture>, DEINTERLACE_BUFFER_COUNT> m_deinterlace_buffers;
  std::unique_ptr<GPUTexture> m_deinterlace_texture;

  std::unique_ptr<GPUPipeline> m_chroma_smoothing_pipeline;
  std::unique_ptr<GPUTexture> m_chroma_smoothing_texture;

  std::unique_ptr<GPUPipeline> m_display_pipeline;
  GPUTexture* m_display_texture = nullptr;
  GPUTexture* m_display_depth_buffer = nullptr;
  s32 m_display_texture_view_x = 0;
  s32 m_display_texture_view_y = 0;
  s32 m_display_texture_view_width = 0;
  s32 m_display_texture_view_height = 0;
};
