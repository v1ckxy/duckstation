// SPDX-FileCopyrightText: 2019-2024 Connor McLaughlin <stenzek@gmail.com>
// SPDX-License-Identifier: (GPL-3.0 OR CC-BY-NC-ND-4.0)

#pragma once

#include "gpu.h"
#include "gpu_backend.h"

#include "util/gpu_device.h"

#include "common/heap_array.h"

#include <array>
#include <memory>
#include <vector>

// TODO: Move to cpp
// TODO: Rename to GPUSWBackend, preserved to avoid conflicts.
class GPU_SW final : public GPUBackend
{
public:
  GPU_SW();
  ~GPU_SW() override;

  bool Initialize(bool clear_vram, Error* error) override;

  std::tuple<u32, u32> GetEffectiveDisplayResolution(bool scaled = true) const override;
  std::tuple<u32, u32> GetFullDisplayResolution(bool scaled = true) const override;

  void UpdateResolutionScale() override;

  void FlushRender() override;

  void RestoreDeviceContext() override;

  bool DoState(GPUTexture** host_texture, bool is_reading, bool update_display) override;
  void ClearVRAM() override;

  void ReadVRAM(u32 x, u32 y, u32 width, u32 height) override;

  void DrawPolygon(const GPUBackendDrawPolygonCommand* cmd) override;
  void DrawPrecisePolygon(const GPUBackendDrawPrecisePolygonCommand* cmd) override;
  void DrawLine(const GPUBackendDrawLineCommand* cmd) override;
  void DrawSprite(const GPUBackendDrawRectangleCommand * cmd) override;
  void DrawingAreaChanged(const GPUDrawingArea& new_drawing_area, const GSVector4i clamped_drawing_area) override;
  void ClearCache() override;
  void UpdateCLUT(GPUTexturePaletteReg reg, bool clut_is_8bit) override;
  void OnBufferSwapped() override;

  void UpdateDisplay(const GPUBackendUpdateDisplayCommand* cmd) override;

private:
  template<GPUTexture::Format display_format>
  bool CopyOut15Bit(u32 src_x, u32 src_y, u32 width, u32 height, u32 line_skip);

  template<GPUTexture::Format display_format>
  bool CopyOut24Bit(u32 src_x, u32 src_y, u32 skip_x, u32 width, u32 height, u32 line_skip);

  bool CopyOut(u32 src_x, u32 src_y, u32 skip_x, u32 width, u32 height, u32 line_skip, bool is_24bit);

  void SetDisplayTextureFormat();
  GPUTexture* GetDisplayTexture(u32 width, u32 height, GPUTexture::Format format);

  FixedHeapArray<u8, GPU_MAX_DISPLAY_WIDTH * GPU_MAX_DISPLAY_HEIGHT * sizeof(u32)> m_upload_buffer;
  GPUTexture::Format m_16bit_display_format = GPUTexture::Format::RGB565;
  GPUTexture::Format m_24bit_display_format = GPUTexture::Format::RGBA8;
  std::unique_ptr<GPUTexture> m_upload_texture;
};
