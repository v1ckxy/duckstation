// SPDX-FileCopyrightText: 2019-2024 Connor McLaughlin <stenzek@gmail.com>
// SPDX-License-Identifier: (GPL-3.0 OR CC-BY-NC-ND-4.0)

#include "gpu_sw.h"
#include "gpu.h"
#include "gpu_sw_rasterizer.h"
#include "system.h"

#include "util/gpu_device.h"

#include "common/align.h"
#include "common/assert.h"
#include "common/intrin.h"
#include "common/log.h"

#include <algorithm>

Log_SetChannel(GPU_SW);

GPU_SW::GPU_SW() = default;

GPU_SW::~GPU_SW() = default;

bool GPU_SW::Initialize(bool clear_vram, Error* error)
{
  if (!GPUBackend::Initialize(clear_vram, error))
    return false;

  GPU_SW_Rasterizer::SelectImplementation();

  // if we're using "new" vram, clear it out here
  if (clear_vram)
    std::memset(g_vram, 0, sizeof(g_vram));

  SetDisplayTextureFormat();
  return true;
}

bool GPU_SW::DoState(GPUTexture** host_texture, bool is_reading, bool update_display)
{
  // TODO: FIXME
  // ignore the host texture for software mode, since we want to save vram here
  return true;
}

void GPU_SW::ClearVRAM()
{
  std::memset(g_vram, 0, sizeof(g_vram));
  std::memset(g_gpu_clut, 0, sizeof(g_gpu_clut));
}

std::tuple<u32, u32> GPU_SW::GetEffectiveDisplayResolution(bool scaled /* = true */) const
{
  return std::tie(m_display_vram_width, m_display_vram_height);
}

std::tuple<u32, u32> GPU_SW::GetFullDisplayResolution(bool scaled /* = true */) const
{
  return std::tie(m_display_width, m_display_height);
}

void GPU_SW::UpdateResolutionScale()
{
}

void GPU_SW::ReadVRAM(u32 x, u32 y, u32 width, u32 height)
{
}

void GPU_SW::DrawPolygon(const GPUBackendDrawPolygonCommand* cmd)
{
  const GPURenderCommand rc{cmd->rc.bits};
  const bool dithering_enable = rc.IsDitheringEnabled() && cmd->draw_mode.dither_enable;

  const GPU_SW_Rasterizer::DrawTriangleFunction DrawFunction = GPU_SW_Rasterizer::GetDrawTriangleFunction(
    rc.shading_enable, rc.texture_enable, rc.raw_texture_enable, rc.transparency_enable, dithering_enable);

  DrawFunction(cmd, &cmd->vertices[0], &cmd->vertices[1], &cmd->vertices[2]);
  if (rc.quad_polygon)
    DrawFunction(cmd, &cmd->vertices[2], &cmd->vertices[1], &cmd->vertices[3]);
}

void GPU_SW::DrawPrecisePolygon(const GPUBackendDrawPrecisePolygonCommand* cmd)
{
  const GPURenderCommand rc{cmd->rc.bits};
  const bool dithering_enable = rc.IsDitheringEnabled() && cmd->draw_mode.dither_enable;

  const GPU_SW_Rasterizer::DrawTriangleFunction DrawFunction = GPU_SW_Rasterizer::GetDrawTriangleFunction(
    rc.shading_enable, rc.texture_enable, rc.raw_texture_enable, rc.transparency_enable, dithering_enable);

  // Need to cut out the irrelevant bits.
  // TODO: In _theory_ we could use the fixed-point parts here.
  GPUBackendDrawPolygonCommand::Vertex vertices[4];
  for (u32 i = 0; i < cmd->num_vertices; i++)
  {
    const GPUBackendDrawPrecisePolygonCommand::Vertex& src = cmd->vertices[i];
    GPUBackendDrawPolygonCommand::Vertex& dst = vertices[i];

    dst.x = src.native_x;
    dst.y = src.native_y;
    dst.color = src.color;
    dst.texcoord = src.texcoord;
  }

  DrawFunction(cmd, &vertices[0], &vertices[1], &vertices[2]);
  if (rc.quad_polygon)
    DrawFunction(cmd, &vertices[2], &vertices[1], &vertices[3]);
}

void GPU_SW::DrawSprite(const GPUBackendDrawRectangleCommand* cmd)
{
  const GPURenderCommand rc{cmd->rc.bits};

  const GPU_SW_Rasterizer::DrawRectangleFunction DrawFunction =
    GPU_SW_Rasterizer::GetDrawRectangleFunction(rc.texture_enable, rc.raw_texture_enable, rc.transparency_enable);

  DrawFunction(cmd);
}

void GPU_SW::DrawLine(const GPUBackendDrawLineCommand* cmd)
{
  const GPU_SW_Rasterizer::DrawLineFunction DrawFunction = GPU_SW_Rasterizer::GetDrawLineFunction(
    cmd->rc.shading_enable, cmd->rc.transparency_enable, cmd->IsDitheringEnabled());

  for (u16 i = 1; i < cmd->num_vertices; i++)
    DrawFunction(cmd, &cmd->vertices[i - 1], &cmd->vertices[i]);
}

void GPU_SW::DrawingAreaChanged(const GPUDrawingArea& new_drawing_area, const GSVector4i clamped_drawing_area)
{
  GPU_SW_Rasterizer::g_drawing_area = new_drawing_area;
}

void GPU_SW::ClearCache()
{
}

void GPU_SW::UpdateCLUT(GPUTexturePaletteReg reg, bool clut_is_8bit)
{
  GPU_SW_Rasterizer::UpdateCLUT(reg, clut_is_8bit);
}

void GPU_SW::OnBufferSwapped()
{
}

void GPU_SW::FlushRender()
{
}

void GPU_SW::RestoreDeviceContext()
{
}

void GPU_SW::SetDisplayTextureFormat()
{
  static constexpr const std::array formats_for_16bit = {GPUTexture::Format::RGB565, GPUTexture::Format::RGBA5551,
                                                         GPUTexture::Format::RGBA8, GPUTexture::Format::BGRA8};
  static constexpr const std::array formats_for_24bit = {GPUTexture::Format::RGBA8, GPUTexture::Format::BGRA8,
                                                         GPUTexture::Format::RGB565, GPUTexture::Format::RGBA5551};
  for (const GPUTexture::Format format : formats_for_16bit)
  {
    if (g_gpu_device->SupportsTextureFormat(format))
    {
      m_16bit_display_format = format;
      break;
    }
  }
  for (const GPUTexture::Format format : formats_for_24bit)
  {
    if (g_gpu_device->SupportsTextureFormat(format))
    {
      m_24bit_display_format = format;
      break;
    }
  }
}
GPUTexture* GPU_SW::GetDisplayTexture(u32 width, u32 height, GPUTexture::Format format)
{
  if (!m_upload_texture || m_upload_texture->GetWidth() != width || m_upload_texture->GetHeight() != height ||
      m_upload_texture->GetFormat() != format)
  {
    ClearDisplayTexture();
    g_gpu_device->RecycleTexture(std::move(m_upload_texture));
    m_upload_texture =
      g_gpu_device->FetchTexture(width, height, 1, 1, 1, GPUTexture::Type::DynamicTexture, format, nullptr, 0);
    if (!m_upload_texture) [[unlikely]]
      ERROR_LOG("Failed to create {}x{} {} texture", width, height, static_cast<u32>(format));
  }

  return m_upload_texture.get();
}

template<GPUTexture::Format out_format, typename out_type>
static void CopyOutRow16(const u16* src_ptr, out_type* dst_ptr, u32 width);

template<GPUTexture::Format out_format, typename out_type>
static out_type VRAM16ToOutput(u16 value);

template<>
ALWAYS_INLINE u16 VRAM16ToOutput<GPUTexture::Format::RGBA5551, u16>(u16 value)
{
  return (value & 0x3E0) | ((value >> 10) & 0x1F) | ((value & 0x1F) << 10);
}

template<>
ALWAYS_INLINE u16 VRAM16ToOutput<GPUTexture::Format::RGB565, u16>(u16 value)
{
  return ((value & 0x3E0) << 1) | ((value & 0x20) << 1) | ((value >> 10) & 0x1F) | ((value & 0x1F) << 11);
}

template<>
ALWAYS_INLINE u32 VRAM16ToOutput<GPUTexture::Format::RGBA8, u32>(u16 value)
{
  const u32 value32 = ZeroExtend32(value);
  const u32 r = (value32 & 31u) << 3;
  const u32 g = ((value32 >> 5) & 31u) << 3;
  const u32 b = ((value32 >> 10) & 31u) << 3;
  const u32 a = ((value >> 15) != 0) ? 255 : 0;
  return ZeroExtend32(r) | (ZeroExtend32(g) << 8) | (ZeroExtend32(b) << 16) | (ZeroExtend32(a) << 24);
}

template<>
ALWAYS_INLINE u32 VRAM16ToOutput<GPUTexture::Format::BGRA8, u32>(u16 value)
{
  const u32 value32 = ZeroExtend32(value);
  const u32 r = (value32 & 31u) << 3;
  const u32 g = ((value32 >> 5) & 31u) << 3;
  const u32 b = ((value32 >> 10) & 31u) << 3;
  return ZeroExtend32(b) | (ZeroExtend32(g) << 8) | (ZeroExtend32(r) << 16) | (0xFF000000u);
}

template<>
ALWAYS_INLINE void CopyOutRow16<GPUTexture::Format::RGBA5551, u16>(const u16* src_ptr, u16* dst_ptr, u32 width)
{
  u32 col = 0;

  const u32 aligned_width = Common::AlignDownPow2(width, 8);
  for (; col < aligned_width; col += 8)
  {
    constexpr GSVector4i single_mask = GSVector4i::cxpr16(0x1F);
    GSVector4i value = GSVector4i::load<false>(src_ptr);
    src_ptr += 8;
    GSVector4i a = value & GSVector4i::cxpr16(0x3E0);
    GSVector4i b = value.srl16<10>() & single_mask;
    GSVector4i c = (value & single_mask).sll16<10>();
    value = (a | b) | c;
    GSVector4i::store<false>(dst_ptr, value);
    dst_ptr += 8;
  }

  for (; col < width; col++)
    *(dst_ptr++) = VRAM16ToOutput<GPUTexture::Format::RGBA5551, u16>(*(src_ptr++));
}

template<>
ALWAYS_INLINE void CopyOutRow16<GPUTexture::Format::RGB565, u16>(const u16* src_ptr, u16* dst_ptr, u32 width)
{
  u32 col = 0;

  const u32 aligned_width = Common::AlignDownPow2(width, 8);
  for (; col < aligned_width; col += 8)
  {
    constexpr GSVector4i single_mask = GSVector4i::cxpr16(0x1F);
    GSVector4i value = GSVector4i::load<false>(src_ptr);
    src_ptr += 8;
    GSVector4i a = (value & GSVector4i::cxpr16(0x3E0)).sll16<1>(); // (value & 0x3E0) << 1
    GSVector4i b = (value & GSVector4i::cxpr16(0x20)).sll16<1>();  // (value & 0x20) << 1
    GSVector4i c = (value.srl16<10>() & single_mask);              // ((value >> 10) & 0x1F)
    GSVector4i d = (value & single_mask).sll16<11>();              // ((value & 0x1F) << 11)
    value = (((a | b) | c) | d);
    GSVector4i::store<false>(dst_ptr, value);
    dst_ptr += 8;
  }

  for (; col < width; col++)
    *(dst_ptr++) = VRAM16ToOutput<GPUTexture::Format::RGB565, u16>(*(src_ptr++));
}

template<>
ALWAYS_INLINE void CopyOutRow16<GPUTexture::Format::RGBA8, u32>(const u16* src_ptr, u32* dst_ptr, u32 width)
{
  for (u32 col = 0; col < width; col++)
    *(dst_ptr++) = VRAM16ToOutput<GPUTexture::Format::RGBA8, u32>(*(src_ptr++));
}

template<>
ALWAYS_INLINE void CopyOutRow16<GPUTexture::Format::BGRA8, u32>(const u16* src_ptr, u32* dst_ptr, u32 width)
{
  for (u32 col = 0; col < width; col++)
    *(dst_ptr++) = VRAM16ToOutput<GPUTexture::Format::BGRA8, u32>(*(src_ptr++));
}

template<GPUTexture::Format display_format>
ALWAYS_INLINE_RELEASE bool GPU_SW::CopyOut15Bit(u32 src_x, u32 src_y, u32 width, u32 height, u32 line_skip)
{
  using OutputPixelType =
    std::conditional_t<display_format == GPUTexture::Format::RGBA8 || display_format == GPUTexture::Format::BGRA8, u32,
                       u16>;

  GPUTexture* texture = GetDisplayTexture(width, height, display_format);
  if (!texture) [[unlikely]]
    return false;

  u32 dst_stride = width * sizeof(OutputPixelType);
  u8* dst_ptr = m_upload_buffer.data();
  const bool mapped = texture->Map(reinterpret_cast<void**>(&dst_ptr), &dst_stride, 0, 0, width, height);

  // Fast path when not wrapping around.
  if ((src_x + width) <= VRAM_WIDTH && (src_y + height) <= VRAM_HEIGHT)
  {
    const u16* src_ptr = &g_vram[src_y * VRAM_WIDTH + src_x];
    const u32 src_step = VRAM_WIDTH << line_skip;
    for (u32 row = 0; row < height; row++)
    {
      CopyOutRow16<display_format>(src_ptr, reinterpret_cast<OutputPixelType*>(dst_ptr), width);
      src_ptr += src_step;
      dst_ptr += dst_stride;
    }
  }
  else
  {
    const u32 end_x = src_x + width;
    const u32 y_step = (1 << line_skip);
    for (u32 row = 0; row < height; row++)
    {
      const u16* src_row_ptr = &g_vram[(src_y % VRAM_HEIGHT) * VRAM_WIDTH];
      OutputPixelType* dst_row_ptr = reinterpret_cast<OutputPixelType*>(dst_ptr);

      for (u32 col = src_x; col < end_x; col++)
        *(dst_row_ptr++) = VRAM16ToOutput<display_format, OutputPixelType>(src_row_ptr[col % VRAM_WIDTH]);

      src_y += y_step;
      dst_ptr += dst_stride;
    }
  }

  if (mapped)
    texture->Unmap();
  else
    texture->Update(0, 0, width, height, m_upload_buffer.data(), dst_stride);

  return true;
}

template<GPUTexture::Format display_format>
ALWAYS_INLINE_RELEASE bool GPU_SW::CopyOut24Bit(u32 src_x, u32 src_y, u32 skip_x, u32 width, u32 height, u32 line_skip)
{
  using OutputPixelType =
    std::conditional_t<display_format == GPUTexture::Format::RGBA8 || display_format == GPUTexture::Format::BGRA8, u32,
                       u16>;

  GPUTexture* texture = GetDisplayTexture(width, height, display_format);
  if (!texture) [[unlikely]]
    return false;

  u32 dst_stride = Common::AlignUpPow2<u32>(width * sizeof(OutputPixelType), 4);
  u8* dst_ptr = m_upload_buffer.data();
  const bool mapped = texture->Map(reinterpret_cast<void**>(&dst_ptr), &dst_stride, 0, 0, width, height);

  if ((src_x + width) <= VRAM_WIDTH && (src_y + (height << line_skip)) <= VRAM_HEIGHT)
  {
    const u8* src_ptr = reinterpret_cast<const u8*>(&g_vram[src_y * VRAM_WIDTH + src_x]) + (skip_x * 3);
    const u32 src_stride = (VRAM_WIDTH << line_skip) * sizeof(u16);
    for (u32 row = 0; row < height; row++)
    {
      if constexpr (display_format == GPUTexture::Format::RGBA8)
      {
        const u8* src_row_ptr = src_ptr;
        u8* dst_row_ptr = reinterpret_cast<u8*>(dst_ptr);
        for (u32 col = 0; col < width; col++)
        {
          *(dst_row_ptr++) = *(src_row_ptr++);
          *(dst_row_ptr++) = *(src_row_ptr++);
          *(dst_row_ptr++) = *(src_row_ptr++);
          *(dst_row_ptr++) = 0xFF;
        }
      }
      else if constexpr (display_format == GPUTexture::Format::BGRA8)
      {
        const u8* src_row_ptr = src_ptr;
        u8* dst_row_ptr = reinterpret_cast<u8*>(dst_ptr);
        for (u32 col = 0; col < width; col++)
        {
          *(dst_row_ptr++) = src_row_ptr[2];
          *(dst_row_ptr++) = src_row_ptr[1];
          *(dst_row_ptr++) = src_row_ptr[0];
          *(dst_row_ptr++) = 0xFF;
          src_row_ptr += 3;
        }
      }
      else if constexpr (display_format == GPUTexture::Format::RGB565)
      {
        const u8* src_row_ptr = src_ptr;
        u16* dst_row_ptr = reinterpret_cast<u16*>(dst_ptr);
        for (u32 col = 0; col < width; col++)
        {
          *(dst_row_ptr++) = ((static_cast<u16>(src_row_ptr[0]) >> 3) << 11) |
                             ((static_cast<u16>(src_row_ptr[1]) >> 2) << 5) | (static_cast<u16>(src_row_ptr[2]) >> 3);
          src_row_ptr += 3;
        }
      }
      else if constexpr (display_format == GPUTexture::Format::RGBA5551)
      {
        const u8* src_row_ptr = src_ptr;
        u16* dst_row_ptr = reinterpret_cast<u16*>(dst_ptr);
        for (u32 col = 0; col < width; col++)
        {
          *(dst_row_ptr++) = ((static_cast<u16>(src_row_ptr[0]) >> 3) << 10) |
                             ((static_cast<u16>(src_row_ptr[1]) >> 3) << 5) | (static_cast<u16>(src_row_ptr[2]) >> 3);
          src_row_ptr += 3;
        }
      }

      src_ptr += src_stride;
      dst_ptr += dst_stride;
    }
  }
  else
  {
    const u32 y_step = (1 << line_skip);

    for (u32 row = 0; row < height; row++)
    {
      const u16* src_row_ptr = &g_vram[(src_y % VRAM_HEIGHT) * VRAM_WIDTH];
      OutputPixelType* dst_row_ptr = reinterpret_cast<OutputPixelType*>(dst_ptr);

      for (u32 col = 0; col < width; col++)
      {
        const u32 offset = (src_x + (((skip_x + col) * 3) / 2));
        const u16 s0 = src_row_ptr[offset % VRAM_WIDTH];
        const u16 s1 = src_row_ptr[(offset + 1) % VRAM_WIDTH];
        const u8 shift = static_cast<u8>(col & 1u) * 8;
        const u32 rgb = (((ZeroExtend32(s1) << 16) | ZeroExtend32(s0)) >> shift);

        if constexpr (display_format == GPUTexture::Format::RGBA8)
        {
          *(dst_row_ptr++) = rgb | 0xFF000000u;
        }
        else if constexpr (display_format == GPUTexture::Format::BGRA8)
        {
          *(dst_row_ptr++) = (rgb & 0x00FF00) | ((rgb & 0xFF) << 16) | ((rgb >> 16) & 0xFF) | 0xFF000000u;
        }
        else if constexpr (display_format == GPUTexture::Format::RGB565)
        {
          *(dst_row_ptr++) = ((rgb >> 3) & 0x1F) | (((rgb >> 10) << 5) & 0x7E0) | (((rgb >> 19) << 11) & 0x3E0000);
        }
        else if constexpr (display_format == GPUTexture::Format::RGBA5551)
        {
          *(dst_row_ptr++) = ((rgb >> 3) & 0x1F) | (((rgb >> 11) << 5) & 0x3E0) | (((rgb >> 19) << 10) & 0x1F0000);
        }
      }

      src_y += y_step;
      dst_ptr += dst_stride;
    }
  }

  if (mapped)
    texture->Unmap();
  else
    texture->Update(0, 0, width, height, m_upload_buffer.data(), dst_stride);

  return true;
}

bool GPU_SW::CopyOut(u32 src_x, u32 src_y, u32 skip_x, u32 width, u32 height, u32 line_skip, bool is_24bit)
{
  if (!is_24bit)
  {
    DebugAssert(skip_x == 0);

    switch (m_16bit_display_format)
    {
      case GPUTexture::Format::RGBA5551:
        return CopyOut15Bit<GPUTexture::Format::RGBA5551>(src_x, src_y, width, height, line_skip);

      case GPUTexture::Format::RGB565:
        return CopyOut15Bit<GPUTexture::Format::RGB565>(src_x, src_y, width, height, line_skip);

      case GPUTexture::Format::RGBA8:
        return CopyOut15Bit<GPUTexture::Format::RGBA8>(src_x, src_y, width, height, line_skip);

      case GPUTexture::Format::BGRA8:
        return CopyOut15Bit<GPUTexture::Format::BGRA8>(src_x, src_y, width, height, line_skip);

      default:
        UnreachableCode();
    }
  }
  else
  {
    switch (m_24bit_display_format)
    {
      case GPUTexture::Format::RGBA5551:
        return CopyOut24Bit<GPUTexture::Format::RGBA5551>(src_x, src_y, skip_x, width, height, line_skip);

      case GPUTexture::Format::RGB565:
        return CopyOut24Bit<GPUTexture::Format::RGB565>(src_x, src_y, skip_x, width, height, line_skip);

      case GPUTexture::Format::RGBA8:
        return CopyOut24Bit<GPUTexture::Format::RGBA8>(src_x, src_y, skip_x, width, height, line_skip);

      case GPUTexture::Format::BGRA8:
        return CopyOut24Bit<GPUTexture::Format::BGRA8>(src_x, src_y, skip_x, width, height, line_skip);

      default:
        UnreachableCode();
    }
  }
}

void GPU_SW::UpdateDisplay(const GPUBackendUpdateDisplayCommand* cmd)
{
  if (!g_settings.debugging.show_vram)
  {
    if (cmd->display_disabled)
    {
      ClearDisplayTexture();
      return;
    }

    const bool is_24bit = cmd->display_24bit;
    const bool interlaced = cmd->interlaced_display_enabled;
    const u32 field = cmd->interlaced_display_field;
    const u32 vram_offset_x = is_24bit ? cmd->X : cmd->display_vram_left;
    const u32 vram_offset_y = cmd->display_vram_top + ((interlaced && cmd->interlaced_display_interleaved) ? field : 0);
    const u32 skip_x = is_24bit ? (cmd->display_vram_left - cmd->X) : 0;
    const u32 read_width = cmd->display_vram_width;
    const u32 read_height = interlaced ? (cmd->display_vram_height / 2) : cmd->display_vram_height;

    if (cmd->interlaced_display_enabled)
    {
      const u32 line_skip = cmd->interlaced_display_interleaved;
      if (CopyOut(vram_offset_x, vram_offset_y, skip_x, read_width, read_height, line_skip, is_24bit))
      {
        SetDisplayTexture(m_upload_texture.get(), nullptr, 0, 0, read_width, read_height);
        if (is_24bit && g_settings.gpu_24bit_chroma_smoothing)
        {
          if (ApplyChromaSmoothing())
            Deinterlace(field, 0);
        }
        else
        {
          Deinterlace(field, 0);
        }
      }
    }
    else
    {
      if (CopyOut(vram_offset_x, vram_offset_y, skip_x, read_width, read_height, 0, is_24bit))
      {
        SetDisplayTexture(m_upload_texture.get(), nullptr, 0, 0, read_width, read_height);
        if (is_24bit && g_settings.gpu_24bit_chroma_smoothing)
          ApplyChromaSmoothing();
      }
    }
  }
  else
  {
    if (CopyOut(0, 0, 0, VRAM_WIDTH, VRAM_HEIGHT, 0, false))
      SetDisplayTexture(m_upload_texture.get(), nullptr, 0, 0, VRAM_WIDTH, VRAM_HEIGHT);
  }
}

std::unique_ptr<GPUBackend> GPUBackend::CreateSoftwareBackend()
{
  return std::make_unique<GPU_SW>();
}
