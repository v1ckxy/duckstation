// SPDX-FileCopyrightText: 2019-2023 Connor McLaughlin <stenzek@gmail.com>
// SPDX-License-Identifier: (GPL-3.0 OR CC-BY-NC-ND-4.0)

#ifdef __INTELLISENSE__

#include "common/gsvector.h"
#include "gpu.h"
#include <algorithm>

#define USE_VECTOR 1
#define GSVECTOR_HAS_SRLV 1

extern GPU_SW_Rasterizer::DitherLUT g_dither_lut;

namespace GPU_SW_Rasterizer {

#endif

// TODO: UpdateVRAM, FillVRAM, etc.

#ifdef USE_VECTOR
#if 0
static u16 s_vram_backup[VRAM_WIDTH * VRAM_HEIGHT];
static u16 s_new_vram[VRAM_WIDTH * VRAM_HEIGHT];
#define BACKUP_VRAM()                                                                                                  \
  do                                                                                                                   \
  {                                                                                                                    \
    std::memcpy(s_vram_backup, g_vram, sizeof(g_vram));                                                                \
    s_bad_counter++;                                                                                                   \
  } while (0)
#define CHECK_VRAM(drawer)                                                                                             \
  do                                                                                                                   \
  {                                                                                                                    \
    std::memcpy(s_new_vram, g_vram, sizeof(g_vram));                                                                   \
    std::memcpy(g_vram, s_vram_backup, sizeof(g_vram));                                                                \
                                                                                                                       \
    drawer;                                                                                                            \
    for (u32 vidx = 0; vidx < (VRAM_WIDTH * VRAM_HEIGHT); vidx++)                                                      \
    {                                                                                                                  \
      if (s_new_vram[vidx] != g_vram[vidx])                                                                            \
      {                                                                                                                \
        fprintf(stderr, "[%u] Mismatch at %d,%d, expected %04x got %04x\n", s_bad_counter, (vidx % VRAM_WIDTH),        \
                (vidx / VRAM_WIDTH), g_vram[vidx], s_new_vram[vidx]);                                                  \
        AssertMsg(false, "Mismatch");                                                                                  \
      }                                                                                                                \
    }                                                                                                                  \
    /*Assert(std::memcmp(g_vram, s_new_vram, sizeof(g_vram)) == 0)*/                                                   \
  } while (0)
#else
#define BACKUP_VRAM()
#define CHECK_VRAM(drawer)
#endif
#endif

namespace {
enum
{
  Line_XY_FractBits = 32
};
enum
{
  Line_RGB_FractBits = 12
};

struct i_deltas
{
  u32 du_dx, dv_dx;
  u32 dr_dx, dg_dx, db_dx;

  u32 du_dy, dv_dy;
  u32 dr_dy, dg_dy, db_dy;
};

struct i_group
{
  u32 u, v;
  u32 r, g, b;
};
struct line_fxp_coord
{
  u64 x, y;
  u32 r, g, b;
};

struct line_fxp_step
{
  s64 dx_dk, dy_dk;
  s32 dr_dk, dg_dk, db_dk;
};
} // namespace

[[maybe_unused]] ALWAYS_INLINE_RELEASE static u16 GetPixel(const u32 x, const u32 y)
{
  return g_vram[VRAM_WIDTH * y + x];
}
[[maybe_unused]] ALWAYS_INLINE_RELEASE static u16* GetPixelPtr(const u32 x, const u32 y)
{
  return &g_vram[VRAM_WIDTH * y + x];
}
[[maybe_unused]] ALWAYS_INLINE_RELEASE static void SetPixel(const u32 x, const u32 y, const u16 value)
{
  g_vram[VRAM_WIDTH * y + x] = value;
}

[[maybe_unused]] ALWAYS_INLINE_RELEASE static constexpr std::tuple<u8, u8> UnpackTexcoord(u16 texcoord)
{
  return std::make_tuple(static_cast<u8>(texcoord), static_cast<u8>(texcoord >> 8));
}

[[maybe_unused]] ALWAYS_INLINE_RELEASE static constexpr std::tuple<u8, u8, u8> UnpackColorRGB24(u32 rgb24)
{
  return std::make_tuple(static_cast<u8>(rgb24), static_cast<u8>(rgb24 >> 8), static_cast<u8>(rgb24 >> 16));
}

template<bool texture_enable, bool raw_texture_enable, bool transparency_enable, bool dithering_enable>
[[maybe_unused]] ALWAYS_INLINE_RELEASE static void ShadePixel(const GPUBackendDrawCommand* cmd, u32 x, u32 y,
                                                              u8 color_r, u8 color_g, u8 color_b, u8 texcoord_x,
                                                              u8 texcoord_y)
{
  u16 color;
  if constexpr (texture_enable)
  {
    // Apply texture window
    texcoord_x = (texcoord_x & cmd->window.and_x) | cmd->window.or_x;
    texcoord_y = (texcoord_y & cmd->window.and_y) | cmd->window.or_y;

    u16 texture_color;
    switch (cmd->draw_mode.texture_mode)
    {
      case GPUTextureMode::Palette4Bit:
      {
        const u16 palette_value =
          GetPixel((cmd->draw_mode.GetTexturePageBaseX() + ZeroExtend32(texcoord_x / 4)) % VRAM_WIDTH,
                   (cmd->draw_mode.GetTexturePageBaseY() + ZeroExtend32(texcoord_y)) % VRAM_HEIGHT);
        const size_t palette_index = (palette_value >> ((texcoord_x % 4) * 4)) & 0x0Fu;
        texture_color = g_gpu_clut[palette_index];
      }
      break;

      case GPUTextureMode::Palette8Bit:
      {
        const u16 palette_value =
          GetPixel((cmd->draw_mode.GetTexturePageBaseX() + ZeroExtend32(texcoord_x / 2)) % VRAM_WIDTH,
                   (cmd->draw_mode.GetTexturePageBaseY() + ZeroExtend32(texcoord_y)) % VRAM_HEIGHT);
        const size_t palette_index = (palette_value >> ((texcoord_x % 2) * 8)) & 0xFFu;
        texture_color = g_gpu_clut[palette_index];
      }
      break;

      default:
      {
        texture_color = GetPixel((cmd->draw_mode.GetTexturePageBaseX() + ZeroExtend32(texcoord_x)) % VRAM_WIDTH,
                                 (cmd->draw_mode.GetTexturePageBaseY() + ZeroExtend32(texcoord_y)) % VRAM_HEIGHT);
      }
      break;
    }

    if (texture_color == 0)
      return;

    if constexpr (raw_texture_enable)
    {
      color = texture_color;
    }
    else
    {
      const u32 dither_y = (dithering_enable) ? (y & 3u) : 2u;
      const u32 dither_x = (dithering_enable) ? (x & 3u) : 3u;

      color =
        (ZeroExtend16(g_dither_lut[dither_y][dither_x][(u16(texture_color & 0x1Fu) * u16(color_r)) >> 4]) << 0) |
        (ZeroExtend16(g_dither_lut[dither_y][dither_x][(u16((texture_color >> 5) & 0x1Fu) * u16(color_g)) >> 4]) << 5) |
        (ZeroExtend16(g_dither_lut[dither_y][dither_x][(u16((texture_color >> 10) & 0x1Fu) * u16(color_b)) >> 4])
         << 10) |
        (texture_color & 0x8000u);
    }
  }
  else
  {
    const u32 dither_y = (dithering_enable) ? (y & 3u) : 2u;
    const u32 dither_x = (dithering_enable) ? (x & 3u) : 3u;

    // Non-textured transparent polygons don't set bit 15, but are treated as transparent.
    color = (ZeroExtend16(g_dither_lut[dither_y][dither_x][color_r]) << 0) |
            (ZeroExtend16(g_dither_lut[dither_y][dither_x][color_g]) << 5) |
            (ZeroExtend16(g_dither_lut[dither_y][dither_x][color_b]) << 10) | (transparency_enable ? 0x8000u : 0);
  }

  const u16 bg_color = GetPixel(static_cast<u32>(x), static_cast<u32>(y));
  if constexpr (transparency_enable)
  {
    if (color & 0x8000u || !texture_enable)
    {
      // Based on blargg's efficient 15bpp pixel math.
      u32 bg_bits = ZeroExtend32(bg_color);
      u32 fg_bits = ZeroExtend32(color);
      switch (cmd->draw_mode.transparency_mode)
      {
        case GPUTransparencyMode::HalfBackgroundPlusHalfForeground:
        {
          bg_bits |= 0x8000u;
          color = Truncate16(((fg_bits + bg_bits) - ((fg_bits ^ bg_bits) & 0x0421u)) >> 1);
        }
        break;

        case GPUTransparencyMode::BackgroundPlusForeground:
        {
          bg_bits &= ~0x8000u;

          const u32 sum = fg_bits + bg_bits;
          const u32 carry = (sum - ((fg_bits ^ bg_bits) & 0x8421u)) & 0x8420u;

          color = Truncate16((sum - carry) | (carry - (carry >> 5)));
        }
        break;

        case GPUTransparencyMode::BackgroundMinusForeground:
        {
          bg_bits |= 0x8000u;
          fg_bits &= ~0x8000u;

          const u32 diff = bg_bits - fg_bits + 0x108420u;
          const u32 borrow = (diff - ((bg_bits ^ fg_bits) & 0x108420u)) & 0x108420u;

          color = Truncate16((diff - borrow) & (borrow - (borrow >> 5)));
        }
        break;

        case GPUTransparencyMode::BackgroundPlusQuarterForeground:
        {
          bg_bits &= ~0x8000u;
          fg_bits = ((fg_bits >> 2) & 0x1CE7u) | 0x8000u;

          const u32 sum = fg_bits + bg_bits;
          const u32 carry = (sum - ((fg_bits ^ bg_bits) & 0x8421u)) & 0x8420u;

          color = Truncate16((sum - carry) | (carry - (carry >> 5)));
        }
        break;

        default:
          break;
      }

      // See above.
      if constexpr (!texture_enable)
        color &= ~0x8000u;
    }
  }

  const u16 mask_and = cmd->params.GetMaskAND();
  if ((bg_color & mask_and) != 0)
    return;

  DebugAssert(static_cast<u32>(x) < VRAM_WIDTH && static_cast<u32>(y) < VRAM_HEIGHT);
  SetPixel(static_cast<u32>(x), static_cast<u32>(y), color | cmd->params.GetMaskOR());
}

#ifndef USE_VECTOR

template<bool texture_enable, bool raw_texture_enable, bool transparency_enable>
static void DrawRectangle(const GPUBackendDrawRectangleCommand* cmd)
{
  const s32 origin_x = cmd->x;
  const s32 origin_y = cmd->y;
  const auto [r, g, b] = UnpackColorRGB24(cmd->color);
  const auto [origin_texcoord_x, origin_texcoord_y] = UnpackTexcoord(cmd->texcoord);

  for (u32 offset_y = 0; offset_y < cmd->height; offset_y++)
  {
    const s32 y = origin_y + static_cast<s32>(offset_y);
    if (y < static_cast<s32>(g_drawing_area.top) || y > static_cast<s32>(g_drawing_area.bottom) ||
        (cmd->params.interlaced_rendering && cmd->params.active_line_lsb == (Truncate8(static_cast<u32>(y)) & 1u)))
    {
      continue;
    }

    const u32 draw_y = static_cast<u32>(y) & VRAM_HEIGHT_MASK;
    const u8 texcoord_y = Truncate8(ZeroExtend32(origin_texcoord_y) + offset_y);

    for (u32 offset_x = 0; offset_x < cmd->width; offset_x++)
    {
      const s32 x = origin_x + static_cast<s32>(offset_x);
      if (x < static_cast<s32>(g_drawing_area.left) || x > static_cast<s32>(g_drawing_area.right))
        continue;

      const u8 texcoord_x = Truncate8(ZeroExtend32(origin_texcoord_x) + offset_x);

      ShadePixel<texture_enable, raw_texture_enable, transparency_enable, false>(cmd, static_cast<u32>(x), draw_y, r, g,
                                                                                 b, texcoord_x, texcoord_y);
    }
  }
}

#else // USE_VECTOR

ALWAYS_INLINE_RELEASE static GSVector4i GatherVector(GSVector4i coord_x, GSVector4i coord_y)
{
  GSVector4i offsets = coord_y.sll32<11>();    // y * 2048 (1024 * sizeof(pixel))
  offsets = offsets.add32(coord_x.sll32<1>()); // x * 2 (x * sizeof(pixel))

  const u32 o0 = offsets.extract32<0>();
  const u32 o1 = offsets.extract32<1>();
  const u32 o2 = offsets.extract32<2>();
  const u32 o3 = offsets.extract32<3>();

  // TODO: split in two, merge, maybe could be zx loaded instead..
  u16 p0, p1, p2, p3;
  std::memcpy(&p0, reinterpret_cast<const u8*>(g_vram) + o0, sizeof(p0));
  std::memcpy(&p1, reinterpret_cast<const u8*>(g_vram) + o1, sizeof(p1));
  std::memcpy(&p2, reinterpret_cast<const u8*>(g_vram) + o2, sizeof(p2));
  std::memcpy(&p3, reinterpret_cast<const u8*>(g_vram) + o3, sizeof(p3));
  GSVector4i pixels = GSVector4i::load(p0);
  pixels = pixels.insert16<2>(p1);
  pixels = pixels.insert16<4>(p2);
  pixels = pixels.insert16<6>(p3);

  return pixels;
}

ALWAYS_INLINE_RELEASE static GSVector4i GatherCLUTVector(GSVector4i indices)
{
  const GSVector4i offsets = indices.sll32<1>(); // x * 2 (x * sizeof(pixel))
  const u32 o0 = offsets.extract32<0>();
  const u32 o1 = offsets.extract32<1>();
  const u32 o2 = offsets.extract32<2>();
  const u32 o3 = offsets.extract32<3>();

  // TODO: split in two, merge, maybe could be zx loaded instead..
  u16 p0, p1, p2, p3;
  std::memcpy(&p0, reinterpret_cast<const u8*>(g_gpu_clut) + o0, sizeof(p0));
  std::memcpy(&p1, reinterpret_cast<const u8*>(g_gpu_clut) + o1, sizeof(p1));
  std::memcpy(&p2, reinterpret_cast<const u8*>(g_gpu_clut) + o2, sizeof(p2));
  std::memcpy(&p3, reinterpret_cast<const u8*>(g_gpu_clut) + o3, sizeof(p3));
  GSVector4i pixels = GSVector4i::load(p0);
  pixels = pixels.insert16<2>(p1);
  pixels = pixels.insert16<4>(p2);
  pixels = pixels.insert16<6>(p3);

  return pixels;
}

ALWAYS_INLINE_RELEASE static GSVector4i LoadVector(u32 x, u32 y)
{
  if (x <= (VRAM_WIDTH - 4))
  {
    return GSVector4i::loadl(&g_vram[y * VRAM_WIDTH + x]).u16to32();
  }
  else
  {
    const u16* line = &g_vram[y * VRAM_WIDTH];
    GSVector4i pixels = GSVector4i(line[(x++) & VRAM_WIDTH_MASK]);
    pixels = pixels.insert16<2>(line[(x++) & VRAM_WIDTH_MASK]);
    pixels = pixels.insert16<4>(line[(x++) & VRAM_WIDTH_MASK]);
    pixels = pixels.insert16<6>(line[x & VRAM_WIDTH_MASK]);
    return pixels;
  }
}

ALWAYS_INLINE_RELEASE static void StoreVector(u32 x, u32 y, GSVector4i color)
{
  if (x <= (VRAM_WIDTH - 4))
  {
    GSVector4i::storel(&g_vram[y * VRAM_WIDTH + x], color);
  }
  else
  {
    u16* line = &g_vram[y * VRAM_WIDTH];
    line[(x++) & VRAM_WIDTH_MASK] = Truncate16(color.extract16<0>());
    line[(x++) & VRAM_WIDTH_MASK] = Truncate16(color.extract16<1>());
    line[(x++) & VRAM_WIDTH_MASK] = Truncate16(color.extract16<2>());
    line[x & VRAM_WIDTH_MASK] = Truncate16(color.extract16<3>());
  }
}

ALWAYS_INLINE_RELEASE static void RGB5A1ToRG_BA(GSVector4i rgb5a1, GSVector4i& rg, GSVector4i& ba)
{
  rg = rgb5a1 & GSVector4i::cxpr(0x1F);                     // R | R | R | R
  rg = rg | (rgb5a1 & GSVector4i::cxpr(0x3E0)).sll32<11>(); // R0G0 | R0G0 | R0G0 | R0G0
  ba = rgb5a1.srl32<10>() & GSVector4i::cxpr(0x1F);         // B | B | B | B
  ba = ba | (rgb5a1 & GSVector4i::cxpr(0x8000)).sll32<1>(); // B0A0 | B0A0 | B0A0 | B0A0
}

ALWAYS_INLINE_RELEASE static GSVector4i RG_BAToRGB5A1(GSVector4i rg, GSVector4i ba)
{
  GSVector4i res;

  res = rg & GSVector4i::cxpr(0x1F);                       // R | R | R | R
  res = res | (rg.srl32<11>() & GSVector4i::cxpr(0x3E0));  // RG | RG | RG | RG
  res = res | ((ba & GSVector4i::cxpr(0x1F)).sll32<10>()); // RGB | RGB | RGB | RGB
  res = res | ba.srl32<16>().sll32<15>();                  // RGBA | RGBA | RGBA | RGBA

  return res;
}

// Color repeated twice for RG packing, then duplicated to we can load based on the X offset.
static constexpr s16 VECTOR_DITHER_MATRIX[4][16] = {
#define P(m, n) static_cast<s16>(DITHER_MATRIX[m][n]), static_cast<s16>(DITHER_MATRIX[m][n])
#define R(m) P(m, 0), P(m, 1), P(m, 2), P(m, 3), P(m, 0), P(m, 1), P(m, 2), P(m, 3)

  {R(0)}, {R(1)}, {R(2)}, {R(3)}

#undef R
#undef P
};

template<bool texture_enable, bool raw_texture_enable, bool transparency_enable, bool dithering_enable>
ALWAYS_INLINE_RELEASE static void
ShadePixel(const GPUBackendDrawCommand* cmd, u32 start_x, u32 y, GSVector4i vertex_color_rg, GSVector4i vertex_color_ba,
           GSVector4i texcoord_x, GSVector4i texcoord_y, GSVector4i preserve_mask, GSVector4i dither)
{
  static constinit GSVector4i coord_mask_x = GSVector4i::cxpr(VRAM_WIDTH_MASK);
  static constinit GSVector4i coord_mask_y = GSVector4i::cxpr(VRAM_HEIGHT_MASK);

  GSVector4i color;

  if constexpr (texture_enable)
  {
    // Apply texture window
    texcoord_x = (texcoord_x & GSVector4i(cmd->window.and_x)) | GSVector4i(cmd->window.or_x);
    texcoord_y = (texcoord_y & GSVector4i(cmd->window.and_y)) | GSVector4i(cmd->window.or_y);

    const GSVector4i base_x = GSVector4i(cmd->draw_mode.GetTexturePageBaseX());
    const GSVector4i base_y = GSVector4i(cmd->draw_mode.GetTexturePageBaseY());
    const GSVector4i palette_x = GSVector4i(cmd->palette.GetXBase());
    const GSVector4i palette_y = GSVector4i(cmd->palette.GetYBase());

    texcoord_y = base_y.add32(texcoord_y) & coord_mask_y;

    GSVector4i texture_color;
    switch (cmd->draw_mode.texture_mode)
    {
      case GPUTextureMode::Palette4Bit:
      {
        GSVector4i load_texcoord_x = texcoord_x.srl32<2>();
        load_texcoord_x = base_x.add32(load_texcoord_x);
        load_texcoord_x = load_texcoord_x & coord_mask_x;

        // todo: sse4 path
        GSVector4i palette_shift = (texcoord_x & GSVector4i::cxpr(3)).sll32<2>();
        GSVector4i palette_indices = GatherVector(load_texcoord_x, texcoord_y);
#ifdef GSVECTOR_HAS_SRLV
        palette_indices = palette_indices.srlv32(palette_shift) & GSVector4i::cxpr(0x0F);
#else
        Assert(false && "Fixme");
#endif

        texture_color = GatherCLUTVector(palette_indices);
      }
      break;

      case GPUTextureMode::Palette8Bit:
      {
        GSVector4i load_texcoord_x = texcoord_x.srl32<1>();
        load_texcoord_x = base_x.add32(load_texcoord_x);
        load_texcoord_x = load_texcoord_x & coord_mask_x;

        GSVector4i palette_shift = (texcoord_x & GSVector4i::cxpr(1)).sll32<3>();
        GSVector4i palette_indices = GatherVector(load_texcoord_x, texcoord_y);
#ifdef GSVECTOR_HAS_SRLV
        palette_indices = palette_indices.srlv32(palette_shift) & GSVector4i::cxpr(0xFF);
#else
        Assert(false && "Fixme");
#endif

        texture_color = GatherCLUTVector(palette_indices);
      }
      break;

      default:
      {
        texcoord_x = base_x.add32(texcoord_x);
        texcoord_x = texcoord_x & coord_mask_x;
        texture_color = GatherVector(texcoord_x, texcoord_y);
      }
      break;
    }

    // check for zero texture colour across the 4 pixels, early out if so
    const GSVector4i texture_transparent_mask = texture_color.eq32(GSVector4i::zero());
    if (texture_transparent_mask.alltrue())
      return;

    preserve_mask = preserve_mask | texture_transparent_mask;

    if constexpr (raw_texture_enable)
    {
      color = texture_color;
    }
    else
    {
      GSVector4i trg, tba;
      RGB5A1ToRG_BA(texture_color, trg, tba);

      // now we have both the texture and vertex color in RG/GA pairs, for 4 pixels, which we can multiply
      GSVector4i rg = trg.mul16l(vertex_color_rg);
      GSVector4i ba = tba.mul16l(vertex_color_ba);

      // TODO: Dither
      // Convert to 5bit.
      if constexpr (dithering_enable)
      {
        rg = rg.sra16<4>().add16(dither).max_i16(GSVector4i::zero()).sra16<3>();
        ba = ba.sra16<4>().add16(dither).max_i16(GSVector4i::zero()).sra16<3>();
      }
      else
      {
        rg = rg.sra16<7>();
        ba = ba.sra16<7>();
      }

      // Bit15 gets passed through as-is.
      ba = ba.blend16<0xaa>(tba);

      // Clamp to 5bit.
      static constexpr GSVector4i colclamp = GSVector4i::cxpr16(0x1F);
      rg = rg.min_u16(colclamp);
      ba = ba.min_u16(colclamp);

      // And interleave back to 16bpp.
      color = RG_BAToRGB5A1(rg, ba);
    }
  }
  else
  {
    // Non-textured transparent polygons don't set bit 15, but are treated as transparent.
    if constexpr (dithering_enable)
    {
      GSVector4i rg = vertex_color_rg.add16(dither).max_i16(GSVector4i::zero()).sra16<3>();
      GSVector4i ba = vertex_color_ba.add16(dither).max_i16(GSVector4i::zero()).sra16<3>();

      // Clamp to 5bit. We use 32bit for BA to set a to zero.
      rg = rg.min_u16(GSVector4i::cxpr16(0x1F));
      ba = ba.min_u16(GSVector4i::cxpr(0x1F));

      // And interleave back to 16bpp.
      color = RG_BAToRGB5A1(rg, ba);
    }
    else
    {
      // Note that bit15 is set to 0 here, which the shift will do.
      const GSVector4i rg = vertex_color_rg.srl16<3>();
      const GSVector4i ba = vertex_color_ba.srl16<3>();
      color = RG_BAToRGB5A1(rg, ba);
    }
  }

  GSVector4i bg_color = LoadVector(start_x, y);

  if constexpr (transparency_enable)
  {
    [[maybe_unused]] GSVector4i transparent_mask;
    if constexpr (texture_enable)
    {
      // Compute transparent_mask, ffff per lane if transparent otherwise 0000
      transparent_mask = color.sra16<15>();
    }

    // TODO: We don't need to OR color here with 0x8000 for textures.
    // 0x8000 is added to match serial path.

    GSVector4i blended_color;
    switch (cmd->draw_mode.transparency_mode)
    {
      case GPUTransparencyMode::HalfBackgroundPlusHalfForeground:
      {
        const GSVector4i fg_bits = color | GSVector4i::cxpr(0x8000u);
        const GSVector4i bg_bits = bg_color | GSVector4i::cxpr(0x8000u);
        const GSVector4i res = fg_bits.add32(bg_bits).sub32((fg_bits ^ bg_bits) & GSVector4i::cxpr(0x0421u)).srl32<1>();
        blended_color = res & GSVector4i::cxpr(0xffff);
      }
      break;

      case GPUTransparencyMode::BackgroundPlusForeground:
      {
        const GSVector4i fg_bits = color | GSVector4i::cxpr(0x8000u);
        const GSVector4i bg_bits = bg_color & GSVector4i::cxpr(0x7FFFu);
        const GSVector4i sum = fg_bits.add32(bg_bits);
        const GSVector4i carry =
          (sum.sub32((fg_bits ^ bg_bits) & GSVector4i::cxpr(0x8421u))) & GSVector4i::cxpr(0x8420u);
        const GSVector4i res = sum.sub32(carry) | carry.sub32(carry.srl32<5>());
        blended_color = res & GSVector4i::cxpr(0xffff);
      }
      break;

      case GPUTransparencyMode::BackgroundMinusForeground:
      {
        const GSVector4i bg_bits = bg_color | GSVector4i::cxpr(0x8000u);
        const GSVector4i fg_bits = color & GSVector4i::cxpr(0x7FFFu);
        const GSVector4i diff = bg_bits.sub32(fg_bits).add32(GSVector4i::cxpr(0x108420u));
        const GSVector4i borrow =
          diff.sub32((bg_bits ^ fg_bits) & GSVector4i::cxpr(0x108420u)) & GSVector4i::cxpr(0x108420u);
        const GSVector4i res = diff.sub32(borrow) & borrow.sub32(borrow.srl32<5>());
        blended_color = res & GSVector4i::cxpr(0xffff);
      }
      break;

      case GPUTransparencyMode::BackgroundPlusQuarterForeground:
      default:
      {
        const GSVector4i bg_bits = bg_color & GSVector4i::cxpr(0x7FFFu);
        const GSVector4i fg_bits =
          ((color | GSVector4i::cxpr(0x8000)).srl32<2>() & GSVector4i::cxpr(0x1CE7u)) | GSVector4i::cxpr(0x8000u);
        const GSVector4i sum = fg_bits.add32(bg_bits);
        const GSVector4i carry = sum.sub32((fg_bits ^ bg_bits) & GSVector4i::cxpr(0x8421u)) & GSVector4i::cxpr(0x8420u);
        const GSVector4i res = sum.sub32(carry) | carry.sub32(carry.srl32<5>());
        blended_color = res & GSVector4i::cxpr(0xffff);
      }
      break;
    }

    // select blended pixels for transparent pixels, otherwise consider opaque
    // TODO: SSE2
    if constexpr (texture_enable)
      color = color.blend8(blended_color, transparent_mask);
    else
      color = blended_color & GSVector4i::cxpr(0x7fff);
  }

  // TODO: lift out to parent?
  const GSVector4i mask_and = GSVector4i(cmd->params.GetMaskAND());
  const GSVector4i mask_or = GSVector4i(cmd->params.GetMaskOR());

  GSVector4i mask_bits_set = bg_color & mask_and; // 8000 if masked else 0000
  mask_bits_set = mask_bits_set.sra16<15>();      // ffff if masked else 0000
  preserve_mask = preserve_mask | mask_bits_set;  // ffff if preserved else 0000

  bg_color = bg_color & preserve_mask;
  color = (color | mask_or).andnot(preserve_mask);
  color = color | bg_color;

  const GSVector4i packed_color = color.pu32();
  StoreVector(start_x, y, packed_color);
}

template<bool texture_enable, bool raw_texture_enable, bool transparency_enable>
static void DrawRectangle(const GPUBackendDrawRectangleCommand* cmd)
{
  const s32 origin_x = cmd->x;
  const s32 origin_y = cmd->y;

  const GSVector4i rgba = GSVector4i(cmd->color); // RGBA | RGBA | RGBA | RGBA
  GSVector4i rg = rgba.xxxxl();                   // RGRG | RGRG | RGRG | RGRG
  GSVector4i ba = rgba.yyyyl();                   // BABA | BABA | BABA | BABA
  rg = rg.u8to16();                               // R0G0 | R0G0 | R0G0 | R0G0
  ba = ba.u8to16();                               // B0A0 | B0A0 | B0A0 | B0A0

  const GSVector4i texcoord_x = GSVector4i(cmd->texcoord & 0xFF).add32(GSVector4i::cxpr(0, 1, 2, 3));
  GSVector4i texcoord_y = GSVector4i(cmd->texcoord >> 8);

  const GSVector4i clip_left = GSVector4i(g_drawing_area.left);
  const GSVector4i clip_right = GSVector4i(g_drawing_area.right);
  const u32 width = cmd->width;

  BACKUP_VRAM();

  for (u32 offset_y = 0; offset_y < cmd->height; offset_y++)
  {
    const s32 y = origin_y + static_cast<s32>(offset_y);
    if (y < static_cast<s32>(g_drawing_area.top) || y > static_cast<s32>(g_drawing_area.bottom) ||
        (cmd->params.interlaced_rendering && cmd->params.active_line_lsb == (Truncate8(static_cast<u32>(y)) & 1u)))
    {
      continue;
    }

    GSVector4i row_texcoord_x = texcoord_x;
    GSVector4i xvec = GSVector4i(origin_x).add32(GSVector4i::cxpr(0, 1, 2, 3));
    GSVector4i wvec = GSVector4i(width).sub32(GSVector4i::cxpr(1, 2, 3, 4));

    for (u32 offset_x = 0; offset_x < width; offset_x += 4)
    {
      const s32 x = origin_x + static_cast<s32>(offset_x);

      // width test
      GSVector4i preserve_mask = wvec.lt32(GSVector4i::zero());

      // clip test, if all pixels are outside, skip
      preserve_mask = preserve_mask | xvec.lt32(clip_left);
      preserve_mask = preserve_mask | xvec.gt32(clip_right);
      if (!preserve_mask.alltrue())
      {
        ShadePixel<texture_enable, raw_texture_enable, transparency_enable, false>(
          cmd, x, y, rg, ba, row_texcoord_x, texcoord_y, preserve_mask, GSVector4i::zero());
      }

      xvec = xvec.add32(GSVector4i::cxpr(4));
      wvec = wvec.sub32(GSVector4i::cxpr(4));

      if constexpr (texture_enable)
        row_texcoord_x = row_texcoord_x.add32(GSVector4i::cxpr(4)) & GSVector4i::cxpr(0xFF);
    }

    if constexpr (texture_enable)
      texcoord_y = texcoord_y.add32(GSVector4i::cxpr(1)) & GSVector4i::cxpr(0xFF);
  }

  CHECK_VRAM(GPU_SW_Rasterizer::DrawRectangleFunctions[texture_enable][raw_texture_enable][transparency_enable](cmd));
}

#endif // USE_VECTOR

//////////////////////////////////////////////////////////////////////////
// Polygon and line rasterization ported from Mednafen
//////////////////////////////////////////////////////////////////////////

#define COORD_FBS 12
#define COORD_MF_INT(n) ((n) << COORD_FBS)
#define COORD_POST_PADDING 12

ALWAYS_INLINE_RELEASE static s64 MakePolyXFP(s32 x)
{
  return ((u64)x << 32) + ((1ULL << 32) - (1 << 11));
}

ALWAYS_INLINE_RELEASE static s64 MakePolyXFPStep(s32 dx, s32 dy)
{
  s64 ret;
  s64 dx_ex = (u64)dx << 32;

  if (dx_ex < 0)
    dx_ex -= dy - 1;

  if (dx_ex > 0)
    dx_ex += dy - 1;

  ret = dx_ex / dy;

  return (ret);
}

ALWAYS_INLINE_RELEASE static s32 GetPolyXFP_Int(s64 xfp)
{
  return (xfp >> 32);
}

template<bool shading_enable, bool texture_enable>
ALWAYS_INLINE_RELEASE static bool CalcIDeltas(i_deltas& idl, const GPUBackendDrawPolygonCommand::Vertex* A,
                                              const GPUBackendDrawPolygonCommand::Vertex* B,
                                              const GPUBackendDrawPolygonCommand::Vertex* C)
{
#define CALCIS(x, y) (((B->x - A->x) * (C->y - B->y)) - ((C->x - B->x) * (B->y - A->y)))

  s32 denom = CALCIS(x, y);

  if (!denom)
    return false;

  if constexpr (shading_enable)
  {
    idl.dr_dx = (u32)(CALCIS(r, y) * (1 << COORD_FBS) / denom) << COORD_POST_PADDING;
    idl.dr_dy = (u32)(CALCIS(x, r) * (1 << COORD_FBS) / denom) << COORD_POST_PADDING;

    idl.dg_dx = (u32)(CALCIS(g, y) * (1 << COORD_FBS) / denom) << COORD_POST_PADDING;
    idl.dg_dy = (u32)(CALCIS(x, g) * (1 << COORD_FBS) / denom) << COORD_POST_PADDING;

    idl.db_dx = (u32)(CALCIS(b, y) * (1 << COORD_FBS) / denom) << COORD_POST_PADDING;
    idl.db_dy = (u32)(CALCIS(x, b) * (1 << COORD_FBS) / denom) << COORD_POST_PADDING;
  }

  if constexpr (texture_enable)
  {
    idl.du_dx = (u32)(CALCIS(u, y) * (1 << COORD_FBS) / denom) << COORD_POST_PADDING;
    idl.du_dy = (u32)(CALCIS(x, u) * (1 << COORD_FBS) / denom) << COORD_POST_PADDING;

    idl.dv_dx = (u32)(CALCIS(v, y) * (1 << COORD_FBS) / denom) << COORD_POST_PADDING;
    idl.dv_dy = (u32)(CALCIS(x, v) * (1 << COORD_FBS) / denom) << COORD_POST_PADDING;
  }

  return true;

#undef CALCIS
}

template<bool shading_enable, bool texture_enable>
ALWAYS_INLINE_RELEASE static void AddIDeltas_DX(i_group& ig, const i_deltas& idl, u32 count = 1)
{
  if constexpr (shading_enable)
  {
    ig.r += idl.dr_dx * count;
    ig.g += idl.dg_dx * count;
    ig.b += idl.db_dx * count;
  }

  if constexpr (texture_enable)
  {
    ig.u += idl.du_dx * count;
    ig.v += idl.dv_dx * count;
  }
}

template<bool shading_enable, bool texture_enable>
ALWAYS_INLINE_RELEASE static void AddIDeltas_DY(i_group& ig, const i_deltas& idl, u32 count = 1)
{
  if constexpr (shading_enable)
  {
    ig.r += idl.dr_dy * count;
    ig.g += idl.dg_dy * count;
    ig.b += idl.db_dy * count;
  }

  if constexpr (texture_enable)
  {
    ig.u += idl.du_dy * count;
    ig.v += idl.dv_dy * count;
  }
}

#ifndef USE_VECTOR

template<bool shading_enable, bool texture_enable, bool raw_texture_enable, bool transparency_enable,
         bool dithering_enable>
ALWAYS_INLINE_RELEASE static void DrawSpan(const GPUBackendDrawCommand* cmd, s32 y, s32 x_start, s32 x_bound,
                                           i_group ig, const i_deltas& idl)
{
  if (cmd->params.interlaced_rendering && cmd->params.active_line_lsb == (Truncate8(static_cast<u32>(y)) & 1u))
    return;

  s32 x_ig_adjust = x_start;
  s32 w = x_bound - x_start;
  s32 x = TruncateGPUVertexPosition(x_start);

  if (x < static_cast<s32>(g_drawing_area.left))
  {
    s32 delta = static_cast<s32>(g_drawing_area.left) - x;
    x_ig_adjust += delta;
    x += delta;
    w -= delta;
  }

  if ((x + w) > (static_cast<s32>(g_drawing_area.right) + 1))
    w = static_cast<s32>(g_drawing_area.right) + 1 - x;

  if (w <= 0)
    return;

  AddIDeltas_DX<shading_enable, texture_enable>(ig, idl, x_ig_adjust);
  AddIDeltas_DY<shading_enable, texture_enable>(ig, idl, y);

  do
  {
    const u32 r = ig.r >> (COORD_FBS + COORD_POST_PADDING);
    const u32 g = ig.g >> (COORD_FBS + COORD_POST_PADDING);
    const u32 b = ig.b >> (COORD_FBS + COORD_POST_PADDING);
    const u32 u = ig.u >> (COORD_FBS + COORD_POST_PADDING);
    const u32 v = ig.v >> (COORD_FBS + COORD_POST_PADDING);

    ShadePixel<texture_enable, raw_texture_enable, transparency_enable, dithering_enable>(
      cmd, static_cast<u32>(x), static_cast<u32>(y), Truncate8(r), Truncate8(g), Truncate8(b), Truncate8(u),
      Truncate8(v));

    x++;
    AddIDeltas_DX<shading_enable, texture_enable>(ig, idl);
  } while (--w > 0);
}

#else // USE_VECTOR

template<bool shading_enable, bool texture_enable, bool raw_texture_enable, bool transparency_enable,
         bool dithering_enable>
ALWAYS_INLINE_RELEASE static void DrawSpan(const GPUBackendDrawCommand* cmd, s32 y, s32 x_start, s32 x_bound,
                                           i_group ig, const i_deltas& idl)
{
  if (cmd->params.interlaced_rendering && cmd->params.active_line_lsb == (Truncate8(static_cast<u32>(y)) & 1u))
    return;

  s32 x_ig_adjust = x_start;
  s32 w = x_bound - x_start;
  s32 x = TruncateGPUVertexPosition(x_start);

  if (x < static_cast<s32>(g_drawing_area.left))
  {
    s32 delta = static_cast<s32>(g_drawing_area.left) - x;
    x_ig_adjust += delta;
    x += delta;
    w -= delta;
  }

  if ((x + w) > (static_cast<s32>(g_drawing_area.right) + 1))
    w = static_cast<s32>(g_drawing_area.right) + 1 - x;

  if (w <= 0)
    return;

  // TODO: Precompute.

  const auto clip_left = GSVector4i(g_drawing_area.left);
  const auto clip_right = GSVector4i(g_drawing_area.right);

  const GSVector4i dr_dx = GSVector4i(idl.dr_dx * 4);
  const GSVector4i dg_dx = GSVector4i(idl.dg_dx * 4);
  const GSVector4i db_dx = GSVector4i(idl.db_dx * 4);
  const GSVector4i du_dx = GSVector4i(idl.du_dx * 4);
  const GSVector4i dv_dx = GSVector4i(idl.dv_dx * 4);

  // TODO: vectorize
  const GSVector4i dr_dx_offset = GSVector4i(0, idl.dr_dx, idl.dr_dx * 2, idl.dr_dx * 3);
  const GSVector4i dg_dx_offset = GSVector4i(0, idl.dg_dx, idl.dg_dx * 2, idl.dg_dx * 3);
  const GSVector4i db_dx_offset = GSVector4i(0, idl.db_dx, idl.db_dx * 2, idl.db_dx * 3);
  const GSVector4i du_dx_offset = GSVector4i(0, idl.du_dx, idl.du_dx * 2, idl.du_dx * 3);
  const GSVector4i dv_dx_offset = GSVector4i(0, idl.dv_dx, idl.dv_dx * 2, idl.dv_dx * 3);

  GSVector4i dr, dg, db;
  if constexpr (shading_enable)
  {
    dr = GSVector4i(ig.r + idl.dr_dx * x_ig_adjust).add32(dr_dx_offset);
    dg = GSVector4i(ig.g + idl.dg_dx * x_ig_adjust).add32(dg_dx_offset);
    db = GSVector4i(ig.b + idl.db_dx * x_ig_adjust).add32(db_dx_offset);
  }
  else
  {
    // precompute for flat shading
    dr = GSVector4i(ig.r >> (COORD_FBS + COORD_POST_PADDING));
    dg = GSVector4i((ig.g >> (COORD_FBS + COORD_POST_PADDING)) << 16);
    db = GSVector4i(ig.b >> (COORD_FBS + COORD_POST_PADDING));
  }

  GSVector4i du = GSVector4i(ig.u + idl.du_dx * x_ig_adjust).add32(du_dx_offset);
  GSVector4i dv = GSVector4i(ig.v + idl.dv_dx * x_ig_adjust).add32(dv_dx_offset);

  // TODO: Move to caller.
  if constexpr (shading_enable)
  {
    // TODO: vectorize multiply?
    dr = dr.add32(GSVector4i(idl.dr_dy * y));
    dg = dg.add32(GSVector4i(idl.dg_dy * y));
    db = db.add32(GSVector4i(idl.db_dy * y));
  }

  if constexpr (texture_enable)
  {
    du = du.add32(GSVector4i(idl.du_dy * y));
    dv = dv.add32(GSVector4i(idl.dv_dy * y));
  }

  const GSVector4i dither =
    GSVector4i::load<false>(&VECTOR_DITHER_MATRIX[static_cast<u32>(y) & 3][(static_cast<u32>(x) & 3) * 2]);

  GSVector4i xvec = GSVector4i(x).add32(GSVector4i::cxpr(0, 1, 2, 3));
  GSVector4i wvec = GSVector4i(w).sub32(GSVector4i::cxpr(1, 2, 3, 4));

  for (s32 count = (w + 3) / 4; count > 0; --count)
  {
    // R000 | R000 | R000 | R000
    // R0G0 | R0G0 | R0G0 | R0G0
    const GSVector4i r = shading_enable ? dr.srl32<COORD_FBS + COORD_POST_PADDING>() : dr;
    const GSVector4i g =
      shading_enable ? dg.srl32<COORD_FBS + COORD_POST_PADDING>().sll32<16>() : dg; // get G into the correct position
    const GSVector4i b = shading_enable ? db.srl32<COORD_FBS + COORD_POST_PADDING>() : db;
    const GSVector4i u = du.srl32<COORD_FBS + COORD_POST_PADDING>();
    const GSVector4i v = dv.srl32<COORD_FBS + COORD_POST_PADDING>();

    // TODO: no-sse4
    const GSVector4i rg = r.blend16<0xAA>(g);

    // mask based on what's outside the span
    auto preserve_mask = wvec.lt32(GSVector4i::zero());

    // clip test, if all pixels are outside, skip
    preserve_mask = preserve_mask | xvec.lt32(clip_left);
    preserve_mask = preserve_mask | xvec.gt32(clip_right);
    if (!preserve_mask.alltrue())
    {
      ShadePixel<texture_enable, raw_texture_enable, transparency_enable, dithering_enable>(
        cmd, static_cast<u32>(x), static_cast<u32>(y), rg, b, u, v, preserve_mask, dither);
    }

    x += 4;

    xvec = xvec.add32(GSVector4i::cxpr(4));
    wvec = wvec.sub32(GSVector4i::cxpr(4));

    if constexpr (shading_enable)
    {
      dr = dr.add32(dr_dx);
      dg = dg.add32(dg_dx);
      db = db.add32(db_dx);
    }

    if constexpr (texture_enable)
    {
      du = du.add32(du_dx);
      dv = dv.add32(dv_dx);
    }
  }
}

#endif // USE_VECTOR

template<bool shading_enable, bool texture_enable, bool raw_texture_enable, bool transparency_enable,
         bool dithering_enable>
static void DrawTriangle(const GPUBackendDrawCommand* cmd, const GPUBackendDrawPolygonCommand::Vertex* v0,
                         const GPUBackendDrawPolygonCommand::Vertex* v1, const GPUBackendDrawPolygonCommand::Vertex* v2)
{
#if 0
  const GPUBackendDrawPolygonCommand::Vertex* orig_v0 = v0;
  const GPUBackendDrawPolygonCommand::Vertex* orig_v1 = v1;
  const GPUBackendDrawPolygonCommand::Vertex* orig_v2 = v2;
#endif

  u32 core_vertex;
  {
    u32 cvtemp = 0;

    if (v1->x <= v0->x)
    {
      if (v2->x <= v1->x)
        cvtemp = (1 << 2);
      else
        cvtemp = (1 << 1);
    }
    else if (v2->x < v0->x)
      cvtemp = (1 << 2);
    else
      cvtemp = (1 << 0);

    if (v2->y < v1->y)
    {
      std::swap(v2, v1);
      cvtemp = ((cvtemp >> 1) & 0x2) | ((cvtemp << 1) & 0x4) | (cvtemp & 0x1);
    }

    if (v1->y < v0->y)
    {
      std::swap(v1, v0);
      cvtemp = ((cvtemp >> 1) & 0x1) | ((cvtemp << 1) & 0x2) | (cvtemp & 0x4);
    }

    if (v2->y < v1->y)
    {
      std::swap(v2, v1);
      cvtemp = ((cvtemp >> 1) & 0x2) | ((cvtemp << 1) & 0x4) | (cvtemp & 0x1);
    }

    core_vertex = cvtemp >> 1;
  }

  if (v0->y == v2->y)
    return;

  if (static_cast<u32>(std::abs(v2->x - v0->x)) >= MAX_PRIMITIVE_WIDTH ||
      static_cast<u32>(std::abs(v2->x - v1->x)) >= MAX_PRIMITIVE_WIDTH ||
      static_cast<u32>(std::abs(v1->x - v0->x)) >= MAX_PRIMITIVE_WIDTH ||
      static_cast<u32>(v2->y - v0->y) >= MAX_PRIMITIVE_HEIGHT)
  {
    return;
  }

  s64 base_coord = MakePolyXFP(v0->x);
  s64 base_step = MakePolyXFPStep((v2->x - v0->x), (v2->y - v0->y));
  s64 bound_coord_us;
  s64 bound_coord_ls;
  bool right_facing;

  if (v1->y == v0->y)
  {
    bound_coord_us = 0;
    right_facing = (bool)(v1->x > v0->x);
  }
  else
  {
    bound_coord_us = MakePolyXFPStep((v1->x - v0->x), (v1->y - v0->y));
    right_facing = (bool)(bound_coord_us > base_step);
  }

  if (v2->y == v1->y)
    bound_coord_ls = 0;
  else
    bound_coord_ls = MakePolyXFPStep((v2->x - v1->x), (v2->y - v1->y));

  i_deltas idl;
  if (!CalcIDeltas<shading_enable, texture_enable>(idl, v0, v1, v2))
    return;

#ifdef USE_VECTOR
  BACKUP_VRAM();
#endif

  const GPUBackendDrawPolygonCommand::Vertex* vertices[3] = {v0, v1, v2};

  i_group ig;
  if constexpr (texture_enable)
  {
    ig.u = (COORD_MF_INT(vertices[core_vertex]->u) + (1 << (COORD_FBS - 1))) << COORD_POST_PADDING;
    ig.v = (COORD_MF_INT(vertices[core_vertex]->v) + (1 << (COORD_FBS - 1))) << COORD_POST_PADDING;
  }

  ig.r = (COORD_MF_INT(vertices[core_vertex]->r) + (1 << (COORD_FBS - 1))) << COORD_POST_PADDING;
  ig.g = (COORD_MF_INT(vertices[core_vertex]->g) + (1 << (COORD_FBS - 1))) << COORD_POST_PADDING;
  ig.b = (COORD_MF_INT(vertices[core_vertex]->b) + (1 << (COORD_FBS - 1))) << COORD_POST_PADDING;

  AddIDeltas_DX<shading_enable, texture_enable>(ig, idl, -vertices[core_vertex]->x);
  AddIDeltas_DY<shading_enable, texture_enable>(ig, idl, -vertices[core_vertex]->y);

  struct TriangleHalf
  {
    u64 x_coord[2];
    u64 x_step[2];

    s32 y_coord;
    s32 y_bound;

    bool dec_mode;
  } tripart[2];

  u32 vo = 0;
  u32 vp = 0;
  if (core_vertex != 0)
    vo = 1;
  if (core_vertex == 2)
    vp = 3;

  {
    TriangleHalf* tp = &tripart[vo];
    tp->y_coord = vertices[0 ^ vo]->y;
    tp->y_bound = vertices[1 ^ vo]->y;
    tp->x_coord[right_facing] = MakePolyXFP(vertices[0 ^ vo]->x);
    tp->x_step[right_facing] = bound_coord_us;
    tp->x_coord[!right_facing] = base_coord + ((vertices[vo]->y - vertices[0]->y) * base_step);
    tp->x_step[!right_facing] = base_step;
    tp->dec_mode = vo;
  }

  {
    TriangleHalf* tp = &tripart[vo ^ 1];
    tp->y_coord = vertices[1 ^ vp]->y;
    tp->y_bound = vertices[2 ^ vp]->y;
    tp->x_coord[right_facing] = MakePolyXFP(vertices[1 ^ vp]->x);
    tp->x_step[right_facing] = bound_coord_ls;
    tp->x_coord[!right_facing] =
      base_coord + ((vertices[1 ^ vp]->y - vertices[0]->y) *
                    base_step); // base_coord + ((vertices[1].y - vertices[0].y) * base_step);
    tp->x_step[!right_facing] = base_step;
    tp->dec_mode = vp;
  }

  for (u32 i = 0; i < 2; i++)
  {
    s32 yi = tripart[i].y_coord;
    s32 yb = tripart[i].y_bound;

    u64 lc = tripart[i].x_coord[0];
    u64 ls = tripart[i].x_step[0];

    u64 rc = tripart[i].x_coord[1];
    u64 rs = tripart[i].x_step[1];

    if (tripart[i].dec_mode)
    {
      while (yi > yb)
      {
        yi--;
        lc -= ls;
        rc -= rs;

        s32 y = TruncateGPUVertexPosition(yi);

        if (y < static_cast<s32>(g_drawing_area.top))
          break;

        if (y > static_cast<s32>(g_drawing_area.bottom))
          continue;

        DrawSpan<shading_enable, texture_enable, raw_texture_enable, transparency_enable, dithering_enable>(
          cmd, y & VRAM_HEIGHT_MASK, GetPolyXFP_Int(lc), GetPolyXFP_Int(rc), ig, idl);
      }
    }
    else
    {
      while (yi < yb)
      {
        s32 y = TruncateGPUVertexPosition(yi);

        if (y > static_cast<s32>(g_drawing_area.bottom))
          break;

        if (y >= static_cast<s32>(g_drawing_area.top))
        {

          DrawSpan<shading_enable, texture_enable, raw_texture_enable, transparency_enable, dithering_enable>(
            cmd, y & VRAM_HEIGHT_MASK, GetPolyXFP_Int(lc), GetPolyXFP_Int(rc), ig, idl);
        }

        yi++;
        lc += ls;
        rc += rs;
      }
    }
  }

#ifdef USE_VECTOR
  CHECK_VRAM(
    GPU_SW_Rasterizer::DrawTriangleFunctions[shading_enable][texture_enable][raw_texture_enable][transparency_enable]
                                            [dithering_enable](cmd, orig_v0, orig_v1, orig_v2));
#endif
}

ALWAYS_INLINE_RELEASE static s64 LineDivide(s64 delta, s32 dk)
{
  delta = (u64)delta << Line_XY_FractBits;

  if (delta < 0)
    delta -= dk - 1;
  if (delta > 0)
    delta += dk - 1;

  return (delta / dk);
}

template<bool shading_enable, bool transparency_enable, bool dithering_enable>
static void DrawLine(const GPUBackendDrawLineCommand* cmd, const GPUBackendDrawLineCommand::Vertex* p0,
                     const GPUBackendDrawLineCommand::Vertex* p1)
{
  const s32 i_dx = std::abs(p1->x - p0->x);
  const s32 i_dy = std::abs(p1->y - p0->y);
  const s32 k = (i_dx > i_dy) ? i_dx : i_dy;
  if (i_dx >= MAX_PRIMITIVE_WIDTH || i_dy >= MAX_PRIMITIVE_HEIGHT)
    return;

  if (p0->x >= p1->x && k > 0)
    std::swap(p0, p1);

  line_fxp_step step;
  if (k == 0)
  {
    step.dx_dk = 0;
    step.dy_dk = 0;

    if constexpr (shading_enable)
    {
      step.dr_dk = 0;
      step.dg_dk = 0;
      step.db_dk = 0;
    }
  }
  else
  {
    step.dx_dk = LineDivide(p1->x - p0->x, k);
    step.dy_dk = LineDivide(p1->y - p0->y, k);

    if constexpr (shading_enable)
    {
      step.dr_dk = (s32)((u32)(p1->r - p0->r) << Line_RGB_FractBits) / k;
      step.dg_dk = (s32)((u32)(p1->g - p0->g) << Line_RGB_FractBits) / k;
      step.db_dk = (s32)((u32)(p1->b - p0->b) << Line_RGB_FractBits) / k;
    }
  }

  line_fxp_coord cur_point;
  cur_point.x = ((u64)p0->x << Line_XY_FractBits) | (1ULL << (Line_XY_FractBits - 1));
  cur_point.y = ((u64)p0->y << Line_XY_FractBits) | (1ULL << (Line_XY_FractBits - 1));

  cur_point.x -= 1024;

  if (step.dy_dk < 0)
    cur_point.y -= 1024;

  if constexpr (shading_enable)
  {
    cur_point.r = (p0->r << Line_RGB_FractBits) | (1 << (Line_RGB_FractBits - 1));
    cur_point.g = (p0->g << Line_RGB_FractBits) | (1 << (Line_RGB_FractBits - 1));
    cur_point.b = (p0->b << Line_RGB_FractBits) | (1 << (Line_RGB_FractBits - 1));
  }

  for (s32 i = 0; i <= k; i++)
  {
    // Sign extension is not necessary here for x and y, due to the maximum values that ClipX1 and ClipY1 can contain.
    const s32 x = (cur_point.x >> Line_XY_FractBits) & 2047;
    const s32 y = (cur_point.y >> Line_XY_FractBits) & 2047;

    if ((!cmd->params.interlaced_rendering || cmd->params.active_line_lsb != (Truncate8(static_cast<u32>(y)) & 1u)) &&
        x >= static_cast<s32>(g_drawing_area.left) && x <= static_cast<s32>(g_drawing_area.right) &&
        y >= static_cast<s32>(g_drawing_area.top) && y <= static_cast<s32>(g_drawing_area.bottom))
    {
      const u8 r = shading_enable ? static_cast<u8>(cur_point.r >> Line_RGB_FractBits) : p0->r;
      const u8 g = shading_enable ? static_cast<u8>(cur_point.g >> Line_RGB_FractBits) : p0->g;
      const u8 b = shading_enable ? static_cast<u8>(cur_point.b >> Line_RGB_FractBits) : p0->b;

      ShadePixel<false, false, transparency_enable, dithering_enable>(
        cmd, static_cast<u32>(x), static_cast<u32>(y) & VRAM_HEIGHT_MASK, r, g, b, 0, 0);
    }

    cur_point.x += step.dx_dk;
    cur_point.y += step.dy_dk;

    if constexpr (shading_enable)
    {
      cur_point.r += step.dr_dk;
      cur_point.g += step.dg_dk;
      cur_point.b += step.db_dk;
    }
  }
}

constinit const DrawRectangleFunctionTable DrawRectangleFunctions = {
  {{&DrawRectangle<false, false, false>, &DrawRectangle<false, false, true>},
   {&DrawRectangle<false, false, false>, &DrawRectangle<false, false, true>}},
  {{&DrawRectangle<true, false, false>, &DrawRectangle<true, false, true>},
   {&DrawRectangle<true, true, false>, &DrawRectangle<true, true, true>}}};

constinit const DrawLineFunctionTable DrawLineFunctions = {
  {{&DrawLine<false, false, false>, &DrawLine<false, false, true>},
   {&DrawLine<false, true, false>, &DrawLine<false, true, true>}},
  {{&DrawLine<true, false, false>, &DrawLine<true, false, true>},
   {&DrawLine<true, true, false>, &DrawLine<true, true, true>}}};

constinit const DrawTriangleFunctionTable DrawTriangleFunctions = {
  {{{{&DrawTriangle<false, false, false, false, false>, &DrawTriangle<false, false, false, false, true>},
     {&DrawTriangle<false, false, false, true, false>, &DrawTriangle<false, false, false, true, true>}},
    {{&DrawTriangle<false, false, false, false, false>, &DrawTriangle<false, false, false, false, false>},
     {&DrawTriangle<false, false, false, true, false>, &DrawTriangle<false, false, false, true, false>}}},
   {{{&DrawTriangle<false, true, false, false, false>, &DrawTriangle<false, true, false, false, true>},
     {&DrawTriangle<false, true, false, true, false>, &DrawTriangle<false, true, false, true, true>}},
    {{&DrawTriangle<false, true, true, false, false>, &DrawTriangle<false, true, true, false, false>},
     {&DrawTriangle<false, true, true, true, false>, &DrawTriangle<false, true, true, true, false>}}}},
  {{{{&DrawTriangle<true, false, false, false, false>, &DrawTriangle<true, false, false, false, true>},
     {&DrawTriangle<true, false, false, true, false>, &DrawTriangle<true, false, false, true, true>}},
    {{&DrawTriangle<true, false, false, false, false>, &DrawTriangle<true, false, false, false, false>},
     {&DrawTriangle<true, false, false, true, false>, &DrawTriangle<true, false, false, true, false>}}},
   {{{&DrawTriangle<true, true, false, false, false>, &DrawTriangle<true, true, false, false, true>},
     {&DrawTriangle<true, true, false, true, false>, &DrawTriangle<true, true, false, true, true>}},
    {{&DrawTriangle<true, true, true, false, false>, &DrawTriangle<true, true, true, false, false>},
     {&DrawTriangle<true, true, true, true, false>, &DrawTriangle<true, true, true, true, false>}}}}};

#ifdef __INTELLISENSE__
}
#endif
