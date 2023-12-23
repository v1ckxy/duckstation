// SPDX-FileCopyrightText: 2019-2023 Connor McLaughlin <stenzek@gmail.com>
// SPDX-License-Identifier: (GPL-3.0 OR CC-BY-NC-ND-4.0)

#include "gpu_sw_rasterizer.h"
#include "gpu.h"

#include "cpuinfo.h"

#include "common/log.h"
Log_SetChannel(GPU_SW_Rasterizer);

namespace GPU_SW_Rasterizer {
// Default implementation, compatible with all ISAs.
extern const DrawRectangleFunctionTable DrawRectangleFunctions;
extern const DrawTriangleFunctionTable DrawTriangleFunctions;
extern const DrawLineFunctionTable DrawLineFunctions;

constinit const DitherLUT g_dither_lut = []() constexpr {
  DitherLUT lut = {};
  for (u32 i = 0; i < DITHER_MATRIX_SIZE; i++)
  {
    for (u32 j = 0; j < DITHER_MATRIX_SIZE; j++)
    {
      for (u32 value = 0; value < DITHER_LUT_SIZE; value++)
      {
        const s32 dithered_value = (static_cast<s32>(value) + DITHER_MATRIX[i][j]) >> 3;
        lut[i][j][value] = static_cast<u8>((dithered_value < 0) ? 0 : ((dithered_value > 31) ? 31 : dithered_value));
      }
    }
  }
  return lut;
}();

GPUDrawingArea g_drawing_area = {};
} // namespace GPU_SW_Rasterizer

// Default implementation definitions.
namespace GPU_SW_Rasterizer {
#include "gpu_sw_rasterizer.inl"
}

// Initialize with default implementation.
namespace GPU_SW_Rasterizer {
const DrawRectangleFunctionTable* SelectedDrawRectangleFunctions = &DrawRectangleFunctions;
const DrawTriangleFunctionTable* SelectedDrawTriangleFunctions = &DrawTriangleFunctions;
const DrawLineFunctionTable* SelectedDrawLineFunctions = &DrawLineFunctions;
} // namespace GPU_SW_Rasterizer

// Declare alternative implementations.
void GPU_SW_Rasterizer::SelectImplementation()
{
}
