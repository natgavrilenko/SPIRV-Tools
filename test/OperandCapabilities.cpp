// Copyright (c) 2015 The Khronos Group Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and/or associated documentation files (the
// "Materials"), to deal in the Materials without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Materials, and to
// permit persons to whom the Materials are furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Materials.
//
// MODIFICATIONS TO THIS FILE MAY MEAN IT NO LONGER ACCURATELY REFLECTS
// KHRONOS STANDARDS. THE UNMODIFIED, NORMATIVE VERSIONS OF KHRONOS
// SPECIFICATIONS AND HEADER INFORMATION ARE LOCATED AT
//    https://www.khronos.org/registry/
//
// THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.

// Test capability dependencies for enums.

#include "UnitSPIRV.h"

namespace {

// A test case for mapping an enum to a capability mask.
struct EnumCapabilityCase {
  spv_operand_type_t type;
  uint32_t value;
  uint64_t expected_mask;
};

using EnumCapabilityTest = ::testing::TestWithParam<EnumCapabilityCase>;

TEST_P(EnumCapabilityTest, Sample) {
  spv_operand_table operandTable;
  ASSERT_EQ(SPV_SUCCESS, spvOperandTableGet(&operandTable));
  spv_operand_desc entry;
  ASSERT_EQ(SPV_SUCCESS,
            spvOperandTableValueLookup(operandTable, GetParam().type,
                                       GetParam().value, &entry));
  EXPECT_EQ(GetParam().expected_mask, entry->capabilities);
}

#define CASE0(TYPE, VALUE) \
  { SPV_OPERAND_TYPE_##TYPE, uint32_t(Spv##VALUE), 0 }
#define CASE1(TYPE, VALUE, CAP)                    \
  {                                                \
    SPV_OPERAND_TYPE_##TYPE, uint32_t(Spv##VALUE), \
        SPV_CAPABILITY_AS_MASK(SpvCapability##CAP) \
  }
#define CASE2(TYPE, VALUE, CAP1, CAP2)                 \
  {                                                    \
    SPV_OPERAND_TYPE_##TYPE, uint32_t(Spv##VALUE),     \
        (SPV_CAPABILITY_AS_MASK(SpvCapability##CAP1) | \
         SPV_CAPABILITY_AS_MASK(SpvCapability##CAP2))  \
  }

// See SPIR-V Section 3.3 Execution Model
INSTANTIATE_TEST_CASE_P(
    ExecutionModel, EnumCapabilityTest,
    ::testing::ValuesIn(std::vector<EnumCapabilityCase>{
        CASE1(EXECUTION_MODEL, ExecutionModelVertex, Shader),
        CASE1(EXECUTION_MODEL, ExecutionModelTessellationControl, Tessellation),
        CASE1(EXECUTION_MODEL, ExecutionModelTessellationEvaluation,
              Tessellation),
        CASE1(EXECUTION_MODEL, ExecutionModelGeometry, Geometry),
        CASE1(EXECUTION_MODEL, ExecutionModelFragment, Shader),
        CASE1(EXECUTION_MODEL, ExecutionModelGLCompute, Shader),
        CASE1(EXECUTION_MODEL, ExecutionModelKernel, Kernel),
    }));

// See SPIR-V Section 3.4 Addressing Model
INSTANTIATE_TEST_CASE_P(
    AddressingModel, EnumCapabilityTest,
    ::testing::ValuesIn(std::vector<EnumCapabilityCase>{
        CASE0(ADDRESSING_MODEL, AddressingModelLogical),
        CASE1(ADDRESSING_MODEL, AddressingModelPhysical32, Addresses),
        CASE1(ADDRESSING_MODEL, AddressingModelPhysical64, Addresses),
    }));

// See SPIR-V Section 3.5 Memory Model
INSTANTIATE_TEST_CASE_P(MemoryModel, EnumCapabilityTest,
                        ::testing::ValuesIn(std::vector<EnumCapabilityCase>{
                            CASE1(MEMORY_MODEL, MemoryModelSimple, Shader),
                            CASE1(MEMORY_MODEL, MemoryModelGLSL450, Shader),
                            CASE1(MEMORY_MODEL, MemoryModelOpenCL, Kernel),
                        }));

// See SPIR-V Section 3.6 Execution Mode
INSTANTIATE_TEST_CASE_P(
    ExecutionMode, EnumCapabilityTest,
    ::testing::ValuesIn(std::vector<EnumCapabilityCase>{
        CASE1(EXECUTION_MODE, ExecutionModeInvocations, Geometry),
        CASE1(EXECUTION_MODE, ExecutionModeSpacingEqual, Tessellation),
        CASE1(EXECUTION_MODE, ExecutionModeSpacingFractionalEven, Tessellation),
        CASE1(EXECUTION_MODE, ExecutionModeSpacingFractionalOdd, Tessellation),
        CASE1(EXECUTION_MODE, ExecutionModeVertexOrderCw, Tessellation),
        CASE1(EXECUTION_MODE, ExecutionModeVertexOrderCcw, Tessellation),
        CASE1(EXECUTION_MODE, ExecutionModePixelCenterInteger, Shader),
        CASE1(EXECUTION_MODE, ExecutionModeOriginUpperLeft, Shader),
        CASE1(EXECUTION_MODE, ExecutionModeOriginLowerLeft, Shader),
        CASE1(EXECUTION_MODE, ExecutionModeEarlyFragmentTests, Shader),
        CASE1(EXECUTION_MODE, ExecutionModePointMode, Tessellation),
        CASE1(EXECUTION_MODE, ExecutionModeXfb, TransformFeedback),
        CASE1(EXECUTION_MODE, ExecutionModeDepthReplacing, Shader),
        CASE1(EXECUTION_MODE, ExecutionModeDepthGreater, Shader),
        CASE1(EXECUTION_MODE, ExecutionModeDepthLess, Shader),
        CASE1(EXECUTION_MODE, ExecutionModeDepthUnchanged, Shader),
        CASE0(EXECUTION_MODE, ExecutionModeLocalSize),
        CASE1(EXECUTION_MODE, ExecutionModeLocalSizeHint, Kernel),
        CASE1(EXECUTION_MODE, ExecutionModeInputPoints, Geometry),
        CASE1(EXECUTION_MODE, ExecutionModeInputLines, Geometry),
        CASE1(EXECUTION_MODE, ExecutionModeInputLinesAdjacency, Geometry),
        CASE2(EXECUTION_MODE, ExecutionModeTriangles, Geometry, Tessellation),
        CASE1(EXECUTION_MODE, ExecutionModeInputTrianglesAdjacency, Geometry),
        CASE1(EXECUTION_MODE, ExecutionModeQuads, Tessellation),
        CASE1(EXECUTION_MODE, ExecutionModeIsolines, Tessellation),
        CASE2(EXECUTION_MODE, ExecutionModeOutputVertices, Geometry,
              Tessellation),
        CASE1(EXECUTION_MODE, ExecutionModeOutputPoints, Geometry),
        CASE1(EXECUTION_MODE, ExecutionModeOutputLineStrip, Geometry),
        CASE1(EXECUTION_MODE, ExecutionModeOutputTriangleStrip, Geometry),
        CASE1(EXECUTION_MODE, ExecutionModeVecTypeHint, Kernel),
        CASE1(EXECUTION_MODE, ExecutionModeContractionOff, Kernel),
    }));

// See SPIR-V Section 3.7 Storage Class
INSTANTIATE_TEST_CASE_P(
    StorageClass, EnumCapabilityTest,
    ::testing::ValuesIn(std::vector<EnumCapabilityCase>{
        CASE0(STORAGE_CLASS, StorageClassUniformConstant),
        CASE1(STORAGE_CLASS, StorageClassInput, Shader),
        CASE1(STORAGE_CLASS, StorageClassUniform, Shader),
        CASE1(STORAGE_CLASS, StorageClassOutput, Shader),
        CASE0(STORAGE_CLASS, StorageClassWorkgroup),
        CASE0(STORAGE_CLASS, StorageClassCrossWorkgroup),
        CASE1(STORAGE_CLASS, StorageClassPrivate, Shader),
        CASE0(STORAGE_CLASS, StorageClassFunction),
        CASE1(STORAGE_CLASS, StorageClassGeneric, Kernel),
        CASE1(STORAGE_CLASS, StorageClassPushConstant, Shader),
        CASE1(STORAGE_CLASS, StorageClassAtomicCounter, AtomicStorage),
        CASE0(STORAGE_CLASS, StorageClassImage),
    }));

// See SPIR-V Section 3.8 Dim
INSTANTIATE_TEST_CASE_P(
    Dim, EnumCapabilityTest,
    ::testing::ValuesIn(std::vector<EnumCapabilityCase>{
        CASE1(DIMENSIONALITY, Dim1D, Sampled1D), CASE0(DIMENSIONALITY, Dim2D),
        CASE0(DIMENSIONALITY, Dim3D), CASE1(DIMENSIONALITY, DimCube, Shader),
        CASE1(DIMENSIONALITY, DimRect, SampledRect),
        CASE1(DIMENSIONALITY, DimBuffer, SampledBuffer),
        CASE1(DIMENSIONALITY, DimSubpassData, InputAttachment),
    }));

// See SPIR-V Section 3.9 Sampler Addressing Mode
INSTANTIATE_TEST_CASE_P(
    SamplerAddressingMode, EnumCapabilityTest,
    ::testing::ValuesIn(std::vector<EnumCapabilityCase>{
        CASE1(SAMPLER_ADDRESSING_MODE, SamplerAddressingModeNone, Kernel),
        CASE1(SAMPLER_ADDRESSING_MODE, SamplerAddressingModeClampToEdge,
              Kernel),
        CASE1(SAMPLER_ADDRESSING_MODE, SamplerAddressingModeClamp, Kernel),
        CASE1(SAMPLER_ADDRESSING_MODE, SamplerAddressingModeRepeat, Kernel),
        CASE1(SAMPLER_ADDRESSING_MODE, SamplerAddressingModeRepeatMirrored,
              Kernel),
    }));

// See SPIR-V Section 3.10 Sampler Filter Mode
INSTANTIATE_TEST_CASE_P(
    SamplerFilterMode, EnumCapabilityTest,
    ::testing::ValuesIn(std::vector<EnumCapabilityCase>{
        CASE1(SAMPLER_FILTER_MODE, SamplerFilterModeNearest, Kernel),
        CASE1(SAMPLER_FILTER_MODE, SamplerFilterModeLinear, Kernel),
    }));

// clang-format off
// See SPIR-V Section 3.11 Image Format
INSTANTIATE_TEST_CASE_P(
    ImageFormat, EnumCapabilityTest,
    ::testing::ValuesIn(std::vector<EnumCapabilityCase>{
        CASE0(SAMPLER_IMAGE_FORMAT, ImageFormatUnknown),
        CASE1(SAMPLER_IMAGE_FORMAT, ImageFormatRgba32f, Shader),
        CASE1(SAMPLER_IMAGE_FORMAT, ImageFormatRgba16f, Shader),
        CASE1(SAMPLER_IMAGE_FORMAT, ImageFormatR32f, Shader),
        CASE1(SAMPLER_IMAGE_FORMAT, ImageFormatRgba8, Shader),
        CASE1(SAMPLER_IMAGE_FORMAT, ImageFormatRgba8Snorm, Shader),
        CASE1(SAMPLER_IMAGE_FORMAT, ImageFormatRg32f, StorageImageExtendedFormats),
        CASE1(SAMPLER_IMAGE_FORMAT, ImageFormatRg16f, StorageImageExtendedFormats),
        CASE1(SAMPLER_IMAGE_FORMAT, ImageFormatR11fG11fB10f, StorageImageExtendedFormats),
        CASE1(SAMPLER_IMAGE_FORMAT, ImageFormatR16f, StorageImageExtendedFormats),
        CASE1(SAMPLER_IMAGE_FORMAT, ImageFormatRgba16, StorageImageExtendedFormats),
        CASE1(SAMPLER_IMAGE_FORMAT, ImageFormatRgb10A2, StorageImageExtendedFormats),
        CASE1(SAMPLER_IMAGE_FORMAT, ImageFormatRg16, StorageImageExtendedFormats),
        CASE1(SAMPLER_IMAGE_FORMAT, ImageFormatRg8, StorageImageExtendedFormats),
        CASE1(SAMPLER_IMAGE_FORMAT, ImageFormatR16, StorageImageExtendedFormats),
        CASE1(SAMPLER_IMAGE_FORMAT, ImageFormatR8, StorageImageExtendedFormats),
        CASE1(SAMPLER_IMAGE_FORMAT, ImageFormatRgba16Snorm, StorageImageExtendedFormats),
        CASE1(SAMPLER_IMAGE_FORMAT, ImageFormatRg16Snorm, StorageImageExtendedFormats),
        CASE1(SAMPLER_IMAGE_FORMAT, ImageFormatRg8Snorm, StorageImageExtendedFormats),
        CASE1(SAMPLER_IMAGE_FORMAT, ImageFormatR16Snorm, StorageImageExtendedFormats),
        CASE1(SAMPLER_IMAGE_FORMAT, ImageFormatR8Snorm, StorageImageExtendedFormats),
        CASE1(SAMPLER_IMAGE_FORMAT, ImageFormatRgba32i, Shader),
        CASE1(SAMPLER_IMAGE_FORMAT, ImageFormatRgba16i, Shader),
        CASE1(SAMPLER_IMAGE_FORMAT, ImageFormatRgba8i, Shader),
        CASE1(SAMPLER_IMAGE_FORMAT, ImageFormatR32i, Shader),
        CASE1(SAMPLER_IMAGE_FORMAT, ImageFormatRg32i, StorageImageExtendedFormats),
        CASE1(SAMPLER_IMAGE_FORMAT, ImageFormatRg16i, StorageImageExtendedFormats),
        CASE1(SAMPLER_IMAGE_FORMAT, ImageFormatRg8i, StorageImageExtendedFormats),
        CASE1(SAMPLER_IMAGE_FORMAT, ImageFormatR16i, StorageImageExtendedFormats),
        CASE1(SAMPLER_IMAGE_FORMAT, ImageFormatR8i, StorageImageExtendedFormats),
        CASE1(SAMPLER_IMAGE_FORMAT, ImageFormatRgba32ui, Shader),
        CASE1(SAMPLER_IMAGE_FORMAT, ImageFormatRgba16ui, Shader),
        CASE1(SAMPLER_IMAGE_FORMAT, ImageFormatRgba8ui, Shader),
        CASE1(SAMPLER_IMAGE_FORMAT, ImageFormatRgba8ui, Shader),
        CASE1(SAMPLER_IMAGE_FORMAT, ImageFormatRgb10a2ui, StorageImageExtendedFormats),
        CASE1(SAMPLER_IMAGE_FORMAT, ImageFormatRg32ui, StorageImageExtendedFormats),
        CASE1(SAMPLER_IMAGE_FORMAT, ImageFormatRg16ui, StorageImageExtendedFormats),
        CASE1(SAMPLER_IMAGE_FORMAT, ImageFormatRg8ui, StorageImageExtendedFormats),
        CASE1(SAMPLER_IMAGE_FORMAT, ImageFormatR16ui, StorageImageExtendedFormats),
        CASE1(SAMPLER_IMAGE_FORMAT, ImageFormatR8ui, StorageImageExtendedFormats),
    }));
// clang-format on

// See SPIR-V Section 3.12 Image Channel Order
INSTANTIATE_TEST_CASE_P(
    ImageChannelOrder, EnumCapabilityTest,
    ::testing::ValuesIn(std::vector<EnumCapabilityCase>{
        CASE1(IMAGE_CHANNEL_ORDER, ImageChannelOrderR, Kernel),
        CASE1(IMAGE_CHANNEL_ORDER, ImageChannelOrderA, Kernel),
        CASE1(IMAGE_CHANNEL_ORDER, ImageChannelOrderRG, Kernel),
        CASE1(IMAGE_CHANNEL_ORDER, ImageChannelOrderRA, Kernel),
        CASE1(IMAGE_CHANNEL_ORDER, ImageChannelOrderRGB, Kernel),
        CASE1(IMAGE_CHANNEL_ORDER, ImageChannelOrderRGBA, Kernel),
        CASE1(IMAGE_CHANNEL_ORDER, ImageChannelOrderBGRA, Kernel),
        CASE1(IMAGE_CHANNEL_ORDER, ImageChannelOrderARGB, Kernel),
        CASE1(IMAGE_CHANNEL_ORDER, ImageChannelOrderIntensity, Kernel),
        CASE1(IMAGE_CHANNEL_ORDER, ImageChannelOrderLuminance, Kernel),
        CASE1(IMAGE_CHANNEL_ORDER, ImageChannelOrderRx, Kernel),
        CASE1(IMAGE_CHANNEL_ORDER, ImageChannelOrderRGx, Kernel),
        CASE1(IMAGE_CHANNEL_ORDER, ImageChannelOrderRGBx, Kernel),
        CASE1(IMAGE_CHANNEL_ORDER, ImageChannelOrderDepth, Kernel),
        CASE1(IMAGE_CHANNEL_ORDER, ImageChannelOrderDepthStencil, Kernel),
        CASE1(IMAGE_CHANNEL_ORDER, ImageChannelOrdersRGB, Kernel),
        CASE1(IMAGE_CHANNEL_ORDER, ImageChannelOrdersRGBx, Kernel),
        CASE1(IMAGE_CHANNEL_ORDER, ImageChannelOrdersRGBA, Kernel),
        CASE1(IMAGE_CHANNEL_ORDER, ImageChannelOrdersBGRA, Kernel),
    }));

// See SPIR-V Section 3.13 Image Channel Data Type
INSTANTIATE_TEST_CASE_P(
    ImageChannelDataType, EnumCapabilityTest,
    ::testing::ValuesIn(std::vector<EnumCapabilityCase>{
        CASE1(IMAGE_CHANNEL_DATA_TYPE, ImageChannelDataTypeSnormInt8, Kernel),
        CASE1(IMAGE_CHANNEL_DATA_TYPE, ImageChannelDataTypeSnormInt16, Kernel),
        CASE1(IMAGE_CHANNEL_DATA_TYPE, ImageChannelDataTypeUnormInt8, Kernel),
        CASE1(IMAGE_CHANNEL_DATA_TYPE, ImageChannelDataTypeUnormInt16, Kernel),
        CASE1(IMAGE_CHANNEL_DATA_TYPE, ImageChannelDataTypeUnormShort565,
              Kernel),
        CASE1(IMAGE_CHANNEL_DATA_TYPE, ImageChannelDataTypeUnormShort555,
              Kernel),
        CASE1(IMAGE_CHANNEL_DATA_TYPE, ImageChannelDataTypeUnormInt101010,
              Kernel),
        CASE1(IMAGE_CHANNEL_DATA_TYPE, ImageChannelDataTypeSignedInt8, Kernel),
        CASE1(IMAGE_CHANNEL_DATA_TYPE, ImageChannelDataTypeSignedInt16, Kernel),
        CASE1(IMAGE_CHANNEL_DATA_TYPE, ImageChannelDataTypeSignedInt32, Kernel),
        CASE1(IMAGE_CHANNEL_DATA_TYPE, ImageChannelDataTypeUnsignedInt8,
              Kernel),
        CASE1(IMAGE_CHANNEL_DATA_TYPE, ImageChannelDataTypeUnsignedInt16,
              Kernel),
        CASE1(IMAGE_CHANNEL_DATA_TYPE, ImageChannelDataTypeUnsignedInt32,
              Kernel),
        CASE1(IMAGE_CHANNEL_DATA_TYPE, ImageChannelDataTypeHalfFloat, Kernel),
        CASE1(IMAGE_CHANNEL_DATA_TYPE, ImageChannelDataTypeFloat, Kernel),
        CASE1(IMAGE_CHANNEL_DATA_TYPE, ImageChannelDataTypeUnormInt24, Kernel),
        CASE1(IMAGE_CHANNEL_DATA_TYPE, ImageChannelDataTypeUnormInt101010_2,
              Kernel),
    }));

// See SPIR-V Section 3.14 Image Operands
INSTANTIATE_TEST_CASE_P(
    ImageOperands, EnumCapabilityTest,
    ::testing::ValuesIn(std::vector<EnumCapabilityCase>{
        CASE0(OPTIONAL_IMAGE, ImageOperandsMaskNone),
        CASE1(OPTIONAL_IMAGE, ImageOperandsBiasMask, Shader),
        CASE0(OPTIONAL_IMAGE, ImageOperandsLodMask),
        CASE0(OPTIONAL_IMAGE, ImageOperandsGradMask),
        CASE0(OPTIONAL_IMAGE, ImageOperandsConstOffsetMask),
        CASE1(OPTIONAL_IMAGE, ImageOperandsOffsetMask, ImageGatherExtended),
        CASE0(OPTIONAL_IMAGE, ImageOperandsConstOffsetsMask),
        CASE0(OPTIONAL_IMAGE, ImageOperandsSampleMask),
        CASE1(OPTIONAL_IMAGE, ImageOperandsMinLodMask, MinLod),
    }));

// See SPIR-V Section 3.15 FP Fast Math Mode
INSTANTIATE_TEST_CASE_P(
    FPFastMathMode, EnumCapabilityTest,
    ::testing::ValuesIn(std::vector<EnumCapabilityCase>{
        CASE0(FP_FAST_MATH_MODE, FPFastMathModeMaskNone),
        CASE1(FP_FAST_MATH_MODE, FPFastMathModeNotNaNMask, Kernel),
        CASE1(FP_FAST_MATH_MODE, FPFastMathModeNotInfMask, Kernel),
        CASE1(FP_FAST_MATH_MODE, FPFastMathModeNSZMask, Kernel),
        CASE1(FP_FAST_MATH_MODE, FPFastMathModeAllowRecipMask, Kernel),
        CASE1(FP_FAST_MATH_MODE, FPFastMathModeFastMask, Kernel),
    }));

// See SPIR-V Section 3.16 FP Rounding Mode
INSTANTIATE_TEST_CASE_P(FPRoundingMode, EnumCapabilityTest,
                        ::testing::ValuesIn(std::vector<EnumCapabilityCase>{
                            CASE1(FP_ROUNDING_MODE, FPRoundingModeRTE, Kernel),
                            CASE1(FP_ROUNDING_MODE, FPRoundingModeRTZ, Kernel),
                            CASE1(FP_ROUNDING_MODE, FPRoundingModeRTP, Kernel),
                            CASE1(FP_ROUNDING_MODE, FPRoundingModeRTN, Kernel),
                        }));

// See SPIR-V Section 3.17 Linkage Type
INSTANTIATE_TEST_CASE_P(LinkageType, EnumCapabilityTest,
                        ::testing::ValuesIn(std::vector<EnumCapabilityCase>{
                            CASE1(LINKAGE_TYPE, LinkageTypeExport, Linkage),
                            CASE1(LINKAGE_TYPE, LinkageTypeImport, Linkage),
                        }));

// See SPIR-V Section 3.18 Access Qualifier
INSTANTIATE_TEST_CASE_P(
    AccessQualifier, EnumCapabilityTest,
    ::testing::ValuesIn(std::vector<EnumCapabilityCase>{
        CASE1(ACCESS_QUALIFIER, AccessQualifierReadOnly, Kernel),
        CASE1(ACCESS_QUALIFIER, AccessQualifierWriteOnly, Kernel),
        CASE1(ACCESS_QUALIFIER, AccessQualifierReadWrite, Kernel),
    }));

// See SPIR-V Section 3.19 Function Parameter Attribute
INSTANTIATE_TEST_CASE_P(FunctionParameterAttribute, EnumCapabilityTest,
                        ::testing::ValuesIn(std::vector<EnumCapabilityCase>{
                            CASE1(FUNCTION_PARAMETER_ATTRIBUTE,
                                  FunctionParameterAttributeZext, Kernel),
                            CASE1(FUNCTION_PARAMETER_ATTRIBUTE,
                                  FunctionParameterAttributeSext, Kernel),
                            CASE1(FUNCTION_PARAMETER_ATTRIBUTE,
                                  FunctionParameterAttributeByVal, Kernel),
                            CASE1(FUNCTION_PARAMETER_ATTRIBUTE,
                                  FunctionParameterAttributeSret, Kernel),
                            CASE1(FUNCTION_PARAMETER_ATTRIBUTE,
                                  FunctionParameterAttributeNoAlias, Kernel),
                            CASE1(FUNCTION_PARAMETER_ATTRIBUTE,
                                  FunctionParameterAttributeNoCapture, Kernel),
                            CASE1(FUNCTION_PARAMETER_ATTRIBUTE,
                                  FunctionParameterAttributeNoWrite, Kernel),
                            CASE1(FUNCTION_PARAMETER_ATTRIBUTE,
                                  FunctionParameterAttributeNoReadWrite,
                                  Kernel),
                        }));

// See SPIR-V Section 3.20 Decoration
INSTANTIATE_TEST_CASE_P(
    Decoration, EnumCapabilityTest,
    ::testing::ValuesIn(std::vector<EnumCapabilityCase>{
        CASE1(DECORATION, DecorationRelaxedPrecision, Shader),
        CASE1(DECORATION, DecorationSpecId, Shader),
        CASE1(DECORATION, DecorationBlock, Shader),
        CASE1(DECORATION, DecorationBufferBlock, Shader),
        CASE1(DECORATION, DecorationRowMajor, Matrix),
        CASE1(DECORATION, DecorationColMajor, Matrix),
        CASE1(DECORATION, DecorationArrayStride, Shader),
        CASE1(DECORATION, DecorationMatrixStride, Shader),
        CASE1(DECORATION, DecorationGLSLShared, Shader),
        CASE1(DECORATION, DecorationGLSLPacked, Shader),
        CASE1(DECORATION, DecorationCPacked, Kernel),
        CASE1(DECORATION, DecorationBuiltIn, Shader),
        // Value 12 placeholder
        CASE1(DECORATION, DecorationNoPerspective, Shader),
        CASE1(DECORATION, DecorationFlat, Shader),
        CASE1(DECORATION, DecorationPatch, Tessellation),
        CASE1(DECORATION, DecorationCentroid, Shader),
        CASE1(DECORATION, DecorationSample, Shader),
        CASE1(DECORATION, DecorationInvariant, Shader),
        CASE0(DECORATION, DecorationRestrict),
        CASE0(DECORATION, DecorationAliased),
        CASE0(DECORATION, DecorationVolatile),
        CASE1(DECORATION, DecorationConstant, Kernel),
        CASE0(DECORATION, DecorationCoherent),
        CASE0(DECORATION, DecorationNonWritable),
        CASE0(DECORATION, DecorationNonReadable),
        CASE1(DECORATION, DecorationUniform, Shader),
        // Value 27 is an intentional gap in the spec numbering.
        CASE1(DECORATION, DecorationSaturatedConversion, Kernel),
        CASE1(DECORATION, DecorationStream, GeometryStreams),
        CASE1(DECORATION, DecorationLocation, Shader),
        CASE1(DECORATION, DecorationComponent, Shader),
        CASE1(DECORATION, DecorationIndex, Shader),
        CASE1(DECORATION, DecorationBinding, Shader),
        CASE1(DECORATION, DecorationDescriptorSet, Shader),
        CASE0(DECORATION, DecorationOffset),
        CASE1(DECORATION, DecorationXfbBuffer, TransformFeedback),
        CASE1(DECORATION, DecorationXfbStride, TransformFeedback),
        CASE1(DECORATION, DecorationFuncParamAttr, Kernel),
        CASE1(DECORATION, DecorationFPRoundingMode, Kernel),
        CASE1(DECORATION, DecorationFPFastMathMode, Kernel),
        CASE1(DECORATION, DecorationLinkageAttributes, Linkage),
        CASE1(DECORATION, DecorationNoContraction, Shader),
        CASE1(DECORATION, DecorationInputAttachmentIndex, InputAttachment),
        CASE1(DECORATION, DecorationAlignment, Kernel),
    }));

// See SPIR-V Section 3.21 BuiltIn
INSTANTIATE_TEST_CASE_P(
    BuiltIn, EnumCapabilityTest,
    ::testing::ValuesIn(std::vector<EnumCapabilityCase>{
        CASE1(BUILT_IN, BuiltInPosition, Shader),
        CASE1(BUILT_IN, BuiltInPointSize, Shader),
        // 2 is an intentional gap in the spec numbering.
        CASE1(BUILT_IN, BuiltInClipDistance, Shader),
        CASE1(BUILT_IN, BuiltInCullDistance, Shader),
        CASE1(BUILT_IN, BuiltInVertexId, Shader),
        CASE1(BUILT_IN, BuiltInInstanceId, Shader),
        CASE2(BUILT_IN, BuiltInPrimitiveId, Geometry, Tessellation),
        CASE2(BUILT_IN, BuiltInInvocationId, Geometry, Tessellation),
        CASE1(BUILT_IN, BuiltInLayer, Geometry),
        CASE1(BUILT_IN, BuiltInViewportIndex, Geometry),
        CASE1(BUILT_IN, BuiltInTessLevelOuter, Tessellation),
        CASE1(BUILT_IN, BuiltInTessLevelInner, Tessellation),
        CASE1(BUILT_IN, BuiltInTessCoord, Tessellation),
        CASE1(BUILT_IN, BuiltInPatchVertices, Tessellation),
        CASE1(BUILT_IN, BuiltInFragCoord, Shader),
        CASE1(BUILT_IN, BuiltInPointCoord, Shader),
        CASE1(BUILT_IN, BuiltInFrontFacing, Shader),
        CASE1(BUILT_IN, BuiltInSampleId, Shader),
        CASE1(BUILT_IN, BuiltInSamplePosition, Shader),
        CASE1(BUILT_IN, BuiltInSampleMask, Shader),
        // Value 21 intentionally missing
        CASE1(BUILT_IN, BuiltInFragDepth, Shader),
        CASE1(BUILT_IN, BuiltInHelperInvocation, Shader),
        CASE0(BUILT_IN, BuiltInNumWorkgroups),
        CASE0(BUILT_IN, BuiltInWorkgroupSize),
        CASE0(BUILT_IN, BuiltInWorkgroupId),
        CASE0(BUILT_IN, BuiltInLocalInvocationId),
        CASE0(BUILT_IN, BuiltInGlobalInvocationId),
        CASE1(BUILT_IN, BuiltInLocalInvocationIndex, Shader),
        CASE1(BUILT_IN, BuiltInWorkDim, Kernel),
        CASE1(BUILT_IN, BuiltInGlobalSize, Kernel),
        CASE1(BUILT_IN, BuiltInEnqueuedWorkgroupSize, Kernel),
        CASE1(BUILT_IN, BuiltInGlobalOffset, Kernel),
        CASE1(BUILT_IN, BuiltInGlobalLinearId, Kernel),
        // Value 35 intentionally missing
        CASE1(BUILT_IN, BuiltInSubgroupSize, Kernel),
        CASE1(BUILT_IN, BuiltInSubgroupMaxSize, Kernel),
        CASE1(BUILT_IN, BuiltInNumSubgroups, Kernel),
        CASE1(BUILT_IN, BuiltInNumEnqueuedSubgroups, Kernel),
        CASE1(BUILT_IN, BuiltInSubgroupId, Kernel),
        CASE1(BUILT_IN, BuiltInSubgroupLocalInvocationId, Kernel),
        CASE1(BUILT_IN, BuiltInVertexIndex, Shader),
        CASE1(BUILT_IN, BuiltInInstanceIndex, Shader),
    }));

// See SPIR-V Section 3.22 Selection Control
INSTANTIATE_TEST_CASE_P(
    SelectionControl, EnumCapabilityTest,
    ::testing::ValuesIn(std::vector<EnumCapabilityCase>{
        CASE0(SELECTION_CONTROL, SelectionControlMaskNone),
        CASE0(SELECTION_CONTROL, SelectionControlFlattenMask),
        CASE0(SELECTION_CONTROL, SelectionControlDontFlattenMask),
    }));

// See SPIR-V Section 3.23 Loop Control
INSTANTIATE_TEST_CASE_P(LoopControl, EnumCapabilityTest,
                        ::testing::ValuesIn(std::vector<EnumCapabilityCase>{
                            CASE0(LOOP_CONTROL, LoopControlMaskNone),
                            CASE0(LOOP_CONTROL, LoopControlUnrollMask),
                            CASE0(LOOP_CONTROL, LoopControlDontUnrollMask),
                        }));

// See SPIR-V Section 3.24 Function Control
INSTANTIATE_TEST_CASE_P(FunctionControl, EnumCapabilityTest,
                        ::testing::ValuesIn(std::vector<EnumCapabilityCase>{
                            CASE0(FUNCTION_CONTROL, FunctionControlMaskNone),
                            CASE0(FUNCTION_CONTROL, FunctionControlInlineMask),
                            CASE0(FUNCTION_CONTROL,
                                  FunctionControlDontInlineMask),
                            CASE0(FUNCTION_CONTROL, FunctionControlPureMask),
                            CASE0(FUNCTION_CONTROL, FunctionControlConstMask),
                        }));

// See SPIR-V Section 3.25 Memory Semantics <id>
INSTANTIATE_TEST_CASE_P(
    MemorySemantics, EnumCapabilityTest,
    ::testing::ValuesIn(std::vector<EnumCapabilityCase>{
        CASE0(MEMORY_SEMANTICS_ID, MemorySemanticsMaskNone),
        CASE0(MEMORY_SEMANTICS_ID, MemorySemanticsAcquireMask),
        CASE0(MEMORY_SEMANTICS_ID, MemorySemanticsReleaseMask),
        CASE0(MEMORY_SEMANTICS_ID, MemorySemanticsAcquireReleaseMask),
        CASE0(MEMORY_SEMANTICS_ID, MemorySemanticsSequentiallyConsistentMask),
        CASE1(MEMORY_SEMANTICS_ID, MemorySemanticsUniformMemoryMask, Shader),
        CASE0(MEMORY_SEMANTICS_ID, MemorySemanticsSubgroupMemoryMask),
        CASE0(MEMORY_SEMANTICS_ID, MemorySemanticsWorkgroupMemoryMask),
        CASE0(MEMORY_SEMANTICS_ID, MemorySemanticsCrossWorkgroupMemoryMask),
        CASE1(MEMORY_SEMANTICS_ID, MemorySemanticsAtomicCounterMemoryMask,
              Shader),
        CASE0(MEMORY_SEMANTICS_ID, MemorySemanticsImageMemoryMask),
    }));

// See SPIR-V Section 3.26 Memory Access
INSTANTIATE_TEST_CASE_P(
    MemoryAccess, EnumCapabilityTest,
    ::testing::ValuesIn(std::vector<EnumCapabilityCase>{
        CASE0(OPTIONAL_MEMORY_ACCESS, MemoryAccessMaskNone),
        CASE0(OPTIONAL_MEMORY_ACCESS, MemoryAccessVolatileMask),
        CASE0(OPTIONAL_MEMORY_ACCESS, MemoryAccessAlignedMask),
        CASE0(OPTIONAL_MEMORY_ACCESS, MemoryAccessNontemporalMask),
    }));

// See SPIR-V Section 3.27 Scope <id>
INSTANTIATE_TEST_CASE_P(
    Scope, EnumCapabilityTest,
    ::testing::ValuesIn(std::vector<EnumCapabilityCase>{
        CASE0(SCOPE_ID, ScopeCrossDevice), CASE0(SCOPE_ID, ScopeDevice),
        CASE0(SCOPE_ID, ScopeWorkgroup), CASE0(SCOPE_ID, ScopeSubgroup),
        CASE0(SCOPE_ID, ScopeInvocation),
    }));

// See SPIR-V Section 3.28 Group Operation
INSTANTIATE_TEST_CASE_P(
    GroupOperation, EnumCapabilityTest,
    ::testing::ValuesIn(std::vector<EnumCapabilityCase>{
        CASE1(GROUP_OPERATION, GroupOperationReduce, Kernel),
        CASE1(GROUP_OPERATION, GroupOperationInclusiveScan, Kernel),
        CASE1(GROUP_OPERATION, GroupOperationExclusiveScan, Kernel),
    }));

// See SPIR-V Section 3.29 Kernel Enqueue Flags
INSTANTIATE_TEST_CASE_P(
    KernelEnqueueFlags, EnumCapabilityTest,
    ::testing::ValuesIn(std::vector<EnumCapabilityCase>{
        CASE1(KERNEL_ENQ_FLAGS, KernelEnqueueFlagsNoWait, Kernel),
        CASE1(KERNEL_ENQ_FLAGS, KernelEnqueueFlagsWaitKernel, Kernel),
        CASE1(KERNEL_ENQ_FLAGS, KernelEnqueueFlagsWaitWorkGroup, Kernel),
    }));

// See SPIR-V Section 3.30 Kernel Profiling Info
INSTANTIATE_TEST_CASE_P(KernelProfilingInfo, EnumCapabilityTest,
                        ::testing::ValuesIn(std::vector<EnumCapabilityCase>{
                            CASE0(KERNEL_PROFILING_INFO,
                                  KernelProfilingInfoMaskNone),
                            CASE1(KERNEL_PROFILING_INFO,
                                  KernelProfilingInfoCmdExecTimeMask, Kernel),
                        }));

// See SPIR-V Section 3.31 Capability
INSTANTIATE_TEST_CASE_P(
    Capability, EnumCapabilityTest,
    ::testing::ValuesIn(std::vector<EnumCapabilityCase>{
        CASE0(CAPABILITY, CapabilityMatrix),
        CASE1(CAPABILITY, CapabilityShader, Matrix),
        CASE1(CAPABILITY, CapabilityGeometry, Shader),
        CASE1(CAPABILITY, CapabilityTessellation, Shader),
        CASE0(CAPABILITY, CapabilityAddresses),
        CASE0(CAPABILITY, CapabilityLinkage),
        CASE0(CAPABILITY, CapabilityKernel),
        CASE1(CAPABILITY, CapabilityVector16, Kernel),
        CASE1(CAPABILITY, CapabilityFloat16Buffer, Kernel),
        CASE1(CAPABILITY, CapabilityFloat16, Float16Buffer),
        CASE0(CAPABILITY, CapabilityFloat64),
        CASE0(CAPABILITY, CapabilityInt64),
        CASE1(CAPABILITY, CapabilityInt64Atomics, Int64),
        CASE1(CAPABILITY, CapabilityImageBasic, Kernel),
        CASE1(CAPABILITY, CapabilityImageReadWrite, Kernel),
        CASE1(CAPABILITY, CapabilityImageMipmap, Kernel),
        // Value 16 intentionally missing.
        CASE1(CAPABILITY, CapabilityPipes, Kernel),
        CASE0(CAPABILITY, CapabilityGroups),
        CASE1(CAPABILITY, CapabilityDeviceEnqueue, Kernel),
        CASE1(CAPABILITY, CapabilityLiteralSampler, Kernel),
        CASE1(CAPABILITY, CapabilityAtomicStorage, Shader),
        CASE0(CAPABILITY, CapabilityInt16),
        CASE1(CAPABILITY, CapabilityTessellationPointSize, Tessellation),
        CASE1(CAPABILITY, CapabilityGeometryPointSize, Geometry),
        CASE1(CAPABILITY, CapabilityImageGatherExtended, Shader),
        // Value 26 intentionally missing.
        CASE1(CAPABILITY, CapabilityStorageImageMultisample, Shader),
        CASE1(CAPABILITY, CapabilityUniformBufferArrayDynamicIndexing, Shader),
        CASE1(CAPABILITY, CapabilitySampledImageArrayDynamicIndexing, Shader),
        CASE1(CAPABILITY, CapabilityStorageBufferArrayDynamicIndexing, Shader),
        CASE1(CAPABILITY, CapabilityStorageImageArrayDynamicIndexing, Shader),
        CASE1(CAPABILITY, CapabilityClipDistance, Shader),
        CASE1(CAPABILITY, CapabilityCullDistance, Shader),
        CASE1(CAPABILITY, CapabilityImageCubeArray, SampledCubeArray),
        CASE1(CAPABILITY, CapabilitySampleRateShading, Shader),
        CASE1(CAPABILITY, CapabilityImageRect, SampledRect),
        CASE1(CAPABILITY, CapabilitySampledRect, Shader),
        CASE1(CAPABILITY, CapabilityGenericPointer, Addresses),
        CASE1(CAPABILITY, CapabilityInt8, Kernel),
        CASE1(CAPABILITY, CapabilityInputAttachment, Shader),
        CASE1(CAPABILITY, CapabilitySparseResidency, Shader),
        CASE1(CAPABILITY, CapabilityMinLod, Shader),
        CASE1(CAPABILITY, CapabilitySampled1D, Shader),
        CASE1(CAPABILITY, CapabilityImage1D, Sampled1D),
        CASE1(CAPABILITY, CapabilitySampledCubeArray, Shader),
        CASE1(CAPABILITY, CapabilitySampledBuffer, Shader),
        CASE1(CAPABILITY, CapabilityImageBuffer, SampledBuffer),
        CASE1(CAPABILITY, CapabilityImageMSArray, Shader),
        CASE1(CAPABILITY, CapabilityStorageImageExtendedFormats, Shader),
        CASE1(CAPABILITY, CapabilityImageQuery, Shader),
        CASE1(CAPABILITY, CapabilityDerivativeControl, Shader),
        CASE1(CAPABILITY, CapabilityInterpolationFunction, Shader),
        CASE1(CAPABILITY, CapabilityTransformFeedback, Shader),
        CASE1(CAPABILITY, CapabilityGeometryStreams, Geometry),
        CASE1(CAPABILITY, CapabilityStorageImageReadWithoutFormat, Shader),
        CASE1(CAPABILITY, CapabilityStorageImageWriteWithoutFormat, Shader),
    }));

#undef CASE0
#undef CASE1
#undef CASE2

}  // anonymous namespace
