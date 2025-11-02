/*
See LICENSE folder for this sample’s licensing information.

Abstract:
Types and enums that are shared between shaders and the host app code.
*/

#ifndef ShaderTypes_h
#define ShaderTypes_h

#include <simd/simd.h>

enum TextureIndices {
    kTextureY = 0,
    kTextureCbCr = 1,
    kTextureDepth = 2,
    kTextureConfidence = 3
};

enum BufferIndices {
    kPointCloudUniforms = 0,
    kParticleUniforms = 1,
    kGridPoints = 2,
    kMeshUniforms = 3,
    kMeshInfos = 4,
    kMeshVoxels = 5,
};

struct RGBUniforms {
    matrix_float3x3 viewToCamera;
    float viewRatio;
    float radius;
};

struct PointCloudUniforms {
    matrix_float4x4 viewProjectionMatrix;
    matrix_float4x4 localToWorld;
    matrix_float3x3 cameraIntrinsicsInversed;
    simd_float2 cameraResolution;
    
    float particleSize;
    int maxPoints;
    int pointCloudCurrentIndex;
    int confidenceThreshold;
    int meshCount;
    int voxelResolution;   // per-mesh voxel grid resolution (e.g., 16)
    int voxelThreshold;    // points needed to mark a face as covered
};

struct ParticleUniforms {
    simd_float3 position;
    simd_float3 color;
    float confidence;
    int meshId; // -1 if not associated
};

struct MeshUniforms {
    matrix_float4x4 viewProjectionMatrix;
    matrix_float4x4 modelMatrix;
    matrix_float3x3 viewToCamera;
    float viewRatio;
    int meshId;
    simd_float3 boundsCenter;
    simd_float3 boundsExtent;
};

// Compact per-mesh info for point classification on GPU
struct MeshInfo {
    simd_float3 center;
    simd_float3 extent; // axis-aligned bounds extent
    int meshId;         // stable id used for coloring
    int _pad;           // alignment padding
};

#endif /* ShaderTypes_h */
