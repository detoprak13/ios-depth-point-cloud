/*
See LICENSE folder for this sample's licensing information.

Abstract:
The sample app's shaders.
*/

#include <metal_stdlib>
#include <simd/simd.h>
#import "ShaderTypes.h"

using namespace metal;

// Camera's RGB vertex shader outputs
struct RGBVertexOut {
    float4 position [[position]];
    float2 texCoord;
};

// Particle vertex shader outputs and fragment shader inputs
struct ParticleVertexOut {
    float4 position [[position]];
    float pointSize [[point_size]];
    float4 color;
    float confidence;
    int meshId;
};

constexpr sampler colorSampler(mip_filter::linear, mag_filter::linear, min_filter::linear);
constant auto yCbCrToRGB = float4x4(float4(+1.0000f, +1.0000f, +1.0000f, +0.0000f),
                                    float4(+0.0000f, -0.3441f, +1.7720f, +0.0000f),
                                    float4(+1.4020f, -0.7141f, +0.0000f, +0.0000f),
                                    float4(-0.7010f, +0.5291f, -0.8860f, +1.0000f));
constant float2 viewVertices[] = { float2(-1, 1), float2(-1, -1), float2(1, 1), float2(1, -1) };
constant float2 viewTexCoords[] = { float2(0, 0), float2(0, 1), float2(1, 0), float2(1, 1) };

/// Retrieves the world position of a specified camera point with depth
static simd_float4 worldPoint(simd_float2 cameraPoint, float depth, matrix_float3x3 cameraIntrinsicsInversed, matrix_float4x4 localToWorld) {
    const auto localPoint = cameraIntrinsicsInversed * simd_float3(cameraPoint, 1) * depth;
    const auto worldPoint = localToWorld * simd_float4(localPoint, 1);
    
    return worldPoint / worldPoint.w;
}

///  Vertex shader that takes in a 2D grid-point and infers its 3D position in world-space, along with RGB and confidence
vertex void unprojectVertex(uint vertexID [[vertex_id]],
                            constant PointCloudUniforms &uniforms [[buffer(kPointCloudUniforms)]],
                            device ParticleUniforms *particleUniforms [[buffer(kParticleUniforms)]],
                            constant float2 *gridPoints [[buffer(kGridPoints)]],
                            const device MeshInfo *meshInfos [[buffer(kMeshInfos)]],
                            device atomic_uint *meshVoxels [[buffer(kMeshVoxels)]],
                            texture2d<float, access::sample> capturedImageTextureY [[texture(kTextureY)]],
                            texture2d<float, access::sample> capturedImageTextureCbCr [[texture(kTextureCbCr)]],
                            texture2d<float, access::sample> depthTexture [[texture(kTextureDepth)]],
                            texture2d<unsigned int, access::sample> confidenceTexture [[texture(kTextureConfidence)]]) {
    
    const auto gridPoint = gridPoints[vertexID];
    const auto currentPointIndex = (uniforms.pointCloudCurrentIndex + vertexID) % uniforms.maxPoints;
    const auto texCoord = gridPoint / uniforms.cameraResolution;
    // Sample the depth map to get the depth value
    const auto depth = depthTexture.sample(colorSampler, texCoord).r;
    // With a 2D point plus depth, we can now get its 3D position
    const auto position = worldPoint(gridPoint, depth, uniforms.cameraIntrinsicsInversed, uniforms.localToWorld);
    
    // Sample Y and CbCr textures to get the YCbCr color at the given texture coordinate
    const auto ycbcr = float4(capturedImageTextureY.sample(colorSampler, texCoord).r, capturedImageTextureCbCr.sample(colorSampler, texCoord.xy).rg, 1);
    const auto sampledColor = (yCbCrToRGB * ycbcr).rgb;
    // Sample the confidence map to get the confidence value
    const auto confidence = confidenceTexture.sample(colorSampler, texCoord).r;
    
    // Classify point to a mesh via AABB proximity
    int bestMesh = -1;
    float bestD2 = INFINITY;
    const float margin = 0.03; // 3cm tolerance
    for (int i = 0; i < uniforms.meshCount; ++i) {
        const float3 c = meshInfos[i].center;
        const float3 e = meshInfos[i].extent * 0.5 + float3(margin);
        const float3 p = position.xyz;
        const bool inside = (p.x >= c.x - e.x) && (p.x <= c.x + e.x)
                         && (p.y >= c.y - e.y) && (p.y <= c.y + e.y)
                         && (p.z >= c.z - e.z) && (p.z <= c.z + e.z);
        if (inside) {
            const float3 d = p - c;
            const float d2 = dot(d, d);
            if (d2 < bestD2) {
                bestD2 = d2;
                bestMesh = meshInfos[i].meshId;
            }
        }
    }

    // If classified and confidence ok, accumulate into per-mesh voxel grid
    if (bestMesh >= 0 && confidence >= 1.0 && uniforms.voxelResolution > 0) {
        // Find the matching MeshInfo by meshId to retrieve its bounds
        // Note: meshId is dense [0..N), so index by id if it matches order; otherwise linear search
        int r = uniforms.voxelResolution;
        int voxelsPerMesh = r * r * r;
        int meshIndex = -1;
        for (int i = 0; i < uniforms.meshCount; ++i) {
            if (meshInfos[i].meshId == bestMesh) { meshIndex = i; break; }
        }
        if (meshIndex >= 0) {
            const float3 c = meshInfos[meshIndex].center;
            const float3 ext = max(meshInfos[meshIndex].extent * 0.5, float3(0.05)); // clamp to avoid zero extent
            const float3 p = position.xyz;
            float3 t = saturate((p - (c - ext)) / max(ext * 2.0, float3(1e-3)));
            // Map to voxel coords [0..r-1]
            int ix = min(r - 1, int(t.x * r));
            int iy = min(r - 1, int(t.y * r));
            int iz = min(r - 1, int(t.z * r));
            int base = bestMesh * voxelsPerMesh;
            int idx = base + (iz * r + iy) * r + ix;
            atomic_fetch_add_explicit(&meshVoxels[idx], 1, memory_order_relaxed);
        }
    }

    // Write the data to the buffer
    particleUniforms[currentPointIndex].position = position.xyz;
    particleUniforms[currentPointIndex].color = sampledColor;
    particleUniforms[currentPointIndex].confidence = confidence;
    particleUniforms[currentPointIndex].meshId = bestMesh;
}

vertex RGBVertexOut rgbVertex(uint vertexID [[vertex_id]],
                              constant RGBUniforms &uniforms [[buffer(0)]]) {
    const float3 texCoord = float3(viewTexCoords[vertexID], 1) * uniforms.viewToCamera;
    
    RGBVertexOut out;
    out.position = float4(viewVertices[vertexID], 0, 1);
    out.texCoord = texCoord.xy;
    
    return out;
}

fragment float4 rgbFragment(RGBVertexOut in [[stage_in]],
                            constant RGBUniforms &uniforms [[buffer(0)]],
                            texture2d<float, access::sample> capturedImageTextureY [[texture(kTextureY)]],
                            texture2d<float, access::sample> capturedImageTextureCbCr [[texture(kTextureCbCr)]]) {
    
    const float2 offset = (in.texCoord - 0.5) * float2(1, 1 / uniforms.viewRatio) * 2;
    const float visibility = saturate(uniforms.radius * uniforms.radius - length_squared(offset));
    const float4 ycbcr = float4(capturedImageTextureY.sample(colorSampler, in.texCoord.xy).r, capturedImageTextureCbCr.sample(colorSampler, in.texCoord.xy).rg, 1);
    
    // convert and save the color back to the buffer
    const float3 sampledColor = (yCbCrToRGB * ycbcr).rgb;
    return float4(sampledColor, 1) * visibility;
}

vertex ParticleVertexOut particleVertex(uint vertexID [[vertex_id]],
                                        constant PointCloudUniforms &uniforms [[buffer(kPointCloudUniforms)]],
                                        constant ParticleUniforms *particleUniforms [[buffer(kParticleUniforms)]]) {
    
    // get point data
    const auto particleData = particleUniforms[vertexID];
    const auto position = particleData.position;
    const auto confidence = particleData.confidence;
    const auto sampledColor = particleData.color;
    const auto visibility = confidence >= uniforms.confidenceThreshold;
    
    // animate and project the point
    float4 projectedPosition = uniforms.viewProjectionMatrix * float4(position, 1.0);
    const float pointSize = max(uniforms.particleSize / max(1.0, projectedPosition.z), 2.0);
    projectedPosition /= projectedPosition.w;
    
    // prepare for output
    ParticleVertexOut out;
    out.position = projectedPosition;
    out.pointSize = pointSize;
    out.color = float4(sampledColor, visibility);
    out.confidence = confidence;
    out.meshId = particleData.meshId;
    return out;
}

static inline float3 colorFromMeshId(int meshId) {
    // Simple hash -> color mapping (pastel-like)
    uint h = uint(meshId * 747796405u + 2891336453u);
    float r = float((h >> 16) & 255u) / 255.0;
    float g = float((h >> 8) & 255u) / 255.0;
    float b = float(h & 255u) / 255.0;
    // mix with white for pastel
    return mix(float3(r, g, b), float3(1.0, 1.0, 1.0), 0.35);
}

fragment float4 particleFragment(ParticleVertexOut in [[stage_in]],
                                 const float2 coords [[point_coord]]) {
    // we draw within a circle
    const float distSquared = length_squared(coords - float2(0.5));
    if (in.color.a == 0 || distSquared > 0.25) {
        discard_fragment();
    }
    // Discard low confidence points
    if (in.confidence < 1.0) {
        discard_fragment(); // low confidence
    }
    // Color by mesh membership if available; otherwise neutral
    float3 col = (in.meshId >= 0) ? colorFromMeshId(in.meshId) : float3(0.6, 0.6, 0.6);
    return float4(col, 0.6);
}

// MARK: - Mesh rendering

struct MeshVertexOut {
    float4 position [[position]];
    float3 color;
    float2 texCoord; // screen-space UV mapped to camera
    float3 worldPos;
};

vertex MeshVertexOut meshVertex(uint vertexID [[vertex_id]],
                                constant MeshUniforms &uniforms [[buffer(kMeshUniforms)]],
                                const device float3 *positions [[buffer(1)]]) {
    MeshVertexOut out;
    const float3 pos = positions[vertexID];
    float4 world = uniforms.modelMatrix * float4(pos, 1.0);
    float4 clip = uniforms.viewProjectionMatrix * world;
    out.position = clip;
    out.worldPos = world.xyz;
    float2 view01 = (clip.xy / max(clip.w, 1e-6)) * float2(0.5, -0.5) + float2(0.5, 0.5);
    // map to camera texcoords using the same transform as RGB path
    float3 cam = float3(view01, 1) * uniforms.viewToCamera;
    out.texCoord = cam.xy;
    out.color = float3(1.0, 1.0, 1.0);
    return out;
}

fragment float4 meshFragment(MeshVertexOut in [[stage_in]],
                             constant MeshUniforms &uniforms [[buffer(kMeshUniforms)]],
                             constant PointCloudUniforms &pc [[buffer(kPointCloudUniforms)]],
                             const device uint *meshVoxels [[buffer(kMeshVoxels)]]) {
    // Default color by meshId
    float3 baseCol = colorFromMeshId(uniforms.meshId);
    int r = max(1, pc.voxelResolution);
    int voxelsPerMesh = r * r * r;
    // Compute voxel index using mesh bounds
    float3 ext = max(uniforms.boundsExtent * 0.5, float3(0.05));
    float3 t = saturate((in.worldPos - (uniforms.boundsCenter - ext)) / max(ext * 2.0, float3(1e-3)));
    int ix = min(r - 1, int(t.x * r));
    int iy = min(r - 1, int(t.y * r));
    int iz = min(r - 1, int(t.z * r));
    int base = uniforms.meshId * voxelsPerMesh;
    int idx = base + (iz * r + iy) * r + ix;
    uint count = meshVoxels[idx];
    bool covered = (int(count) >= pc.voxelThreshold);
    float3 col = covered ? baseCol : mix(baseCol, float3(0.2,0.2,0.2), 0.65);
    return float4(col, covered ? 0.6 : 0.25);
}

// Faint wireframe overlay for mesh borders
fragment float4 meshWireFragment(MeshVertexOut in [[stage_in]]) {
    // subtle dark edge
    return float4(0.0, 0.0, 0.0, 0.25);
}
