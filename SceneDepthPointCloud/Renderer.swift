/*
 See LICENSE folder for this sample's licensing information.
 
 Abstract:
 The host app renderer.
 */

import Metal
import MetalKit
import ARKit
import Foundation
import UIKit

final class Renderer {
    // Whether recording is on
    public var isRecording = false;
    // Current folder for saving data
    public var currentFolder = ""
    // Pick every n frames (~1/sampling frequency)
    public var pickFrames = 5 // default to save 1/5 of the new frames
    public var currentFrameIndex = 0;
    // Task delegate for informing ViewController of tasks
    public weak var delegate: TaskDelegate?
    
    // Maximum number of points we store in the point cloud
    private let maxPoints = 4_000_000
    // Number of sample points on the grid
    private let numGridPoints = 500
    // Particle's size in pixels
    private let particleSize: Float = 10
    // We only use landscape orientation in this app
    private let orientation = UIInterfaceOrientation.landscapeRight
    // Camera's threshold values for detecting when the camera moves so that we can accumulate the points
    private let cameraRotationThreshold = cos(2 * .degreesToRadian)
    private let cameraTranslationThreshold: Float = pow(0.02, 2)   // (meter-squared)
    // The max number of command buffers in flight
    private let maxInFlightBuffers = 3
    
    private lazy var rotateToARCamera = Self.makeRotateToARCameraMatrix(orientation: orientation)
    private let session: ARSession
    
    // Metal objects and textures
    private let device: MTLDevice
    private let library: MTLLibrary
    private let renderDestination: RenderDestinationProvider
    private let relaxedStencilState: MTLDepthStencilState
    private let depthStencilState: MTLDepthStencilState
    private let commandQueue: MTLCommandQueue
    private lazy var unprojectPipelineState = makeUnprojectionPipelineState()!
    private lazy var rgbPipelineState = makeRGBPipelineState()!
    private lazy var particlePipelineState = makeParticlePipelineState()!
    private lazy var meshPipelineState = makeMeshPipelineState()!
    // texture cache for captured image
    private lazy var textureCache = makeTextureCache()
    private var capturedImageTextureY: CVMetalTexture?
    private var capturedImageTextureCbCr: CVMetalTexture?
    private var depthTexture: CVMetalTexture?
    private var confidenceTexture: CVMetalTexture?
    
    // Multi-buffer rendering pipeline
    private let inFlightSemaphore: DispatchSemaphore
    private var currentBufferIndex = 0
    
    // The current viewport size
    private var viewportSize = CGSize()
    // The grid of sample points
    private lazy var gridPointsBuffer = MetalBuffer<Float2>(device: device,
                                                            array: makeGridPoints(),
                                                            index: kGridPoints.rawValue, options: [])
    
    // RGB buffer
    private lazy var rgbUniforms: RGBUniforms = {
        var uniforms = RGBUniforms()
        uniforms.radius = rgbRadius
        uniforms.viewToCamera.copy(from: viewToCamera)
        uniforms.viewRatio = Float(viewportSize.width / viewportSize.height)
        return uniforms
    }()
    private var rgbUniformsBuffers = [MetalBuffer<RGBUniforms>]()
    // Point Cloud buffer
    // This is not the point cloud data, but some parameters
    private lazy var pointCloudUniforms: PointCloudUniforms = {
        var uniforms = PointCloudUniforms()
        uniforms.maxPoints = Int32(maxPoints)
        uniforms.confidenceThreshold = Int32(confidenceThreshold)
        uniforms.particleSize = particleSize
        uniforms.cameraResolution = cameraResolution
        return uniforms
    }()
    private var pointCloudUniformsBuffers = [MetalBuffer<PointCloudUniforms>]()
    private var meshUniformsBuffers = [MetalBuffer<MeshUniforms>]()
    // Particles buffer
    // Saves the point cloud data, filled by unprojectVertex func in Shaders.metal
    private var particlesBuffer: MetalBuffer<ParticleUniforms>
    private var currentPointIndex = 0
    private var currentPointCount = 0

    // Mesh storage per ARMeshAnchor
    private struct MeshBuffers {
        let vertexBuffer: MTLBuffer
        let indexBuffer: MTLBuffer
        let indexCount: Int
        let indexType: MTLIndexType
        let modelMatrix: simd_float4x4
    }
    private var meshesById: [UUID: MeshBuffers] = [:]
    private let meshQueue = DispatchQueue(label: "SceneDepthPointCloud.Renderer.meshQueue", attributes: .concurrent)
    
    // World stabilization to mitigate ARKit world rebases
    private var stabilizationTransform = matrix_identity_float4x4
    private var lastRawCameraTransformForStabilization: simd_float4x4?
    private var wasTrackingNormal = true
    
    // Camera data
    private var sampleFrame: ARFrame { session.currentFrame! }
    private lazy var cameraResolution = Float2(Float(sampleFrame.camera.imageResolution.width), Float(sampleFrame.camera.imageResolution.height))
    private lazy var viewToCamera = sampleFrame.displayTransform(for: orientation, viewportSize: viewportSize).inverted()
    private lazy var lastCameraTransform = sampleFrame.camera.transform
    
    // interfaces
    var confidenceThreshold = 1 {
        didSet {
            // apply the change for the shader
            pointCloudUniforms.confidenceThreshold = Int32(confidenceThreshold)
        }
    }
    
    var rgbRadius: Float = 1.0 {
        didSet {
            // apply the change for the shader
            rgbUniforms.radius = rgbRadius
        }
    }
    
    init(session: ARSession, metalDevice device: MTLDevice, renderDestination: RenderDestinationProvider) {
        self.session = session
        self.device = device
        self.renderDestination = renderDestination
        
        library = device.makeDefaultLibrary()!
        commandQueue = device.makeCommandQueue()!
        
        // initialize our buffers
        for _ in 0 ..< maxInFlightBuffers {
            rgbUniformsBuffers.append(.init(device: device, count: 1, index: 0))
            pointCloudUniformsBuffers.append(.init(device: device, count: 1, index: kPointCloudUniforms.rawValue))
            meshUniformsBuffers.append(.init(device: device, count: 1, index: kMeshUniforms.rawValue))
        }
        particlesBuffer = .init(device: device, count: maxPoints, index: kParticleUniforms.rawValue)
        
        // rbg does not need to read/write depth
        let relaxedStateDescriptor = MTLDepthStencilDescriptor()
        relaxedStencilState = device.makeDepthStencilState(descriptor: relaxedStateDescriptor)!
        
        // setup depth test for point cloud
        let depthStateDescriptor = MTLDepthStencilDescriptor()
        depthStateDescriptor.depthCompareFunction = .lessEqual
        depthStateDescriptor.isDepthWriteEnabled = true
        depthStencilState = device.makeDepthStencilState(descriptor: depthStateDescriptor)!
        
        inFlightSemaphore = DispatchSemaphore(value: maxInFlightBuffers)
    }
    
    func drawRectResized(size: CGSize) {
        viewportSize = size
    }
    
    private func updateCapturedImageTextures(frame: ARFrame) {
        // Create two textures (Y and CbCr) from the provided frame's captured image
        let pixelBuffer = frame.capturedImage
        guard CVPixelBufferGetPlaneCount(pixelBuffer) >= 2 else {
            return
        }
        
        capturedImageTextureY = makeTexture(fromPixelBuffer: pixelBuffer, pixelFormat: .r8Unorm, planeIndex: 0)
        capturedImageTextureCbCr = makeTexture(fromPixelBuffer: pixelBuffer, pixelFormat: .rg8Unorm, planeIndex: 1)
    }
    
    private func updateDepthTextures(frame: ARFrame) -> Bool {
        guard let depthMap = frame.sceneDepth?.depthMap,
              let confidenceMap = frame.sceneDepth?.confidenceMap else {
            return false
        }
        
        depthTexture = makeTexture(fromPixelBuffer: depthMap, pixelFormat: .r32Float, planeIndex: 0)
        confidenceTexture = makeTexture(fromPixelBuffer: confidenceMap, pixelFormat: .r8Uint, planeIndex: 0)
        
        return true
    }
    
    private func update(frame: ARFrame) {
        // frame dependent info
        let camera = frame.camera
        let cameraIntrinsicsInversed = camera.intrinsics.inverse
        let viewMatrix = camera.viewMatrix(for: orientation)
        let viewMatrixInversed = viewMatrix.inverse
        let projectionMatrix = camera.projectionMatrix(for: orientation, viewportSize: viewportSize, zNear: 0.001, zFar: 0)
        // Apply stabilization: viewProjection uses inverse(stabilization), world transforms use stabilization
        pointCloudUniforms.viewProjectionMatrix = projectionMatrix * viewMatrix * simd_inverse(stabilizationTransform)
        pointCloudUniforms.localToWorld = stabilizationTransform * (viewMatrixInversed * rotateToARCamera)
        pointCloudUniforms.cameraIntrinsicsInversed = cameraIntrinsicsInversed
    }
    
    func draw() {
        guard let currentFrame = session.currentFrame,
              let renderDescriptor = renderDestination.currentRenderPassDescriptor,
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderDescriptor) else {
            return
        }
        
        _ = inFlightSemaphore.wait(timeout: DispatchTime.distantFuture)
        commandBuffer.addCompletedHandler { [weak self] commandBuffer in
            if let self = self {
                self.inFlightSemaphore.signal()
            }
        }
        
        // update stabilization based on tracking state transitions or large pose jumps
        let isTrackingNormal: Bool
        switch currentFrame.camera.trackingState {
        case .normal:
            isTrackingNormal = true
        default:
            isTrackingNormal = false
        }
        let currentRawCam = currentFrame.camera.transform
        if let lastCam = lastRawCameraTransformForStabilization {
            // If tracking just became normal, or if there is a large sudden jump, compensate by rebasing stabilization
            let becameNormal = (!wasTrackingNormal && isTrackingNormal)
            let deltaT = distance_squared(lastCam.columns.3, currentRawCam.columns.3)
            let qa = simd_quatf(lastCam)
            let qb = simd_quatf(currentRawCam)
            let rotDelta = acos(min(1.0, max(-1.0, dot(qa.vector, qb.vector)))) * 2.0
            let largeJump = deltaT > 0.25 || rotDelta > (20.0 * .degreesToRadian) // > 0.5m or > 20 degrees
            if becameNormal || largeJump {
                stabilizationTransform = stabilizationTransform * lastCam * simd_inverse(currentRawCam)
            }
        }
        lastRawCameraTransformForStabilization = currentRawCam
        wasTrackingNormal = isTrackingNormal

        // update frame data (with stabilization)
        update(frame: currentFrame)
        updateCapturedImageTextures(frame: currentFrame)
        _ = updateDepthTextures(frame: currentFrame)
        
        // handle buffer rotating
        currentBufferIndex = (currentBufferIndex + 1) % maxInFlightBuffers
        pointCloudUniformsBuffers[currentBufferIndex][0] = pointCloudUniforms
        
        if shouldAccumulate(frame: currentFrame), updateDepthTextures(frame: currentFrame) {
            accumulatePoints(frame: currentFrame, commandBuffer: commandBuffer, renderEncoder: renderEncoder)
            
            if (checkSamplingRate()) {
                // save selected data to disk if not dropped
                autoreleasepool {
                    // selected data are deep copied into custom struct to release currentFrame
                    // if not, the pools of memory reserved for ARFrame will be full and later frames will be dropped
                    let data = ARFrameDataPack(
                        timestamp: currentFrame.timestamp,
                        cameraTransform: currentFrame.camera.transform,
                        cameraEulerAngles: currentFrame.camera.eulerAngles,
                        depthMap: duplicatePixelBuffer(input: currentFrame.sceneDepth!.depthMap),
                        smoothedDepthMap: duplicatePixelBuffer(input: currentFrame.smoothedSceneDepth!.depthMap),
                        confidenceMap: duplicatePixelBuffer(input: currentFrame.sceneDepth!.confidenceMap!),
                        capturedImage: duplicatePixelBuffer(input: currentFrame.capturedImage),
                        localToWorld: pointCloudUniforms.localToWorld,
                        cameraIntrinsicsInversed: pointCloudUniforms.cameraIntrinsicsInversed
                    )
                    saveData(frame: data)
                }
            }
        }
        
        // check and render rgb camera image
        if rgbUniforms.radius > 0 {
            var retainingTextures = [capturedImageTextureY, capturedImageTextureCbCr]
            commandBuffer.addCompletedHandler { buffer in
                retainingTextures.removeAll()
            }
            rgbUniformsBuffers[currentBufferIndex][0] = rgbUniforms
            
            renderEncoder.setDepthStencilState(relaxedStencilState)
            renderEncoder.setRenderPipelineState(rgbPipelineState)
            renderEncoder.setVertexBuffer(rgbUniformsBuffers[currentBufferIndex])
            renderEncoder.setFragmentBuffer(rgbUniformsBuffers[currentBufferIndex])
            renderEncoder.setFragmentTexture(CVMetalTextureGetTexture(capturedImageTextureY!), index: Int(kTextureY.rawValue))
            renderEncoder.setFragmentTexture(CVMetalTextureGetTexture(capturedImageTextureCbCr!), index: Int(kTextureCbCr.rawValue))
            renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        }
        
        // render particles
        renderEncoder.setDepthStencilState(depthStencilState)
        renderEncoder.setRenderPipelineState(particlePipelineState)
        renderEncoder.setVertexBuffer(pointCloudUniformsBuffers[currentBufferIndex])
        renderEncoder.setVertexBuffer(particlesBuffer)
        renderEncoder.drawPrimitives(type: .point, vertexStart: 0, vertexCount: currentPointCount)

        // decide if we should render meshes this frame (avoid relocalization jumps)
        let shouldRenderMeshes: Bool
        switch currentFrame.camera.trackingState {
        case .normal:
            shouldRenderMeshes = true
        default:
            shouldRenderMeshes = false
        }

        // take a snapshot of meshes to avoid concurrent mutation while rendering
        let meshesSnapshot: [(UUID, MeshBuffers)] = meshQueue.sync { Array(self.meshesById) }
        // meshes are baked to world space, render snapshot as-is
        let meshesToRender: [MeshBuffers] = shouldRenderMeshes ? meshesSnapshot.map { $0.1 } : []

        // render meshes if any
        if shouldRenderMeshes && !meshesToRender.isEmpty {
            renderEncoder.setRenderPipelineState(meshPipelineState)
            renderEncoder.setCullMode(.none)
            for mesh in meshesToRender {
                // Meshes are baked into world coordinates; pass camera mapping for confidence lookup
                var uniforms = MeshUniforms(viewProjectionMatrix: pointCloudUniforms.viewProjectionMatrix, modelMatrix: mesh.modelMatrix, viewToCamera: matrix_identity_float3x3, viewRatio: Float(viewportSize.width / max(1.0, viewportSize.height)))
                uniforms.viewToCamera.copy(from: viewToCamera)
                meshUniformsBuffers[currentBufferIndex][0] = uniforms
                renderEncoder.setVertexBuffer(meshUniformsBuffers[currentBufferIndex])
                renderEncoder.setVertexBuffer(mesh.vertexBuffer, offset: 0, index: 1) // raw MTLBuffer for mesh vertices
                // Bind confidence texture for coloring
                if let confTex = confidenceTexture { renderEncoder.setFragmentTexture(CVMetalTextureGetTexture(confTex), index: Int(kTextureConfidence.rawValue)) }
                renderEncoder.drawIndexedPrimitives(type: .triangle,
                                                    indexCount: mesh.indexCount,
                                                    indexType: mesh.indexType,
                                                    indexBuffer: mesh.indexBuffer,
                                                    indexBufferOffset: 0)
            }
        }
        renderEncoder.endEncoding()
        
        commandBuffer.present(renderDestination.currentDrawable!)
        commandBuffer.commit()
    }
    
    // custom struct for pulling necessary data from arframes
    struct ARFrameDataPack {
        var timestamp: Double
        var cameraTransform: simd_float4x4
        var cameraEulerAngles: simd_float3
        var depthMap: CVPixelBuffer
        var smoothedDepthMap: CVPixelBuffer
        var confidenceMap: CVPixelBuffer
        var capturedImage: CVPixelBuffer
        var localToWorld: simd_float4x4
        var cameraIntrinsicsInversed: simd_float3x3
    }
    
    /// Save data to disk in json and jpeg formats.
    private func saveData(frame: ARFrameDataPack) {
        struct DataPack: Codable {
            var timestamp: Double
            var cameraTransform: simd_float4x4 // The position and orientation of the camera in world coordinate space.
            var cameraEulerAngles: simd_float3 // The orientation of the camera, expressed as roll, pitch, and yaw values.
            var depthMap: [[Float32]]
            var smoothedDepthMap: [[Float32]]
            var confidenceMap: [[UInt8]]
            var localToWorld: simd_float4x4
            var cameraIntrinsicsInversed: simd_float3x3
        }
        
        delegate?.didStartTask()
        Task.init(priority: .utility) {
            do {
                let dataPack = await DataPack(
                    timestamp: frame.timestamp,
                    cameraTransform: frame.cameraTransform,
                    cameraEulerAngles: frame.cameraEulerAngles,
                    depthMap: cvPixelBuffer2Map(rawDepth: frame.depthMap),
                    smoothedDepthMap: cvPixelBuffer2Map(rawDepth: frame.smoothedDepthMap),
                    confidenceMap: cvPixelBuffer2Map(rawDepth: frame.confidenceMap),
                    localToWorld: frame.localToWorld,
                    cameraIntrinsicsInversed: frame.cameraIntrinsicsInversed
                )
                
                let jsonEncoder = JSONEncoder()
                jsonEncoder.outputFormatting = .prettyPrinted
                
                let encoded = try jsonEncoder.encode(dataPack)
                let encodedStr = String(data: encoded, encoding: .utf8)!
                try await saveFile(content: encodedStr, filename: "\(frame.timestamp)_\(pickFrames).json", folder: currentFolder + "/data")
                try await savePic(pic: cvPixelBuffer2UIImage(pixelBuffer: frame.capturedImage), filename: "\(frame.timestamp)_\(pickFrames).jpeg", folder: currentFolder + "/data")
                delegate?.didFinishTask()
            } catch {
                print(error.localizedDescription)
            }
        }
    }
    
    /// Save all particles to a point cloud file in ply format.
    func savePointCloud() {
        delegate?.didStartTask()
        Task.init(priority: .utility) {
            do {
                var fileToWrite = ""
                let headers = ["ply", "format ascii 1.0", "element vertex \(currentPointCount)", "property float x", "property float y", "property float z", "property uchar red", "property uchar green", "property uchar blue", "element face 0", "property list uchar int vertex_indices", "end_header"]
                for header in headers {
                    fileToWrite += header
                    fileToWrite += "\r\n"
                }
                
                for i in 0..<currentPointCount {
                    let point = particlesBuffer[i]
                    let colors = point.color
                    
                    let red = colors.x * 255.0
                    let green = colors.y * 255.0
                    let blue = colors.z * 255.0
                    
                    let pvValue = "\(point.position.x) \(point.position.y) \(point.position.z) \(Int(red)) \(Int(green)) \(Int(blue))"
                    fileToWrite += pvValue
                    fileToWrite += "\r\n"
                }
                
                try await saveFile(content: fileToWrite, filename: "\(getTimeStr()).ply", folder: currentFolder)
                
                delegate?.didFinishTask()
            } catch {
                print(error.localizedDescription)
            }
        }
    }
    
    private func shouldAccumulate(frame: ARFrame) -> Bool {
        if (!isRecording) {
            return false
        }
        let cameraTransform = frame.camera.transform
        return currentPointCount == 0
        || dot(cameraTransform.columns.2, lastCameraTransform.columns.2) <= cameraRotationThreshold
        || distance_squared(cameraTransform.columns.3, lastCameraTransform.columns.3) >= cameraTranslationThreshold
    }
    
    /// Check if the current frame should be saved or dropped based on sampling rate configuration
    private func checkSamplingRate() -> Bool {
        currentFrameIndex += 1
        return currentFrameIndex % pickFrames == 0
    }
    
    private func accumulatePoints(frame: ARFrame, commandBuffer: MTLCommandBuffer, renderEncoder: MTLRenderCommandEncoder) {
        pointCloudUniforms.pointCloudCurrentIndex = Int32(currentPointIndex)
        
        var retainingTextures = [capturedImageTextureY, capturedImageTextureCbCr, depthTexture, confidenceTexture]
        commandBuffer.addCompletedHandler { buffer in
            retainingTextures.removeAll()
        }
        
        renderEncoder.setDepthStencilState(relaxedStencilState)
        renderEncoder.setRenderPipelineState(unprojectPipelineState)
        renderEncoder.setVertexBuffer(pointCloudUniformsBuffers[currentBufferIndex])
        renderEncoder.setVertexBuffer(particlesBuffer)
        renderEncoder.setVertexBuffer(gridPointsBuffer)
        renderEncoder.setVertexTexture(CVMetalTextureGetTexture(capturedImageTextureY!), index: Int(kTextureY.rawValue))
        renderEncoder.setVertexTexture(CVMetalTextureGetTexture(capturedImageTextureCbCr!), index: Int(kTextureCbCr.rawValue))
        renderEncoder.setVertexTexture(CVMetalTextureGetTexture(depthTexture!), index: Int(kTextureDepth.rawValue))
        renderEncoder.setVertexTexture(CVMetalTextureGetTexture(confidenceTexture!), index: Int(kTextureConfidence.rawValue))
        renderEncoder.drawPrimitives(type: .point, vertexStart: 0, vertexCount: gridPointsBuffer.count)
        
        currentPointIndex = (currentPointIndex + gridPointsBuffer.count) % maxPoints
        currentPointCount = min(currentPointCount + gridPointsBuffer.count, maxPoints)
        lastCameraTransform = frame.camera.transform
    }
}

// MARK: - Metal Helpers

private extension Renderer {
    func makeMeshPipelineState() -> MTLRenderPipelineState? {
        guard let vertexFunction = library.makeFunction(name: "meshVertex"),
              let fragmentFunction = library.makeFunction(name: "meshFragment") else {
            return nil
        }
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = vertexFunction
        descriptor.fragmentFunction = fragmentFunction
        descriptor.depthAttachmentPixelFormat = renderDestination.depthStencilPixelFormat
        descriptor.colorAttachments[0].pixelFormat = renderDestination.colorPixelFormat
        descriptor.colorAttachments[0].isBlendingEnabled = true
        descriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        descriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        descriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        return try? device.makeRenderPipelineState(descriptor: descriptor)
    }
    func makeUnprojectionPipelineState() -> MTLRenderPipelineState? {
        guard let vertexFunction = library.makeFunction(name: "unprojectVertex") else {
            return nil
        }
        
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = vertexFunction
        descriptor.isRasterizationEnabled = false
        descriptor.depthAttachmentPixelFormat = renderDestination.depthStencilPixelFormat
        descriptor.colorAttachments[0].pixelFormat = renderDestination.colorPixelFormat
        
        return try? device.makeRenderPipelineState(descriptor: descriptor)
    }
    
    func makeRGBPipelineState() -> MTLRenderPipelineState? {
        guard let vertexFunction = library.makeFunction(name: "rgbVertex"),
              let fragmentFunction = library.makeFunction(name: "rgbFragment") else {
            return nil
        }
        
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = vertexFunction
        descriptor.fragmentFunction = fragmentFunction
        descriptor.depthAttachmentPixelFormat = renderDestination.depthStencilPixelFormat
        descriptor.colorAttachments[0].pixelFormat = renderDestination.colorPixelFormat
        
        return try? device.makeRenderPipelineState(descriptor: descriptor)
    }
    
    func makeParticlePipelineState() -> MTLRenderPipelineState? {
        guard let vertexFunction = library.makeFunction(name: "particleVertex"),
              let fragmentFunction = library.makeFunction(name: "particleFragment") else {
            return nil
        }
        
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = vertexFunction
        descriptor.fragmentFunction = fragmentFunction
        descriptor.depthAttachmentPixelFormat = renderDestination.depthStencilPixelFormat
        descriptor.colorAttachments[0].pixelFormat = renderDestination.colorPixelFormat
        descriptor.colorAttachments[0].isBlendingEnabled = true
        descriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        descriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        descriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        
        return try? device.makeRenderPipelineState(descriptor: descriptor)
    }
    
    /// Makes sample points on camera image, also precompute the anchor point for animation
    func makeGridPoints() -> [Float2] {
        let gridArea = cameraResolution.x * cameraResolution.y
        let spacing = sqrt(gridArea / Float(numGridPoints))
        let deltaX = Int(round(cameraResolution.x / spacing))
        let deltaY = Int(round(cameraResolution.y / spacing))
        
        var points = [Float2]()
        for gridY in 0 ..< deltaY {
            let alternatingOffsetX = Float(gridY % 2) * spacing / 2
            for gridX in 0 ..< deltaX {
                let cameraPoint = Float2(alternatingOffsetX + (Float(gridX) + 0.5) * spacing, (Float(gridY) + 0.5) * spacing)
                
                points.append(cameraPoint)
            }
        }
        
        return points
    }
    
    func makeTextureCache() -> CVMetalTextureCache {
        // Create captured image texture cache
        var cache: CVMetalTextureCache!
        CVMetalTextureCacheCreate(nil, nil, device, nil, &cache)
        
        return cache
    }
    
    func makeTexture(fromPixelBuffer pixelBuffer: CVPixelBuffer, pixelFormat: MTLPixelFormat, planeIndex: Int) -> CVMetalTexture? {
        let width = CVPixelBufferGetWidthOfPlane(pixelBuffer, planeIndex)
        let height = CVPixelBufferGetHeightOfPlane(pixelBuffer, planeIndex)
        
        var texture: CVMetalTexture? = nil
        let status = CVMetalTextureCacheCreateTextureFromImage(nil, textureCache, pixelBuffer, nil, pixelFormat, width, height, planeIndex, &texture)
        
        if status != kCVReturnSuccess {
            texture = nil
        }
        
        return texture
    }
    
    static func cameraToDisplayRotation(orientation: UIInterfaceOrientation) -> Int {
        switch orientation {
        case .landscapeLeft:
            return 180
        case .portrait:
            return 90
        case .portraitUpsideDown:
            return -90
        default:
            return 0
        }
    }
    
    static func makeRotateToARCameraMatrix(orientation: UIInterfaceOrientation) -> matrix_float4x4 {
        // flip to ARKit Camera's coordinate
        let flipYZ = matrix_float4x4(
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1] )
        
        let rotationAngle = Float(cameraToDisplayRotation(orientation: orientation)) * .degreesToRadian
        return flipYZ * matrix_float4x4(simd_quaternion(rotationAngle, Float3(0, 0, 1)))
    }
}

// MARK: - Mesh management

extension Renderer {
    func upsert(meshAnchor: ARMeshAnchor) {
        let geometry = meshAnchor.geometry
        let vertexCount = geometry.vertices.count
        
        // Pack positions into a contiguous float3 buffer
        let packedPositionsBuffer = device.makeBuffer(length: MemoryLayout<SIMD3<Float>>.stride * vertexCount, options: .storageModeShared)!
        let dstPositions = packedPositionsBuffer.contents().bindMemory(to: SIMD3<Float>.self, capacity: vertexCount)
        let srcBase = geometry.vertices.buffer.contents().advanced(by: geometry.vertices.offset)
        for i in 0..<vertexCount {
            let src = srcBase.advanced(by: i * geometry.vertices.stride).bindMemory(to: SIMD3<Float>.self, capacity: 1)
            let local = src.pointee
            let world = meshAnchor.transform * SIMD4<Float>(local.x, local.y, local.z, 1.0)
            dstPositions.advanced(by: i).pointee = SIMD3<Float>(world.x, world.y, world.z)
        }
        
        // Copy triangle indices into a contiguous index buffer
        let face = geometry.faces
        let indexCount = face.count * face.indexCountPerPrimitive
        let srcIndexBase = face.buffer.contents()
        let indexType: MTLIndexType = face.bytesPerIndex == 2 ? .uint16 : .uint32
        let indexBuffer: MTLBuffer
        if indexType == .uint16 {
            let length = indexCount * MemoryLayout<UInt16>.stride
            indexBuffer = device.makeBuffer(length: length, options: .storageModeShared)!
            let dst = indexBuffer.contents().bindMemory(to: UInt16.self, capacity: indexCount)
            let src = srcIndexBase.bindMemory(to: UInt16.self, capacity: indexCount)
            dst.assign(from: src, count: indexCount)
        } else {
            let length = indexCount * MemoryLayout<UInt32>.stride
            indexBuffer = device.makeBuffer(length: length, options: .storageModeShared)!
            let dst = indexBuffer.contents().bindMemory(to: UInt32.self, capacity: indexCount)
            let src = srcIndexBase.bindMemory(to: UInt32.self, capacity: indexCount)
            dst.assign(from: src, count: indexCount)
        }
        
        let mesh = MeshBuffers(vertexBuffer: packedPositionsBuffer,
                               indexBuffer: indexBuffer,
                               indexCount: indexCount,
                               indexType: indexType,
                               modelMatrix: matrix_identity_float4x4)
        meshQueue.async(flags: .barrier) {
            self.meshesById[meshAnchor.identifier] = mesh
        }
    }
    
    func removeMesh(anchorIdentifier: UUID) {
        meshQueue.async(flags: .barrier) {
            self.meshesById.removeValue(forKey: anchorIdentifier)
        }
    }
}
