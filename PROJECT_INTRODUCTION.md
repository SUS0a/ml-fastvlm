# FastVLM 项目介绍

## 项目概述

FastVLM 是一个高效的视觉语言模型（Vision Language Model, VLM）项目，发表于 CVPR 2025。该项目由 Apple 开发，旨在实现设备端（On-device）的高性能视觉问答（Visual Question Answering, VQA）功能，特别优化了在移动设备（如 iPhone、iPad）和 Mac 上的推理性能。

## 核心用途

1. **视觉问答（VQA）**：基于图像内容回答自然语言问题
2. **图像描述生成**：自动生成图像的文字描述
3. **实时视频流理解**：通过摄像头实时捕获画面并进行理解分析
4. **移动端 AI 应用**：在 iOS/macOS 设备上实现本地化、隐私保护的视觉理解

## 技术栈

### 后端/模型训练（Python）

- **深度学习框架**：
  - PyTorch 2.6.0
  - TorchVision 0.21.0
  
- **模型架构**：
  - Transformers 4.48.3（HuggingFace）
  - LLaVA 框架（视觉语言模型基础架构）
  - Qwen2 语言模型（0.5B / 1.5B / 7B 变体）
  
- **视觉编码器**：
  - FastViTHD（自研混合视觉编码器）
  - CLIP / MobileCLIP（传统视觉编码器）
  
- **优化与加速**：
  - PEFT（Parameter-Efficient Fine-Tuning）
  - BitsAndBytes（量化）
  - DeepSpeed 0.13.1（分布式训练）
  - Flash Attention 2
  
- **部署工具**：
  - CoreML Tools 8.2（模型转换）
  - Einops（张量操作）

### 前端/移动应用（Swift）

- **开发语言**：Swift
- **支持平台**：iOS 18.2+, macOS 15.2+
- **关键框架**：
  - MLX（Apple Silicon 机器学习推理框架）
  - MLXFast / MLXNN / MLXVLM（MLX 扩展）
  - CoreML（iOS/macOS 机器学习框架）
  - SwiftUI（用户界面）
  - AVFoundation（相机访问与视频处理）
  - CoreImage（图像处理）

## 关键技术点

### 1. FastViTHD 视觉编码器

- **创新点**：混合式架构，输出更少的 token，显著减少高分辨率图像的编码时间
- **性能优势**：
  - 最小变体比 LLaVA-OneVision-0.5B 快 85 倍 TTFT（Time-to-First-Token）
  - 视觉编码器体积小 3.4 倍
  - 7B 变体比 Cambrian-1-8B 快 7.9 倍 TTFT

### 2. 模型量化与压缩

- 支持 FP16、INT8、INT4 量化
- 不同量化级别适配不同设备性能需求
- 通过 CoreML Tools 进行模型转换与优化

### 3. 设备端推理

- 完全在本地设备运行，保障隐私和安全
- 利用 Apple Silicon 的神经引擎（Neural Engine）
- 实时显示 Time-to-First-Token（TTFT）性能指标

### 4. 多阶段训练

- **Stage 2**：预训练阶段，多模态对齐
- **Stage 3**：微调阶段，指令跟随优化

### 5. 实时视频流处理

- 支持连续模式和单次捕获模式
- 异步视频帧处理（AsyncStream）
- 可配置帧率控制

## 调用链与架构流程

### Python 推理调用链（predict.py）

```
用户输入（图像 + 文本提示）
    ↓
[1] 加载预训练模型
    - load_pretrained_model()
        → AutoTokenizer.from_pretrained()
        → LlavaLlamaForCausalLM / LlavaMistralForCausalLM 等
        → load_state_dict（加载视觉编码器和投影器权重）
    ↓
[2] 图像预处理
    - process_images()
        → image_processor（CLIP/MobileCLIP 处理器）
        → 归一化、缩放、padding
    ↓
[3] 文本预处理
    - 构建对话模板（conversation template）
    - tokenizer_image_token()
        → 插入 IMAGE_TOKEN_INDEX
        → 生成 input_ids
    ↓
[4] 模型推理
    - LlavaMetaModel.prepare_inputs_labels_for_multimodal()
        → vision_tower(images) [视觉编码器]
        → mm_projector(image_features) [多模态投影器]
        → 合并文本和视觉 embeddings
    ↓
    - LLM 解码（Qwen2ForCausalLM）
        → 自回归生成
        → 应用 temperature、top_p 等采样策略
    ↓
[5] 输出结果
    - tokenizer.decode()
        → 生成文本答案
```

### Swift 移动端调用链（app/）

```
[用户界面层] ContentView.swift
    ↓
    - 相机控制：selectedCameraType (continuous/single)
    - 文本输入：prompt + promptSuffix
    - 按钮点击：拍照或启动连续模式
    ↓
[相机层] CameraController.swift
    ↓
    - AVCaptureSession 初始化
    - AVCaptureDevice 设备选择（前置/后置摄像头）
    - captureOutput() 捕获视频帧（CMSampleBuffer）
    ↓
    - 通过 AsyncStream 传递帧到 ContentView
    ↓
[模型层] FastVLMModel.swift
    ↓
[6] 模型加载
    - load()
        → ModelContainer.perform()
            → FastVLM.load() [从 MLX 格式加载]
                → 读取 config.json
                → 加载 weights（safetensors 格式）
                → 初始化 Qwen2VL 模型
    ↓
[7] 图像处理
    - resizeImagePixelBuffer()
        → CoreImage 处理
        → 缩放至模型输入尺寸
    ↓
[8] 模型推理
    - FastVLM.generate()
        → prepareInputs()
            → tokenize(prompt)
            → processImages(pixelBuffer) [FastVLM.swift]
                → VisionEncoder.forward()
                    → FastViTHD 编码
                    → 输出 visual tokens
                → VisionProjector.forward()
                    → 投影到 LLM 特征空间
        ↓
        → LanguageModel.forward() [Qwen2ForCausalLM]
            → 应用 Multimodal Rotary Position Embedding
            → Transformer 层处理
                → MultiHeadAttention
                → FeedForward
            → LM Head 预测下一个 token
        ↓
        → 自回归解码循环
            → 采样 / argmax
            → 更新 KV Cache
            → 逐 token 生成
    ↓
[9] 结果展示
    - 更新 FastVLMModel.output
    - 显示 promptTime（TTFT）
    - ContentView 实时更新 UI
```

### 训练调用链（llava/train/）

```
[数据准备]
    - 图像 + 文本对（JSON 格式）
    - DataCollator 批处理
    ↓
[模型初始化]
    - LlavaLlamaForCausalLM.from_pretrained()
        → language_model（预训练 LLM）
        → vision_tower（视觉编码器）
        → mm_projector（多模态投影器，可训练）
    ↓
[训练循环]
    - llava_trainer.py (基于 HuggingFace Trainer)
        ↓
        - forward pass
            → encode_images(vision_tower)
            → project_features(mm_projector)
            → merge_with_text_embeddings()
            → llm_forward()
        ↓
        - 计算损失（Cross-Entropy）
            → 只计算文本回答部分的损失
            → IGNORE_INDEX 遮蔽非回答部分
        ↓
        - backward pass
            → DeepSpeed / FSDP 分布式优化
            → LoRA / 全参数微调
    ↓
[保存模型]
    - save_model()
        → PyTorch checkpoint (.bin/.safetensors)
```

### 模型导出调用链（model_export/）

```
[PyTorch Checkpoint]
    ↓
[导出视觉编码器]
    - export_vision_encoder.py
        → 提取 vision_tower 权重
        → 转换为 CoreML 格式
        → 保存 .mlpackage
    ↓
[导出 LLM]
    - mlx-vlm convert（使用 fastvlm_mlx-vlm.patch）
        → 加载 PyTorch 权重
        → 转换为 MLX 格式（safetensors）
        → 应用量化（FP16/INT8/INT4）
        → 保存到 mlx_model/
    ↓
[移动端模型]
    - 放置在 app/FastVLM.mlxpackage/
    - Xcode 自动打包进 .app
```

## Swift 项目结构详解

FastVLM iOS/macOS 应用采用模块化架构，分为核心推理库、用户界面层和视频捕获层三大部分。

### 整体架构图

```
app/
├── Configuration/           # 构建配置
├── FastVLM/                # 核心推理引擎（Swift Package）
├── FastVLM App/            # 用户界面应用
├── Video/                  # 视频捕获模块
├── FastVLM.xcodeproj/      # Xcode 项目文件
└── get_pretrained_mlx_model.sh  # 模型下载脚本
```

### 1. Configuration/ - 构建配置

#### Build.xcconfig
Xcode 构建配置文件，定义编译选项和路径。

**关键配置：**
- SDK 版本要求
- 最低系统版本（iOS 18.2+, macOS 15.2+）
- 链接器标志
- 搜索路径

---

### 2. FastVLM/ - 核心模型推理库

这是一个独立的 Swift 模块，封装了所有与模型推理相关的逻辑，可以作为库被其他应用导入。

#### FastVLM.h
```swift
// Objective-C 桥接头文件
// 用于混合 Swift 和 Objective-C/C++ 代码
// 导出模块供其他 target 使用
```

#### FastVLM.swift (693 行)
**核心模型实现文件，包含完整的 FastVLM 架构。**

**主要组件：**

##### (1) 命名空间：`Language`
```swift
private enum Language {
    // 语言模型相关组件
    class Attention: Module { ... }           // 多头注意力机制
    class MLP: Module { ... }                 // 前馈神经网络
    class TransformerBlock: Module { ... }    // Transformer 层
    class Qwen2Model: Module { ... }          // Qwen2 基础模型
    class LanguageModel: Module { ... }       // 完整的语言模型
}
```

**关键功能：**
- `applyMultimodalRotaryPositionEmbedding()`: 多模态旋转位置编码
  - 处理 [时间, 高度, 宽度] 三维位置信息
  - 对视觉和文本 tokens 应用不同的位置编码策略
  
- `Attention` 类：
  ```swift
  - heads: 8 个注意力头
  - kvHeads: KV Cache 头数
  - headDim: 每个头的维度
  - q_proj, k_proj, v_proj, o_proj: Query/Key/Value/Output 投影
  - rotaryEmbedding: RoPE 位置编码
  ```

- `TransformerBlock`: 
  ```swift
  - self_attn: Self-Attention 层
  - mlp: Feed-Forward 网络
  - input_layernorm, post_attention_layernorm: 归一化层
  ```

##### (2) 命名空间：`Vision`
```swift
private enum Vision {
    // 视觉编码相关组件
    class VisionModelCoreML { ... }    // CoreML 加速的视觉模型
    class VisionModel: Module { ... }   // MLX 视觉模型包装器
}
```

**VisionModelCoreML 关键方法：**
```swift
func encode(_ hiddenStates: MLXArray) -> MLXArray {
    // 1. 将 MLXArray 转换为 MLMultiArray
    // 2. 调用 CoreML 模型进行推理
    // 3. 输出 visual tokens [1, 256, 3072]
}
```

**作用：**
- 使用 CoreML 加速 FastViTHD 视觉编码器
- 大幅提升 iOS/Mac 设备上的推理速度
- 利用 Neural Engine 硬件加速

##### (3) FastVLMProcessor 类
```swift
public class FastVLMProcessor: UserInputProcessor {
    // 图像预处理器
    func preprocess(image: CIImage) -> (MLXArray, THW)
}
```

**处理流程：**
1. 调整图像尺寸（shortest edge = 336px）
2. 双三次插值重采样
3. CLIP 标准归一化：
   ```swift
   mean = [0.48145466, 0.4578275, 0.40821073]
   std = [0.26862954, 0.26130258, 0.27577711]
   ```
4. 转换为 MLXArray 格式

##### (4) FastVLM 主类
```swift
public class FastVLM: Module, VLMModel {
    @ModuleInfo var visionModel: Vision.VisionModel
    @ModuleInfo var languageModel: Language.LanguageModel
    @ModuleInfo var multiModalProjector: FastVLMMultiModalProjector
    
    // 核心方法
    func callAsFunction(_ inputs: LMInput, cache: [KVCache]?) -> LMOutput
}
```

**推理流程：**
```swift
// 1. 准备输入（图像 + 文本）
let (inputEmbedding, mask) = prepare(inputs, cache: cache)

// 2. 视觉编码
let visionFeatures = visionModel(imagePixels, gridThw)

// 3. 多模态投影
let projectedFeatures = multiModalProjector(visionFeatures)

// 4. 合并文本和视觉 embeddings
let inputEmbedding = merge(textEmbeddings, projectedFeatures)

// 5. 语言模型推理
let output = languageModel(inputEmbedding, cache: cache)
```

##### (5) FastVLMMultiModalProjector
```swift
public class FastVLMMultiModalProjector: Module {
    @ModuleInfo var linear1: Linear  // [3072] -> [4096]
    @ModuleInfo var linear2: Linear  // [4096] -> [896/1536/3584]
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return linear2(gelu(linear1(x)))
    }
}
```

##### (6) 配置结构体
```swift
public struct FastVLMConfiguration: Codable {
    struct VisionConfiguration { ... }
    struct TextConfiguration {
        hiddenSize: 896/1536/3584  // 隐藏层维度
        hiddenLayers: 24/28/28     // Transformer 层数
        attentionHeads: 8/12/28    // 注意力头数
        vocabularySize: 151936     // 词汇表大小
        ropeTheta: 1_000_000       // RoPE 参数
        ropeScaling: ["mrope_section": [2, 1, 1]]
    }
}
```

#### MediaProcessingExtensions.swift (173 行)
**图像和视频处理工具类。**

**主要方法：**

```swift
enum MediaProcessingExtensions {
    // 1. 图像缩放和裁剪
    static func apply(_ image: CIImage, processing: UserInput.Processing?) -> CIImage
    static func centerCrop(_ image: CIImage, size: CGSize) -> CIImage
    static func fitIn(_ size: CGSize, shortestEdge: Int) -> CGSize
    
    // 2. 图像重采样
    static func resampleBicubic(_ image: CIImage, to size: CGSize) -> CIImage
    static func resampleLanczos(_ image: CIImage, to size: CGSize) -> CIImage
    
    // 3. 图像归一化（CLIP 标准）
    static func normalized(
        _ image: CIImage, 
        mean: [Float], 
        std: [Float]
    ) -> CIImage
    
    // 4. 图像格式转换
    static func toMLXArray(_ image: CIImage) -> MLXArray
    static func toCGImage(_ ciImage: CIImage) -> CGImage?
}
```

**技术实现：**
- 使用 CoreImage 的 `CIFilter` 进行高效图像处理
- 使用 Accelerate 框架进行向量化运算
- 支持 GPU 加速的图像变换

---

### 3. FastVLM App/ - 用户界面应用

#### FastVLMApp.swift (16 行)
**应用程序入口点。**

```swift
@main
struct FastVLMApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
```

- 使用 `@main` 属性标记应用入口
- 采用 SwiftUI 的 `Scene` 架构
- 创建主窗口并加载 `ContentView`

#### ContentView.swift (454 行)
**主界面视图，应用的核心交互层。**

**状态管理：**
```swift
@State private var camera = CameraController()
@State private var model = FastVLMModel()
@State private var framesToDisplay: AsyncStream<CVImageBuffer>?
@State private var prompt = "Describe the image in English."
@State private var promptSuffix = "Output should be brief, about 15 words or less."
@State private var selectedCameraType: CameraType = .continuous
@State private var isEditingPrompt: Bool = false
@State private var isShowingInfo: Bool = false
```

**布局结构：**
```
ContentView
├── VStack (主容器)
│   ├── VideoFrameView (相机预览)
│   │   └── 实时视频流显示
│   ├── HStack (控制按钮)
│   │   ├── Picker (相机模式选择)
│   │   ├── CameraControlsView (相机切换)
│   │   └── Capture Button (拍照/开始)
│   ├── TextField (提示词输入)
│   ├── Text (模型输出显示)
│   └── HStack (状态栏)
│       ├── 状态指示器
│       └── TTFT 时间显示
├── Toolbar
│   ├── Info 按钮
│   └── Prompts 菜单
└── Sheet (Info 视图)
```

**核心功能：**

##### (1) 相机模式管理
```swift
enum CameraType {
    case continuous  // 连续模式：实时处理每一帧
    case single      // 单次模式：点击拍照一次
}
```

##### (2) 视频帧分发
```swift
private func distributeVideoFrames(_ frames: AsyncStream<CVImageBuffer>) -> AsyncStream<CVImageBuffer> {
    // 从相机接收帧
    // 根据模式分发：
    //   - continuous: 持续发送每帧
    //   - single: 等待用户触发
}
```

##### (3) 推理触发
```swift
private func startInference(_ frame: CVImageBuffer) {
    // 1. 将 CVImageBuffer 转换为 CIImage
    // 2. 构建 UserInput（图像 + 文本提示）
    // 3. 调用 model.generate()
    // 4. 实时更新 UI 显示生成的文本
}
```

##### (4) UI 更新机制
```swift
// 使用 AsyncStream 实现异步 UI 更新
for await token in generationStream {
    model.output += tokenizer.decode(token)
    // SwiftUI 自动重新渲染
}
```

**平台适配：**
```swift
#if os(iOS)
    // iOS 特定布局
    .navigationBarTitleDisplayMode(.inline)
    .toolbar { ... }
#elseif os(macOS)
    // macOS 特定布局
    .frame(minWidth: 600, minHeight: 800)
#endif
```

#### FastVLMModel.swift (191 行)
**模型管理器，负责模型生命周期和推理调度。**

**核心属性：**
```swift
@Observable
@MainActor
class FastVLMModel {
    // 状态管理
    public var running = false
    public var output = ""
    public var promptTime: String = ""
    
    // 加载状态
    enum LoadState {
        case idle
        case loaded(ModelContainer)
    }
    private var loadState = LoadState.idle
    
    // 推理状态
    enum EvaluationState {
        case idle
        case processingPrompt
        case generatingResponse
    }
    public var evaluationState = EvaluationState.idle
    
    // 参数配置
    let generateParameters = GenerateParameters(temperature: 0.0)
    let maxTokens = 240
    let displayEveryNTokens = 4
}
```

**核心方法：**

##### (1) 模型加载
```swift
public func load() async {
    // 1. 设置 GPU 缓存限制
    MLX.GPU.set(cacheLimit: 20 * 1024 * 1024)
    
    // 2. 从 MLX 包加载模型
    let modelContainer = try await VLMModelFactory.shared.loadContainer(
        configuration: modelConfiguration
    )
    
    // 3. 更新加载状态
    loadState = .loaded(modelContainer)
}
```

##### (2) 推理生成
```swift
public func generate(_ userInput: UserInput) async -> Task<Void, Never> {
    // 1. 加载模型
    let modelContainer = try await _load()
    
    // 2. 图像预处理
    evaluationState = .processingPrompt
    let processedInput = try modelContainer.processor.preprocess(userInput)
    
    // 3. 开始生成
    evaluationState = .generatingResponse
    let startTime = Date()
    
    // 4. 流式生成
    for await result in modelContainer.perform(input: processedInput) {
        switch result {
        case .token(let token):
            // 解码并显示 token
            output += modelContainer.tokenizer.decode(token: token)
            
            // 记录 TTFT
            if output.isEmpty {
                let ttft = Date().timeIntervalSince(startTime)
                promptTime = String(format: "%.3fs", ttft)
            }
            
        case .complete:
            // 生成完成
            evaluationState = .idle
        }
    }
}
```

##### (3) 任务管理
```swift
private var currentTask: Task<Void, Never>?

public func cancelGeneration() {
    currentTask?.cancel()
    running = false
    evaluationState = .idle
}
```

**性能优化：**
- 每 N tokens 更新一次 UI（displayEveryNTokens=4）
- 使用 KV Cache 避免重复计算
- 异步加载模型避免阻塞 UI

#### InfoView.swift (68 行)
**关于和信息页面。**

```swift
struct InfoView: View {
    let paragraph1 = "FastVLM¹ is a new family of Vision-Language models..."
    let paragraph2 = "This app showcases the FastVLM model in action..."
    let footer = "FastVLM: Efficient Vision Encoding... (CVPR 2025)"
    
    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 20.0) {
                Text("\(.init(paragraph1))\n\n\(.init(paragraph2))\n\n")
                Spacer()
                Text(.init(footer))
            }
            .toolbar { /* Close button */ }
        }
    }
}
```

**特性：**
- Markdown 格式支持（粗体、链接等）
- 文本可选择和复制
- 响应式布局（iOS/macOS 适配）

#### FastVLM.entitlements
**应用权限配置文件（XML 格式）。**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<dict>
    <!-- 相机访问权限 -->
    <key>com.apple.security.device.camera</key>
    <true/>
    
    <!-- 网络访问（下载模型） -->
    <key>com.apple.security.network.client</key>
    <true/>
    
    <!-- App Sandbox（macOS） -->
    <key>com.apple.security.app-sandbox</key>
    <true/>
</dict>
```

#### Info.plist
**应用元数据配置。**

```xml
<dict>
    <!-- 相机使用说明 -->
    <key>NSCameraUsageDescription</key>
    <string>FastVLM needs camera access to capture images for vision understanding.</string>
    
    <!-- 支持的界面方向 -->
    <key>UISupportedInterfaceOrientations</key>
    <array>
        <string>UIInterfaceOrientationPortrait</string>
        <string>UIInterfaceOrientationLandscapeLeft</string>
        <string>UIInterfaceOrientationLandscapeRight</string>
    </array>
    
    <!-- 最低系统版本 -->
    <key>LSMinimumSystemVersion</key>
    <string>15.2</string>
</dict>
```

#### Assets.xcassets/
**应用资源目录。**

```
Assets.xcassets/
├── Contents.json           # 资源清单
├── AccentColor.colorset/   # 主题色
│   └── Contents.json
└── AppIcon.appiconset/     # 应用图标
    ├── Contents.json
    ├── icon_16x16.png
    ├── icon_32x32.png
    ├── icon_128x128.png
    ├── icon_256x256.png
    ├── icon_512x512.png
    └── icon_1024x1024.png
```

---

### 4. Video/ - 视频捕获模块

独立的视频处理模块，封装了相机访问和帧捕获逻辑。

#### CameraController.swift (203 行)
**AVFoundation 相机控制器。**

**核心属性：**
```swift
@Observable
public class CameraController: NSObject {
    private var captureSession: AVCaptureSession?
    private var framesContinuation: AsyncStream<CMSampleBuffer>.Continuation?
    private let sessionQueue = DispatchQueue(label: "sessionQueue")
    
    public var backCamera = true  // 前置/后置切换
    public var devices = [AVCaptureDevice]()  // 可用设备列表
    
    @objc dynamic private var rotationCoordinator: AVCaptureDevice.RotationCoordinator?
}
```

**初始化流程：**
```swift
public func start() {
    sessionQueue.async {
        // 1. 创建 AVCaptureSession
        let captureSession = AVCaptureSession()
        
        // 2. 检查相机权限
        self.checkPermission()
        
        // 3. 配置捕获会话
        self.setupCaptureSession(position: .back)
        
        // 4. 开始捕获
        captureSession.startRunning()
    }
}
```

**捕获会话配置：**
```swift
private func setupCaptureSession(position: AVCaptureDevice.Position) {
    // 1. 选择相机设备
    let device = AVCaptureDevice.default(
        .builtInWideAngleCamera, 
        for: .video, 
        position: position
    )
    
    // 2. 创建输入
    let input = try AVCaptureDeviceInput(device: device)
    captureSession.addInput(input)
    
    // 3. 创建输出
    let output = AVCaptureVideoDataOutput()
    output.setSampleBufferDelegate(self, queue: sessionQueue)
    captureSession.addOutput(output)
    
    // 4. 配置分辨率
    captureSession.sessionPreset = .hd1920x1080
}
```

**帧捕获回调：**
```swift
extension CameraController: AVCaptureVideoDataOutputSampleBufferDelegate {
    public func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        // 通过 AsyncStream.Continuation 发送帧
        framesContinuation?.yield(sampleBuffer)
    }
}
```

**旋转处理（iOS）：**
```swift
#if os(iOS)
private func setOrientation(_ orientation: UIDeviceOrientation) {
    let angle: Double?
    switch orientation {
    case .portrait: angle = 90
    case .portraitUpsideDown: angle = 270
    case .landscapeLeft: angle = 0
    case .landscapeRight: angle = 180
    default: angle = nil
    }
    
    // 应用旋转角度
    for output in captureSession.outputs {
        output.connection(with: .video)?.videoRotationAngle = angle
    }
}
#endif
```

**异步流接口：**
```swift
public func attach(continuation: AsyncStream<CMSampleBuffer>.Continuation) {
    sessionQueue.async {
        self.framesContinuation = continuation
    }
}

public func detach() {
    sessionQueue.async {
        self.framesContinuation = nil
    }
}
```

#### VideoFrameView.swift (149 行)
**视频帧显示视图。**

```swift
public struct VideoFrameView: View {
    public let frames: AsyncStream<CVImageBuffer>
    public let cameraType: CameraType
    public let action: ((CVImageBuffer) -> Void)?
    
    @State private var hold: Bool = false
    @State private var videoFrame: CVImageBuffer?
    
    public var body: some View {
        Group {
            if let videoFrame {
                _ImageView(image: videoFrame)
                    .aspectRatio(contentMode: .fit)
            } else {
                Color.gray
            }
        }
        .task {
            // 异步接收视频帧
            for await frame in frames {
                if cameraType == .continuous || !hold {
                    videoFrame = frame
                    action?(frame)
                    
                    if cameraType == .single {
                        hold = true
                    }
                }
            }
        }
    }
}
```

**内部 ImageView：**
```swift
private struct _ImageView: View {
    let image: CVImageBuffer
    
    var body: some View {
        #if os(iOS) || os(visionOS)
        Image(decorative: cgImage, scale: 1.0)
            .resizable()
        #elseif os(macOS)
        Image(nsImage: NSImage(cgImage: cgImage, size: .zero))
            .resizable()
        #endif
    }
    
    private var cgImage: CGImage {
        // CVImageBuffer -> CIImage -> CGImage
        let ciImage = CIImage(cvImageBuffer: image)
        let context = CIContext()
        return context.createCGImage(ciImage, from: ciImage.extent)!
    }
}
```

**工作模式：**
- **Continuous（连续）**：实时更新每一帧
- **Single（单次）**：点击后冻结，只处理一次

#### CameraControlsView.swift
**相机控制按钮视图。**

```swift
struct CameraControlsView: View {
    @Bindable var camera: CameraController
    
    var body: some View {
        HStack {
            ForEach(camera.devices, id: \.uniqueID) { device in
                Button(device.localizedName) {
                    camera.device = device
                }
                .disabled(camera.device == device)
            }
        }
    }
}
```

#### CameraType.swift
**相机模式枚举。**

```swift
public enum CameraType: String, CaseIterable {
    case continuous = "Continuous"
    case single = "Single"
}
```

#### Video.h
**模块头文件（Objective-C）。**

```objc
#import <Foundation/Foundation.h>

FOUNDATION_EXPORT double VideoVersionNumber;
FOUNDATION_EXPORT const unsigned char VideoVersionString[];
```

---

### 5. FastVLM.xcodeproj/ - Xcode 项目配置

#### project.pbxproj
**Xcode 项目配置文件（二进制 plist 格式）。**

**包含内容：**
- 文件引用和组织结构
- Target 配置（FastVLM App, FastVLM Framework, Video Framework）
- 构建设置（编译选项、链接器标志）
- 依赖关系（SPM 包、本地框架）
- 签名和证书配置

**主要 Targets：**
1. **FastVLM App**: iOS/macOS 应用
2. **FastVLM**: 核心推理库（Framework）
3. **Video**: 视频捕获库（Framework）

#### xcshareddata/xcschemes/FastVLM App.xcscheme
**构建和运行方案配置（XML 格式）。**

```xml
<Scheme version="1.3">
    <BuildAction>
        <BuildActionEntries>
            <BuildActionEntry buildForRunning="YES">
                <BuildableReference
                    BuildableName="FastVLM App.app"
                    BlueprintName="FastVLM App"
                    ReferencedContainer="container:FastVLM.xcodeproj">
                </BuildableReference>
            </BuildActionEntry>
        </BuildActionEntries>
    </BuildAction>
    
    <LaunchAction>
        <BuildableProductRunnable>
            <BuildableReference BuildableName="FastVLM App.app"/>
        </BuildableProductRunnable>
    </LaunchAction>
</Scheme>
```

---

### 6. 其他文件

#### get_pretrained_mlx_model.sh
**模型下载脚本（Bash）。**

```bash
#!/bin/bash

# 提示用户选择模型
echo "Select FastVLM model:"
echo "1. FastVLM-0.5B (Fastest)"
echo "2. FastVLM-1.5B (Balanced)"
echo "3. FastVLM-7B (Most Accurate)"

read -p "Enter choice [1-3]: " choice

# 根据选择下载对应模型
case $choice in
    1) MODEL_URL="https://ml-site.cdn-apple.com/.../fastvlm_0.5b_stage3_llm.fp16.zip" ;;
    2) MODEL_URL="https://ml-site.cdn-apple.com/.../fastvlm_1.5b_stage3_llm.int8.zip" ;;
    3) MODEL_URL="https://ml-site.cdn-apple.com/.../fastvlm_7b_stage3_llm.int4.zip" ;;
esac

# 下载并解压到 app/ 目录
curl -L $MODEL_URL -o model.zip
unzip model.zip -d app/
```

#### README.md
**应用使用说明文档。**

包含：
- 功能介绍
- 支持的平台和系统版本
- 模型下载步骤
- 构建和运行指南
- 提示词自定义方法

---

### Swift 项目架构总结

```
┌─────────────────────────────────────────────┐
│         FastVLM App (User Interface)        │
│   - ContentView: 主界面和交互逻辑              │
│   - FastVLMModel: 模型管理和推理调度           │
│   - InfoView: 信息展示                       │
└────────────────┬────────────────────────────┘
                 │ 依赖
                 ↓
┌─────────────────────────────────────────────┐
│      FastVLM Framework (Core Engine)        │
│   - FastVLM: 主模型类（Vision + Language）     │
│   - FastVLMProcessor: 图像预处理              │
│   - MediaProcessingExtensions: 图像工具       │
│   - Language: Qwen2 语言模型                  │
│   - Vision: FastViTHD 视觉编码器              │
└────────────────┬────────────────────────────┘
                 │ 依赖
                 ↓
┌─────────────────────────────────────────────┐
│       Video Framework (Capture Layer)       │
│   - CameraController: 相机控制                │
│   - VideoFrameView: 帧显示                   │
│   - CameraType: 模式定义                     │
└─────────────────────────────────────────────┘
```

**依赖关系：**
- FastVLM App → FastVLM Framework → MLX, MLXVLM, CoreML
- FastVLM App → Video Framework → AVFoundation
- 所有模块 → SwiftUI, Foundation

**数据流：**
```
相机捕获 (CameraController)
    ↓ AsyncStream<CVImageBuffer>
VideoFrameView 显示
    ↓ action callback
ContentView 触发推理
    ↓ UserInput
FastVLMModel.generate()
    ↓ ModelContainer.perform()
FastVLM 推理
    ↓ AsyncStream<LMOutput>
UI 更新显示结果
```

## 性能指标

| 模型         | 参数量 | TTFT   | 准确率        | 应用场景                 |
|------------|------|--------|------------|----------------------|
| FastVLM-0.5B | 0.5B | 极快     | 良好         | 移动设备，速度优先            |
| FastVLM-1.5B | 1.5B | 快      | 优秀         | 平板/Mac，速度和准确率平衡     |
| FastVLM-7B   | 7B   | 较快     | 极优         | Mac/iPad Pro，准确率优先 |

## 数据流总结

```
[输入] 图像（相机/文件） + 文本提示
    ↓
[视觉处理] 图像编码器 → 视觉 tokens
    ↓
[多模态融合] 投影器 + 文本 embeddings
    ↓
[语言模型] 自回归解码 → 生成文本
    ↓
[输出] 文本答案 + 性能指标（TTFT）
```

## 图像到物体识别的详细流程

下面以识别图像中的物体为例，详细说明从输入图像到输出物体名称的完整流程。假设用户输入一张包含"苹果"的图片，并提问"图片中有什么物体？"

### 阶段一：图像预处理（Image Preprocessing）

#### 1.1 图像加载
**Python端（predict.py）：**
```python
image = Image.open(args.image_file).convert('RGB')
# 例如：加载一张 1920x1080 的照片
```

**Swift端（FastVLMModel.swift）：**
```swift
// 从相机或文件获取 CVPixelBuffer
let pixelBuffer = camera.captureFrame()
// 转换为 CIImage
let ciImage = CIImage(cvImageBuffer: pixelBuffer)
```

#### 1.2 图像尺寸调整和标准化
**Python端（mm_utils.py）：**
```python
# 1. 选择最佳分辨率
target_resolution = select_best_resolution(
    original_size=(1920, 1080),
    possible_resolutions=[(336, 336), (672, 336), (672, 672), ...]
)
# 结果：选择 (672, 336) 作为目标尺寸

# 2. 保持纵横比缩放并填充
image = resize_and_pad_image(image, target_resolution)
# 图像被缩放到 672x336，空白区域填充黑色

# 3. 可选：高分辨率分块处理（anyres）
if use_anyres:
    patches = divide_to_patches(image, patch_size=336)
    # 将图像分割为多个 336x336 的小块
```

**Swift端（FastVLM.swift - FastVLMProcessor）：**
```swift
// 1. 调整到最短边
let targetSize = fitIn(image.extent.size, shortestEdge: 336)
image = resampleBicubic(image, to: targetSize)

// 2. 归一化（CLIP 标准）
let normalized = image.normalized(
    mean: [0.48145466, 0.4578275, 0.40821073],
    std: [0.26862954, 0.26130258, 0.27577711]
)
```

#### 1.3 转换为张量
**Python端：**
```python
# CLIP ImageProcessor 处理
image_tensor = image_processor.preprocess(
    images=image, 
    return_tensors='pt'
)['pixel_values']
# 输出形状：[1, 3, 336, 672] (batch, channels, height, width)
```

**Swift端：**
```swift
// 转换为 MLXArray
let pixelValues = MLXArray(
    cgImage,
    channels: .last  // [height, width, channels]
)
// 输出形状：[336, 672, 3]
```

---

### 阶段二：视觉特征编码（Vision Encoding）

#### 2.1 FastViTHD 编码器前向传播

**Python端（clip_encoder.py / FastViTHD）：**
```python
# vision_tower.forward(images)
# FastViTHD 的核心处理流程：

# Step 1: Patch Embedding
# 将图像分割为 patch（例如 16x16 像素一个 patch）
patches = image_to_patches(image_tensor)  # [B, num_patches, patch_dim]
# 对于 336x672 图像，16x16 patch → (336/16) * (672/16) = 21 * 42 = 882 patches

# Step 2: Position Embedding
# 添加位置信息，让模型知道每个 patch 的空间位置
position_embeddings = get_position_embeddings(num_patches=882)
patch_embeddings = patches + position_embeddings

# Step 3: FastViTHD Transformer Blocks
# 通过多层 Transformer 处理
for layer in vision_encoder_layers:
    # Multi-Head Self-Attention：patch 之间相互关注
    attention_output = layer.self_attention(
        query=patch_embeddings,
        key=patch_embeddings,
        value=patch_embeddings
    )
    # Feed-Forward Network：非线性变换
    patch_embeddings = layer.feed_forward(attention_output)

# Step 4: 特征选择
# 选择指定层的输出（通常是倒数第二层，更通用的特征）
selected_features = hidden_states[select_layer]  # 例如 layer -2
# 去掉 CLS token，只保留 patch tokens
image_features = selected_features[:, 1:]  # [B, 882, hidden_dim]
# hidden_dim 通常是 1024 或 768（取决于 CLIP 模型）
```

**Swift端（FastVLM.swift - VisionModel）：**
```swift
// VisionModel.callAsFunction(hiddenStates, gridThw)
// 使用 CoreML 加速的 FastViTHD 模型

let output = visionModelCoreML.prediction(images: inputArray)
// 输入：[1, 336, 672, 3]
// 输出：[1, 256, 3072]
// 256 是压缩后的 visual tokens 数量（FastViTHD 的核心优化）
// 3072 是每个 token 的特征维度
```

**关键优化：Token 压缩**
- 传统 CLIP：882 个 patch tokens
- FastViTHD：256 个 visual tokens（压缩 ~3.4x）
- 大幅减少后续 LLM 的计算量，提升 TTFT

---

### 阶段三：多模态投影（Multimodal Projection）

#### 3.1 视觉特征投影到语言空间

**Python端（llava_arch.py）：**
```python
# LlavaMetaModel.encode_images()
def encode_images(self, images):
    # 1. 视觉编码（上一步的结果）
    image_features = self.vision_tower(images)
    # 形状：[batch=1, num_tokens=256, vision_dim=3072]
    
    # 2. 多模态投影器（MLP）
    image_features = self.mm_projector(image_features)
    # mm_projector 是一个多层感知机（通常 2 层）
    # 将视觉特征维度映射到 LLM 的 hidden_size
    
    # 内部结构（以 2-layer MLP 为例）：
    # Linear1: [3072] -> [intermediate_size=4096]
    # GELU activation
    # Linear2: [4096] -> [hidden_size=896/1536/3584]
    
    return image_features
    # 输出形状：[1, 256, 896]（以 FastVLM-0.5B 为例）
```

**Swift端（FastVLM.swift - FastVLMMultiModalProjector）：**
```swift
// multiModalProjector.callAsFunction(imageFeatures)
public func callAsFunction(_ x: MLXArray) -> MLXArray {
    var x = x
    x = linear1(x)       // [1, 256, 3072] -> [1, 256, 4096]
    x = gelu(x)          // 激活函数
    x = linear2(x)       // [1, 256, 4096] -> [1, 256, 896]
    return x
}
// 输出：[1, 256, 896]
```

#### 3.2 构建多模态输入序列

**Python端（llava_arch.py - prepare_inputs_labels_for_multimodal）：**
```python
# 假设用户提问："图片中有什么物体？"
# 文本经过 tokenizer 后：
# input_ids = [1, 151644, ..., 32001, ..., 151645]
#               ^开始    ^图像token  ^结束
# 其中 32001 是特殊的 IMAGE_TOKEN_INDEX

# 1. 找到图像 token 的位置
image_token_indices = torch.where(input_ids == IMAGE_TOKEN_INDEX)[1]
# 例如位置 5

# 2. 获取文本 embeddings
text_embeddings = self.embed_tokens(input_ids)
# 形状：[1, seq_len=50, hidden_size=896]

# 3. 将图像特征插入到对应位置
# 在 IMAGE_TOKEN_INDEX 位置，用 256 个 visual tokens 替换
new_embeddings = torch.cat([
    text_embeddings[:, :5, :],          # 图像前的文本："图片"
    image_features,                     # [1, 256, 896] 视觉特征
    text_embeddings[:, 6:, :]           # 图像后的文本："中有什么物体？"
], dim=1)

# 最终输入序列形状：[1, 305, 896]
# = 5（前文） + 256（图像） + 44（后文）
```

**Swift端（FastVLM.swift - prepare）：**
```swift
// 构建输入 embeddings
let textTokens = tokenizer.encode(text: prompt)
// [1, 151644, ..., 151645]

let imageEmbeddings = multiModalProjector(visionFeatures)
// [1, 256, 896]

let textEmbeddings = languageModel.embedTokens(textTokens)
// [1, 50, 896]

// 合并（在 image token 位置插入）
let inputEmbedding = concatenate([
    textEmbeddings[.newAxis, 0..<imageTokenPosition, 0...],
    imageEmbeddings,
    textEmbeddings[.newAxis, (imageTokenPosition+1)..., 0...]
], axis: 1)
// [1, 305, 896]
```

---

### 阶段四：语言模型推理（Language Model Inference）

#### 4.1 Transformer 解码

**Python端（Qwen2ForCausalLM）：**
```python
# 语言模型的前向传播
def forward(input_embedding, attention_mask, cache=None):
    hidden_states = input_embedding  # [1, 305, 896]
    
    # 多层 Transformer Decoder
    for layer in range(num_layers):  # 例如 24 层
        # 1. Self-Attention
        # 每个 token 关注之前的所有 tokens（因果注意力）
        query = layer.q_proj(hidden_states)
        key = layer.k_proj(hidden_states)
        value = layer.v_proj(hidden_states)
        
        # 应用 Multimodal Rotary Position Embedding (mrope)
        # 对文本和视觉 tokens 使用不同的位置编码策略
        query, key = apply_multimodal_rotary_embedding(
            query, key, 
            mrope_section=[2, 1, 1]  # 时间、高度、宽度三维位置
        )
        
        # Multi-Head Attention 计算
        attention_scores = (query @ key.transpose(-2, -1)) / sqrt(head_dim)
        attention_weights = softmax(attention_scores, dim=-1)
        attention_output = attention_weights @ value
        
        # 2. Feed-Forward Network
        hidden_states = layer.mlp(attention_output)
        # MLP: Linear -> SwiGLU -> Linear
        
    # 3. 最后归一化
    hidden_states = norm(hidden_states)  # [1, 305, 896]
    
    return hidden_states
```

**Swift端（FastVLM.swift - LanguageModel）：**
```swift
// languageModel.callAsFunction(inputEmbedding)
public func callAsFunction(inputEmbedding: MLXArray, cache: [KVCache]?) -> LMOutput {
    var h = inputEmbedding  // [1, 305, 896]
    
    // 创建因果注意力掩码（防止看到未来的 tokens）
    let mask = createAttentionMask(h: h, cache: cache)
    
    // 逐层处理
    for (i, layer) in layers.enumerated() {
        h = layer(h, mask: mask, cache: cache?[i])
    }
    
    // 归一化
    h = norm(h)
    
    return h  // [1, 305, 896]
}
```

#### 4.2 语言模型头（LM Head）预测下一个词

**Python端：**
```python
# LM Head：将 hidden states 映射到词汇表概率
logits = lm_head(hidden_states)
# lm_head 是一个线性层：[896] -> [vocab_size=151936]
# 输出形状：[1, 305, 151936]

# 只需要最后一个位置的预测（下一个词）
next_token_logits = logits[:, -1, :]  # [1, 151936]

# 采样策略（temperature=0 时取最大概率）
if temperature == 0:
    next_token = torch.argmax(next_token_logits, dim=-1)
else:
    # 温度采样
    probs = softmax(next_token_logits / temperature, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)

# 假设预测出 token_id = 101248（对应"苹"）
```

**Swift端：**
```swift
// LM Head 投影
var logits = lmHead(h)  // [1, 305, 151936]

// 提取最后一个位置
let nextTokenLogits = logits[0..., -1, 0...]  // [1, 151936]

// Argmax 采样
let nextToken = MLX.argMax(nextTokenLogits, axis: -1)
// 例如：token_id = 101248
```

---

### 阶段五：自回归解码（Autoregressive Decoding）

#### 5.1 迭代生成

```python
# 生成循环（生成最多 max_tokens=240 个 token）
generated_tokens = []
current_input = input_embedding  # 初始包含图像+问题

for step in range(max_tokens):
    # 1. 模型前向传播
    outputs = language_model(current_input, cache=kv_cache)
    next_token = sample(outputs.logits[:, -1, :])
    
    # 2. 添加到生成序列
    generated_tokens.append(next_token)
    
    # 3. 检查结束符
    if next_token == eos_token_id:  # 例如 151645
        break
    
    # 4. 更新输入（只需新 token 的 embedding）
    # 使用 KV Cache 避免重复计算之前的 tokens
    next_embedding = embed_tokens(next_token)
    current_input = next_embedding  # [1, 1, 896]

# 示例生成序列（token IDs）：
# [101248, 101311, 151643]
# 对应："苹" + "果" + "。"
```

**KV Cache 优化：**
```python
# 第一次前向：计算所有 305 个 tokens
# K, V shape: [num_layers, batch, num_heads, 305, head_dim]

# 第二次前向：只计算新 token（使用缓存的 K, V）
# 新 K, V shape: [num_layers, batch, num_heads, 1, head_dim]
# 拼接到缓存：cache_k = concat([cache_k, new_k], dim=2)

# 大幅减少计算量（305x → 1x per step）
```

#### 5.2 Token 解码为文本

**Python端：**
```python
# tokenizer.decode() 将 token IDs 转回文本
output_text = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True
)
# 输出："苹果。"
```

**Swift端：**
```swift
// 逐 token 解码并更新 UI
for token in generatedTokens {
    let piece = tokenizer.decode(token: token)
    model.output += piece
    // 实时显示："苹" -> "苹果" -> "苹果。"
}
```

---

### 阶段六：完整回答生成

#### 6.1 拼接完整回答

假设模型生成的完整 token 序列为：
```
[101248, 101311, 151643, 105234, 106891, 99012, 100345, 151645]
```

解码后：
```
"苹果。图片中有一个红色的苹果放在桌子上。"
```

#### 6.2 性能指标记录

**Time-to-First-Token (TTFT)：**
```python
# 从开始推理到生成第一个 token 的时间
ttft = time.time() - start_time
# 例如：FastVLM-0.5B 在 iPhone 15 Pro 上 TTFT ≈ 0.12秒
```

**Tokens per Second (TPS)：**
```python
# 生成速度
tps = len(generated_tokens) / total_time
# 例如：~25 tokens/s
```

---

### 完整流程总结（以"识别苹果"为例）

```
[用户输入]
  图像：apple.jpg (1920x1080, RGB)
  提问："图片中有什么物体？"

[步骤 1] 图像预处理
  → 缩放到 672x336
  → 归一化 (CLIP 标准)
  → 转张量：[1, 3, 336, 672]

[步骤 2] 视觉编码 (FastViTHD)
  → Patch Embedding: 882 patches
  → Transformer 编码: 12层
  → Token 压缩: 882 → 256 tokens
  → 输出：[1, 256, 3072]

[步骤 3] 多模态投影
  → MLP 投影: [3072] → [896]
  → 输出：[1, 256, 896]

[步骤 4] 构建输入序列
  → 文本 tokenize: "图片中有什么物体？"
  → 插入视觉特征到 <image> token 位置
  → 输入序列：[1, 305, 896]
    = [文本前缀] + [256个视觉tokens] + [文本后缀]

[步骤 5] Transformer 推理
  → 24层 Qwen2 Decoder
  → Multi-Head Self-Attention (8 heads)
  → Feed-Forward Networks
  → Multimodal Rotary Position Embedding

[步骤 6] 自回归生成
  迭代 1: 预测 token_id=101248 ("苹")
  迭代 2: 预测 token_id=101311 ("果")
  迭代 3: 预测 token_id=151643 ("。")
  迭代 4: 预测 token_id=105234 ("图")
  ...
  迭代 N: 预测 token_id=151645 (EOS)

[步骤 7] 解码输出
  → tokenizer.decode(tokens)
  → 最终答案："苹果。图片中有一个红色的苹果放在桌子上。"

[性能指标]
  TTFT: 120ms
  生成速度: 25 tokens/s
  总时长: 450ms
```

### 关键技术点

1. **FastViTHD 的核心优势**：
   - 传统方法：882 个 visual tokens → 计算密集
   - FastViTHD：256 个 visual tokens → 减少 3.4x 计算量
   - 结果：TTFT 提升 85 倍（相比 LLaVA-OneVision-0.5B）

2. **多模态位置编码 (Multimodal RoPE)**：
   - 视觉 tokens：2D 空间位置（高度、宽度）
   - 文本 tokens：1D 序列位置（时间）
   - 统一编码：[时间, 高度, 宽度] 三维位置信息

3. **KV Cache 优化**：
   - 首次推理：计算 305 个 tokens 的 K、V
   - 后续推理：只计算新 token，复用缓存
   - 速度提升：~305 倍（每步）

4. **设备端优化（Apple Silicon）**：
   - CoreML 加速视觉编码器
   - MLX 框架高效张量运算
   - Metal GPU 并行计算
   - 量化（INT4/INT8）减少内存占用

## 许可证

- **代码**：Apache License 2.0（继承自 LLaVA）
- **模型**：Apple LICENSE_MODEL

## 相关资源

- 论文：[FastVLM: Efficient Vision Encoding for Vision Language Models](https://www.arxiv.org/abs/2412.13303)
- 基础框架：[LLaVA](https://github.com/haotian-liu/LLaVA)
- MLX 框架：[Apple MLX](https://github.com/ml-explore/mlx)

---

**更新日期**：2026年1月7日
