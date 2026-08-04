// Copyright © 2025 Apple Inc.

import Foundation
import FoundationModels
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private struct CaptionedImage {
    var image: ImageReference
    var caption: String
}

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private struct CaptionedImageList {
    var images: [CaptionedImage]
}

/// The adapter is the only place that can enforce `.vision` for labeled
/// image attachments, because the SDK's own vision guard doesn't inspect
/// these public attachment segments, only its own internal image path. The
/// gate covers prompt attachments: instructions attachments are dropped
/// during conversion and never reach it. The gate must fire before any
/// weight download, so these tests run with no model on disk.
@Suite("MLXLanguageModel vision capability gate")
struct VisionCapabilityGateTests {

    @Test("Image input without .vision throws unsupportedCapability(.vision)")
    func imageWithoutVisionThrows() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let model = makeStubModel(
            "vision/not-declared",
            capabilities: [])
        let executor = try makeMLXExecutor(for: model)

        let attachment = Transcript.AttachmentSegment(
            content: .image(Transcript.ImageAttachment(makeSolidCGImage())),
            label: "photo")
        let prompt = Transcript.Prompt(
            segments: [
                .text(Transcript.TextSegment(content: "Describe this")),
                .attachment(attachment),
            ],
            responseFormat: nil
        )
        let request = makeExecutorRequest(
            transcript: Transcript(entries: [.prompt(prompt)]))
        let channel = LanguageModelExecutorGenerationChannel()

        do {
            try await executor.respond(
                to: request, model: model, streamingInto: channel)
            Issue.record("Expected unsupportedCapability(.vision), but respond returned")
            return
        } catch let error as LanguageModelError {
            guard case .unsupportedCapability(let unsupported) = error else {
                Issue.record("Expected unsupportedCapability, got \(error)")
                return
            }
            #expect(unsupported.capability == .vision)
        } catch {
            Issue.record(
                "Expected LanguageModelError.unsupportedCapability(.vision), got: \(error)")
        }
    }

    /// The labels of the images a transcript carries have to reach the schema a
    /// guided request is constrained to, otherwise the model can name a picture
    /// that cannot be looked up. Computed before the weights are touched, so
    /// this is observable with no model on disk.
    @Test("Labeled attachments pin the guided image reference")
    func labeledAttachmentsPinTheGuidedImageReference() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let attachment = Transcript.AttachmentSegment(
            content: .image(Transcript.ImageAttachment(makeSolidCGImage())),
            label: "Photo_A1B2C3")
        let prompt = Transcript.Prompt(
            segments: [
                .text(Transcript.TextSegment(content: "Caption this")),
                .attachment(attachment),
            ],
            responseFormat: nil
        )
        let request = makeExecutorRequest(
            transcript: Transcript(entries: [.prompt(prompt)]),
            schema: CaptionedImageList.generationSchema)

        let encoded = try MLXLanguageModel.Executor.guidedSchemaJSON(for: request)
        let json = try #require(encoded)
        let root = try #require(
            try JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any])
        let defs = try #require(root["$defs"] as? [String: Any])
        let reference = try #require(defs["ImageReference"] as? [String: Any])
        let properties = try #require(reference["properties"] as? [String: Any])
        let label = try #require(properties["attachmentLabel"] as? [String: Any])
        #expect(label["enum"] as? [String] == ["Photo_A1B2C3"])
    }
}

#endif  // FoundationModelsIntegration && canImport(FoundationModels)
