// Copyright © 2026 Apple Inc.

import Foundation
import FoundationModels
import Testing

@testable import MLXFoundationModels

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private struct ImageAnalysis {
    var image: ImageReference
    var analysis: String
}

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private struct ImageAnalysisList {
    var images: [ImageAnalysis]
}

@Suite("Guided image reference schema")
struct ImageReferenceSchemaTests {

    /// The `attachmentLabel` property of the `ImageReference` definition, as a
    /// parsed JSON object.
    private func labelSchema(in json: String) throws -> [String: Any] {
        let root =
            try JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any] ?? [:]
        let defs = root["$defs"] as? [String: Any] ?? [:]
        let reference = defs["ImageReference"] as? [String: Any] ?? [:]
        let properties = reference["properties"] as? [String: Any] ?? [:]
        return properties["attachmentLabel"] as? [String: Any] ?? [:]
    }

    @Test
    func testKnownLabelsBecomeAnEnumOnTheReference() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let json = try SchemaConverter.encodeToJSON(
            ImageAnalysisList.generationSchema,
            attachmentLabels: ["Photo_A1B2C3", "Photo_D4E5F6"])

        let label = try labelSchema(in: json)
        #expect(label["type"] as? String == "string")
        #expect(label["enum"] as? [String] == ["Photo_A1B2C3", "Photo_D4E5F6"])
    }

    @Test
    func testNoLabelsLeavesTheSchemaUnchanged() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        // An empty enum would make the field unsatisfiable, and a transcript
        // with no labeled images has nothing to constrain against, so the
        // schema has to pass through untouched.
        let json = try SchemaConverter.encodeToJSON(
            ImageAnalysisList.generationSchema, attachmentLabels: [])

        let label = try labelSchema(in: json)
        #expect(label["enum"] == nil)
    }

    @Test
    func testSchemasWithoutAnImageReferenceAreUnaffected() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let json = try SchemaConverter.encodeToJSON(
            String.generationSchema, attachmentLabels: ["Photo_A1B2C3"])

        // No crash, no injected enum, still valid JSON.
        let root = try JSONSerialization.jsonObject(with: Data(json.utf8))
        #expect(root is [String: Any])
        #expect(!json.contains("Photo_A1B2C3"))
    }
}

#endif  // FoundationModelsIntegration && canImport(FoundationModels)
