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
    /// parsed JSON object. Fails the test (rather than returning an empty
    /// dictionary) if any level of the expected shape is missing, so a
    /// lookup miss is distinguishable from a correct absence of `enum`.
    private func labelSchema(in json: String) throws -> [String: Any] {
        let root = try #require(
            try JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any])
        let defs = try #require(root["$defs"] as? [String: Any])
        let reference = try #require(defs["ImageReference"] as? [String: Any])
        let properties = try #require(reference["properties"] as? [String: Any])
        return try #require(properties["attachmentLabel"] as? [String: Any])
    }

    /// Canonicalizes a JSON string by re-serializing with sorted keys, so two
    /// documents that differ only in key order compare equal.
    private func canonicalize(_ json: String) throws -> String {
        let object = try JSONSerialization.jsonObject(with: Data(json.utf8))
        let data = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        return try #require(String(data: data, encoding: .utf8))
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
    func testNoLabelsIsStructurallyIdenticalToPlainEncoding() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        // Compares canonicalized structure, not raw bytes. `JSONEncoder`'s
        // key order for `GenerationSchema` is not stable from call to call
        // (confirmed empirically: encoding the same schema value twice in
        // the same process can already produce different orderings), so a
        // byte-for-byte comparison is inherently flaky and must not be
        // reintroduced here. Sorting both sides' keys before comparing
        // isolates the property this guards: that the no-labels path
        // performs no rewrite (no injected `enum`, no dropped property,
        // no other mutation), independent of encoder key-order churn.
        let schema = ImageAnalysisList.generationSchema
        let direct = String(data: try JSONEncoder().encode(schema), encoding: .utf8)!

        let json = try SchemaConverter.encodeToJSON(schema)

        #expect(try canonicalize(json) == canonicalize(direct))
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

    @Test
    func testRepeatedEncodingOfTheSameSchemaIsByteIdentical() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        // The compiled-grammar cache downstream is keyed on this exact string,
        // so the same schema has to encode to the same bytes every time or the
        // cache never hits and grows an entry per request. `JSONEncoder`'s key
        // order for `GenerationSchema` is not stable on its own, so this holds
        // only because the output is canonicalized. Both paths matter: the
        // rewrite is skipped without labels, the canonicalization is not.
        let schema = ImageAnalysisList.generationSchema
        let labels = ["Photo_A1B2C3", "Photo_D4E5F6"]

        let withoutLabels = try (0 ..< 20).map { _ in
            try SchemaConverter.encodeToJSON(schema)
        }
        #expect(Set(withoutLabels).count == 1)

        let withLabels = try (0 ..< 20).map { _ in
            try SchemaConverter.encodeToJSON(schema, attachmentLabels: labels)
        }
        #expect(Set(withLabels).count == 1)
    }
}

#endif  // FoundationModelsIntegration && canImport(FoundationModels)
