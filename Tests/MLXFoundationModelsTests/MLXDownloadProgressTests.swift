// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import Testing

@testable import MLXFoundationModels

@Suite @MainActor
struct MLXDownloadProgressTests {

    // MARK: - Pure derivation (completedBytes)

    @Test func derivesBytesFromFraction() {
        // 66% of a 100-byte download is 66 bytes, regardless of how many
        // child files have individually finished.
        let bytes = MLXDownloadProgress.completedBytes(
            fraction: 0.66, totalUnitCount: 100, previous: 0)
        #expect(bytes == 66)
    }

    @Test func clampsFractionIntoUnitRange() {
        #expect(
            MLXDownloadProgress.completedBytes(
                fraction: 1.5, totalUnitCount: 100, previous: 0) == 100)
        #expect(
            MLXDownloadProgress.completedBytes(
                fraction: -0.5, totalUnitCount: 100, previous: 0) == 0)
    }

    @Test func isMonotonicAgainstPrevious() {
        // A slightly lower fraction sample must not walk the byte count back.
        let bytes = MLXDownloadProgress.completedBytes(
            fraction: 0.40, totalUnitCount: 100, previous: 50)
        #expect(bytes == 50)
    }

    @Test func handlesUnknownOrZeroTotal() {
        #expect(
            MLXDownloadProgress.completedBytes(
                fraction: 0.5, totalUnitCount: 0, previous: 42) == 42)
    }

    @Test func handlesNonFiniteFraction() {
        #expect(
            MLXDownloadProgress.completedBytes(
                fraction: .nan, totalUnitCount: 100, previous: 10) == 10)
    }

    // MARK: - ingest reproduces the stall via a real parent/child Progress

    @Test func ingestFollowsFractionNotCompletedUnitCount() {
        // Small file done (32) + big file half-done (34/68): Foundation reports
        // parent.fractionCompleted == 0.66 but parent.completedUnitCount == 32.
        let parent = Progress(totalUnitCount: 100)
        let done = Progress(totalUnitCount: 32, parent: parent, pendingUnitCount: 32)
        done.completedUnitCount = 32
        let big = Progress(totalUnitCount: 68, parent: parent, pendingUnitCount: 68)
        big.completedUnitCount = 34

        let p = MLXDownloadProgress()
        p.ingest(
            fraction: parent.fractionCompleted,
            totalUnitCount: parent.totalUnitCount,
            modelID: "test/model")

        #expect(p.isActive)
        #expect(p.totalBytes == 100)
        // Derived from fraction (66), NOT the stalled completedUnitCount (32).
        #expect(p.completedBytes == 66)
        #expect(p.modelName == "test/model")
    }

    @Test func ingestKeepsBytesMonotonicAcrossSamples() {
        let p = MLXDownloadProgress()
        p.ingest(fraction: 0.60, totalUnitCount: 100, modelID: "m")
        #expect(p.completedBytes == 60)
        // A momentary lower sample must not regress the displayed bytes.
        p.ingest(fraction: 0.55, totalUnitCount: 100, modelID: "m")
        #expect(p.completedBytes == 60)
        // fractionCompleted is derived from the monotonic bytes (60/100), so it
        // does not regress to 0.55 even though the raw sample did.
        #expect(p.fractionCompleted == 0.6)
    }

    @Test func reportProgressReadsFractionFromParentProgress() async {
        let parent = Progress(totalUnitCount: 100)
        let child = Progress(totalUnitCount: 100, parent: parent, pendingUnitCount: 100)
        child.completedUnitCount = 60  // parent.completedUnitCount stays 0

        let p = MLXDownloadProgress()
        p.reportProgress(parent, modelID: "m")
        await drainMainActor()

        #expect(p.completedBytes == 60)
    }

    /// Yields repeatedly so the main-actor `Task` enqueued by a `nonisolated`
    /// entry point runs before we assert. Bounded so a wiring regression fails
    /// fast instead of hanging.
    private func drainMainActor() async {
        for _ in 0 ..< 20 { await Task.yield() }
    }

    // MARK: - completion semantics

    @Test func ignoresCachedModelImmediate100() {
        // Fully cached model: swift-huggingface emits a single fraction==1.0
        // callback. It must not activate or flash the download UI.
        let p = MLXDownloadProgress()
        p.ingest(fraction: 1.0, totalUnitCount: 1, modelID: "cached/model")
        #expect(p.isActive == false)
        #expect(p.completedBytes == 0)
        #expect(p.modelName == nil)
    }

    @Test func publishesFinalStateForActiveDownload() {
        // A real download that has been progressing must publish its final
        // 100% state (bytes == total) rather than dropping the last callback.
        let p = MLXDownloadProgress()
        p.ingest(fraction: 0.90, totalUnitCount: 100, modelID: "big/model")
        #expect(p.isActive)
        p.ingest(fraction: 1.0, totalUnitCount: 100, modelID: "big/model")
        #expect(p.isActive)  // still loading the model into memory
        #expect(p.completedBytes == 100)
        #expect(p.fractionCompleted == 1.0)
    }

    @Test func reportCompletedResetsToIdle() {
        let p = MLXDownloadProgress()
        p.ingest(fraction: 1.0, totalUnitCount: 100, modelID: "m")
        p.applyCompleted()
        #expect(p.isActive == false)
        #expect(p.completedBytes == 0)
        #expect(p.totalBytes == 0)
        #expect(p.modelName == nil)
        #expect(p.startedAt == nil)
        #expect(p.throughputBytesPerSec == nil)
    }

    // MARK: - throughput

    @Test func throughputReflectsDerivedByteStream() async throws {
        let p = MLXDownloadProgress()
        p.ingest(fraction: 0.10, totalUnitCount: 1_000_000, modelID: "m")
        // Space the second sample so the rolling window has a positive dt.
        try await Task.sleep(for: .milliseconds(150))
        p.ingest(fraction: 0.40, totalUnitCount: 1_000_000, modelID: "m")
        // 300_000 bytes over ~0.15s => clearly positive, non-nil throughput.
        let rate = p.throughputBytesPerSec
        #expect(rate != nil)
        #expect((rate ?? 0) > 0)
    }
}

#endif  // FoundationModelsIntegration && canImport(FoundationModels, _version: 2)
