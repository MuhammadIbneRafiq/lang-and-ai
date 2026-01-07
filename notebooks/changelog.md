# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial framework for experimental analysis.
- Data preprocessing pipeline for gender and political leaning datasets.

## [1.2.0] - 2026-01-08

### Added
- Cross-condition evaluation for RoBERTa models.
- Final evaluation phase (E4) for comprehensive model comparison.

### Changed
- Updated SVM baseline to include performance drop analysis.
- Improved logging for model training progress.

### Fixed
- Resolved data leakage issues in training sets.
- Fixed inconsistent metric reporting between dev and test sets.

## [1.1.0] - 2026-01-07

### Added
- RoBERTa baseline implementation for gender classification.
- Stylistic SVM baseline with character n-grams.
- Keyword heuristic baseline for initial assessment.

### Fixed
- Addressed TensorFlow warnings and CUDA initialization errors.
- Corrected label distribution reporting in majority baseline.

## [1.0.0] - 2026-01-06

### Added
- Majority baseline for gender classification.
- Initial data loading and validation framework.
- Experiment tracking and results saving infrastructure.
