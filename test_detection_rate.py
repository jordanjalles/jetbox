"""
Script to test detection rate on the injection samples corpus.
"""

from behaviors.security_prompt_injection_detector import PromptInjectionDetectorBehavior
from pathlib import Path

def main():
    behavior = PromptInjectionDetectorBehavior()
    corpus_path = Path(__file__).parent / "tests/security/fixtures/injection_samples.txt"

    with open(corpus_path) as f:
        lines = f.readlines()

    malicious_samples = []
    benign_samples = []

    # Parse corpus
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        parts = line.split('|')
        if len(parts) != 3:
            continue

        label = parts[0].strip()
        expected_conf = float(parts[1].strip())
        sample = parts[2].strip()

        if label == "MALICIOUS":
            malicious_samples.append((sample, expected_conf))
        elif label == "BENIGN":
            benign_samples.append((sample, expected_conf))

    # Test malicious samples
    detected_malicious = 0
    warned_malicious = 0
    for sample, expected_conf in malicious_samples:
        patterns = behavior._detect_injection_patterns(sample)
        heuristics = behavior._detect_injection_heuristics(sample)
        confidence = behavior._compute_confidence(patterns, heuristics)

        if confidence >= 0.75:
            detected_malicious += 1
        elif confidence >= 0.45:
            warned_malicious += 1

    # Test benign samples
    false_positives_block = 0
    false_positives_warn = 0
    for sample, expected_conf in benign_samples:
        patterns = behavior._detect_injection_patterns(sample)
        heuristics = behavior._detect_injection_heuristics(sample)
        confidence = behavior._compute_confidence(patterns, heuristics)

        if confidence >= 0.75:
            false_positives_block += 1
        elif confidence >= 0.45:
            false_positives_warn += 1

    # Calculate rates
    detection_rate_block = (detected_malicious / len(malicious_samples)) * 100
    detection_rate_warn = ((detected_malicious + warned_malicious) / len(malicious_samples)) * 100
    false_positive_rate_block = (false_positives_block / len(benign_samples)) * 100
    false_positive_rate_warn = ((false_positives_block + false_positives_warn) / len(benign_samples)) * 100

    print("=" * 70)
    print("DETECTION RATE ANALYSIS")
    print("=" * 70)
    print(f"\nCorpus: {len(malicious_samples)} malicious, {len(benign_samples)} benign\n")

    print("MALICIOUS SAMPLES:")
    print(f"  Blocked (>=0.75): {detected_malicious}/{len(malicious_samples)} = {detection_rate_block:.1f}%")
    print(f"  Warned (>=0.45): {warned_malicious}/{len(malicious_samples)}")
    print(f"  Total detected: {detected_malicious + warned_malicious}/{len(malicious_samples)} = {detection_rate_warn:.1f}%")

    print("\nBENIGN SAMPLES:")
    print(f"  False positive (blocked): {false_positives_block}/{len(benign_samples)} = {false_positive_rate_block:.1f}%")
    print(f"  False positive (warned): {false_positives_warn}/{len(benign_samples)}")
    print(f"  Total false positives: {false_positives_block + false_positives_warn}/{len(benign_samples)} = {false_positive_rate_warn:.1f}%")

    print("\n" + "=" * 70)
    print("SUMMARY:")
    print(f"  Detection rate (block + warn): {detection_rate_warn:.1f}%")
    print(f"  False positive rate: {false_positive_rate_warn:.1f}%")
    print(f"  Target: 70%+ detection, <10% false positives")

    # Determine pass/fail
    passed = detection_rate_warn >= 70.0 and false_positive_rate_warn < 10.0
    if passed:
        print("\n✓ TARGETS MET")
    else:
        print("\n✗ TARGETS NOT MET")
        if detection_rate_warn < 70.0:
            print(f"  - Detection rate too low ({detection_rate_warn:.1f}% < 70%)")
        if false_positive_rate_warn >= 10.0:
            print(f"  - False positive rate too high ({false_positive_rate_warn:.1f}% >= 10%)")
    print("=" * 70)

    return 0 if passed else 1

if __name__ == "__main__":
    exit(main())
