"""Test modülü v2 — Multimodal sınıflandırma testleri."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import ClassificationResult, CancerCategory, ProcessingRecord
from src.validator import Validator


class TestValidator:
    def setup_method(self):
        self.validator = Validator({
            "confidence_threshold": 0.6,
            "require_human_review_below": 0.4,
            "use_keyword_validation": True,
        })

    def test_lung_keywords(self):
        result = ClassificationResult(
            category="Lung", confidence=0.9, reasoning="Akciğer kanseri",
        )
        text = "Sol akciğer alt lob pnömonektomi. Skuamöz hücreli karsinom. P40 pozitif, TTF-1 negatif."
        val = self.validator.validate(result, text)
        assert val.validated_category == "Lung"
        assert len(val.keyword_matches) > 0

    def test_brain_keywords(self):
        result = ClassificationResult(
            category="Brain", confidence=0.85, reasoning="Beyin tümörü",
        )
        text = "Serebrum beyin tümör rezeksiyonu. Glioblastom, IDH negatif, ATRX pozitif, GFAP pozitif."
        val = self.validator.validate(result, text)
        assert val.validated_category == "Brain"

    def test_ovary_keywords(self):
        result = ClassificationResult(
            category="Ovary", confidence=0.88, reasoning="Over kanseri",
        )
        text = "Sağ ve sol over. Seröz papiller kistadenokarsinom. Kapsül dışına taşan tümöral doku."
        val = self.validator.validate(result, text)
        assert val.validated_category == "Ovary"

    def test_blood_keywords(self):
        result = ClassificationResult(
            category="Blood", confidence=0.82, reasoning="AML",
        )
        text = "AML tanısı. Blast sayımı %25. Myeloid panel normal. Karyotip çıkmadı."
        val = self.validator.validate(result, text)
        assert val.validated_category == "Blood"

    def test_bone_marrow_keywords(self):
        result = ClassificationResult(
            category="Bone_Marrow", confidence=0.78, reasoning="Polisitemi",
        )
        text = "Polisitemi sebebiyle JAK2 mutasyonu çalışması. Kemik iliği biyopsisi (KİBX)."
        val = self.validator.validate(result, text)
        assert val.validated_category == "Bone_Marrow"

    def test_low_confidence_triggers_review(self):
        result = ClassificationResult(
            category="Other", confidence=0.3, reasoning="Belirsiz",
        )
        val = self.validator.validate(result, "Kısa metin")
        assert any("İnsan incelemesi" in w for w in val.warnings)

    def test_benign_with_cancer_category_warns(self):
        result = ClassificationResult(
            category="Lung", confidence=0.7, reasoning="Akciğer",
        )
        text = "Reaktif lenf nodu. Benign bulgular. Malignite açısından negatif."
        val = self.validator.validate(result, text)
        assert any("benign" in w.lower() for w in val.warnings)

    def test_short_text_reduces_confidence(self):
        result = ClassificationResult(
            category="Lung", confidence=0.8, reasoning="Test",
        )
        val = self.validator.validate(result, "Kısa")
        assert val.confidence_adjusted < 0.8


class TestCategoryNormalization:
    """LLM bağlantısı gerektirmeyen birim testleri (service.classification.Classifier)."""

    def _clf(self):
        from service.classification import Classifier
        clf = object.__new__(Classifier)
        clf.valid = {c.value for c in CancerCategory}
        return clf

    def test_exact_and_case(self):
        clf = self._clf()
        assert clf._normalize_category("Lung") == "Lung"
        assert clf._normalize_category("BRAIN") == "Brain"
        assert clf._normalize_category("bone marrow") == "Bone_Marrow"

    def test_turkish_names(self):
        clf = self._clf()
        assert clf._normalize_category("akciğer") == "Lung"
        assert clf._normalize_category("Beyin tümörü") == "Brain"
        assert clf._normalize_category("over") == "Ovary"
        assert clf._normalize_category("kemik iliği") == "Bone_Marrow"
        assert clf._normalize_category("bilinmeyen_kategori") == "Other"

    def test_extract_json_variants(self):
        from service.backends.llm import extract_json
        assert extract_json('{"category": "Lung", "confidence": 0.92}')["category"] == "Lung"
        assert extract_json('Analiz:\n```json\n{"category": "Brain"}\n```')["category"] == "Brain"
        assert extract_json('<think>hmm</think>{"category": "Ovary"}')["category"] == "Ovary"

class TestModels:
    def test_classification_result_with_summary(self):
        r = ClassificationResult(
            category="Lung",
            confidence=0.9,
            reasoning="Test",
            extracted_text_summary="TANI: Karsinom",
        )
        assert r.extracted_text_summary == "TANI: Karsinom"

    def test_processing_record_input_mode(self):
        r = ProcessingRecord(
            filename="test.png",
            file_type="png",
            input_mode="vision",
            classification_category="Lung",
            classification_confidence=0.9,
            validation_passed=True,
            final_category="Lung",
            processing_time_seconds=3.5,
            llm_model="qwen2.5vl:7b",
        )
        assert r.input_mode == "vision"


class TestRealReportScenarios:
    """Yüklenen 8 raporun beklenen sonuçlarını doğrulama ile test et."""

    def setup_method(self):
        self.validator = Validator({
            "confidence_threshold": 0.6,
            "require_human_review_below": 0.4,
            "use_keyword_validation": True,
        })

    def _validate(self, category, confidence, text):
        result = ClassificationResult(
            category=category, confidence=confidence, reasoning="Test",
        )
        return self.validator.validate(result, text)

    def test_pat1_lung_squamous(self):
        text = "Küçük hücreli dışı karsinom, bronkus biyopsi. P40 pozitif TTF-1 negatif. Skuamöz hücreli karsinom."
        v = self._validate("Lung", 0.92, text)
        assert v.validated_category == "Lung"

    def test_pat8_lung_pneumonectomy(self):
        text = "Skuamöz hücreli karsinom az diferansiye sol akciğer alt lob. Pnömonektomi. p40 CK5/6 pozitif."
        v = self._validate("Lung", 0.94, text)
        assert v.validated_category == "Lung"

    def test_pat7_brain_glioblastom(self):
        text = "Serebrum beyin tümör rezeksiyonu. Glioblastom IDH wild tip DSÖ derece 4. ATRX GFAP pozitif Ki67 %75."
        v = self._validate("Brain", 0.95, text)
        assert v.validated_category == "Brain"

    def test_pat4_blood_aml(self):
        text = "AML tanı. Blast %25. Myeloid panel. Karyotip. FISH. Allojenik nakil planlanıyor."
        v = self._validate("Blood", 0.85, text)
        assert v.validated_category == "Blood"

    def test_pat5_bone_marrow(self):
        text = "KİBX kemik iliği biyopsisi. Myeloid paneli. Polisitemi şüphesi. JAK2 mutasyonu."
        v = self._validate("Bone_Marrow", 0.80, text)
        assert v.validated_category == "Bone_Marrow"

    def test_pat6_bone_marrow_jak2(self):
        text = "Polisitemi sebebiyle JAK2 mutasyonu çalışması. Aksaray'dan yönlendirilen hasta."
        v = self._validate("Bone_Marrow", 0.75, text)
        assert v.validated_category == "Bone_Marrow"

    def test_pat3_ovary(self):
        text = "Seröz papiller kistadenokarsinom sağ ve sol over. Adenokarsinoma yayılımı rektum peritonu."
        v = self._validate("Ovary", 0.90, text)
        assert v.validated_category == "Ovary"

    def test_pat2_lung_ngs(self):
        text = "Akciğer CA tanılı hastanın EGFR ALK ROS-1 PDL-1 NGS çalışması. C34.9 bronş akciğer malign neoplazmi."
        v = self._validate("Lung", 0.88, text)
        assert v.validated_category == "Lung"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
