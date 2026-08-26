from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from publication_data import load_publication_data, robustness_report  # noqa: E402
from render_publication_docs import (  # noqa: E402
    load_context,
    render_readme,
    render_release_notes,
    render_summary,
)


class PublicationArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = load_publication_data(REPO_ROOT / "results" / "derived")
        cls.context = load_context()

    def test_primary_metrics_and_paired_difference(self) -> None:
        conditions = self.data["condition_records"]
        self.assertAlmostEqual(conditions["turns_3"]["overall"]["run_success_rate"], 41 / 120)
        self.assertAlmostEqual(conditions["turns_5"]["overall"]["run_success_rate"], 60 / 120)
        paired = self.data["paired_records"]["overall"]
        self.assertAlmostEqual(paired["mean_task_success_rate_difference"], 19 / 120)
        self.assertEqual(int(paired["tasks_improved"]), 9)
        self.assertEqual(int(paired["tasks_declined"]), 2)
        self.assertEqual(int(paired["tasks_tied"]), 13)

    def test_reliability_metrics(self) -> None:
        conditions = self.data["condition_records"]
        t3 = conditions["turns_3"]["overall"]
        t5 = conditions["turns_5"]["overall"]
        self.assertAlmostEqual(t3["flaky_task_rate"], 7 / 24)
        self.assertAlmostEqual(t5["flaky_task_rate"], 10 / 24)
        self.assertAlmostEqual(t3["mean_pairwise_disagreement"], 0.15)
        self.assertAlmostEqual(t5["mean_pairwise_disagreement"], 13 / 60)

    def test_movement_partition_and_concentration(self) -> None:
        report = robustness_report(self.data)
        movement = report["task_movements"]
        self.assertEqual(len(movement["improved"]), 9)
        self.assertEqual(len(movement["declined"]), 2)
        self.assertEqual(len(movement["tied"]), 13)
        declined = {row["task_id"] for row in movement["declined"]}
        self.assertEqual(
            declined,
            {
                "madhuraatmarambhagat_crop-recommendation-dataset_class",
                "digantabhattacharya_usa-house-price-index-and-macroeconomic-variables_reg",
            },
        )
        self.assertEqual(report["improvement_concentration"]["classification"], 13)
        self.assertEqual(report["improvement_concentration"]["regression"], 6)
        self.assertEqual(report["improvement_concentration"]["total"], 19)

    def test_documentation_is_exact_deterministic_render(self) -> None:
        self.assertEqual(
            (REPO_ROOT / "README.md").read_text(encoding="utf-8"),
            render_readme(self.context),
        )
        self.assertEqual(
            (REPO_ROOT / "docs" / "research_summary.md").read_text(encoding="utf-8"),
            render_summary(self.context),
        )
        self.assertEqual(
            (REPO_ROOT / "docs" / "releases" / "v1.0.0-phase1.md").read_text(
                encoding="utf-8"
            ),
            render_release_notes(self.context),
        )
        self.assertIn("| 3 turns | 5/24 (20.8%)", render_readme(self.context))

    def test_public_positioning_is_explicit(self) -> None:
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("independent reliability study", readme)
        self.assertIn("not a new benchmark", readme)
        self.assertIn("does not imply affiliation with Snowflake", readme)
        self.assertIn("No formal significance claim is made", readme)

    def test_readme_local_links_resolve(self) -> None:
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        targets = re.findall(r"\]\(([^)]+)\)", readme)
        local_targets = [target.split("#", 1)[0] for target in targets if not target.startswith("http")]
        self.assertTrue(local_targets)
        for target in local_targets:
            self.assertTrue((REPO_ROOT / target).exists(), f"README link does not resolve: {target}")

    def test_figure_manifest_and_rendered_files(self) -> None:
        manifest = json.loads(
            (REPO_ROOT / "results" / "figures" / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["input_directory"].replace("\\", "/"), "results/derived")
        self.assertEqual(manifest["source_sha256"], self.data["source_hashes"])
        self.assertEqual(len(manifest["figures"]), 7)
        for figure in manifest["figures"]:
            png = REPO_ROOT / figure["files"]["png"]
            pdf = REPO_ROOT / figure["files"]["pdf"]
            self.assertTrue(png.is_file() and png.stat().st_size > 20_000)
            self.assertTrue(pdf.is_file() and pdf.stat().st_size > 10_000)
            with Image.open(png) as image:
                self.assertGreaterEqual(image.width, 1_800)
                self.assertGreaterEqual(image.height, 1_000)
                dpi = image.info.get("dpi", (0, 0))
                self.assertAlmostEqual(dpi[0], 300, delta=1)
                self.assertAlmostEqual(dpi[1], 300, delta=1)


if __name__ == "__main__":
    unittest.main()
