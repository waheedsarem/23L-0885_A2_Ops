"""Controlled fixtures test audit behaviour, not real-dataset statistics."""
import hashlib
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_dataset import inspect_file, summarize


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for folder in ("AiArtData", "RealArt"):
            (self.root / folder).mkdir()
        self.mapping = {"AiArtData": "AI-generated", "RealArt": "real"}

    def tearDown(self):
        self.temp.cleanup()

    def inspect(self, relative):
        return inspect_file(self.root / relative, self.root, self.mapping)

    def test_metadata_uses_content_not_extension(self):
        path = self.root / "AiArtData" / "actually_png.jpg"
        Image.new("RGBA", (12, 7), (12, 23, 34, 255)).save(path, format="PNG")
        row = self.inspect("AiArtData/actually_png.jpg")
        self.assertEqual((row["width"], row["height"], row["colour_mode"], row["format"]), (12, 7, "RGBA", "PNG"))
        self.assertEqual(row["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(row["status"], "ok")

    def test_corrupt_and_non_image_are_distinguished(self):
        (self.root / "RealArt" / "broken.jpg").write_bytes(b"not an image")
        (self.root / "README.txt").write_text("dataset description")
        self.assertEqual(self.inspect("RealArt/broken.jpg")["status"], "corrupt_or_unreadable")
        self.assertEqual(self.inspect("README.txt")["status"], "non_image")

    def test_truncated_pixels_are_detected(self):
        path = self.root / "RealArt" / "truncated.bmp"
        Image.new("RGB", (30, 20), "red").save(path)
        path.write_bytes(path.read_bytes()[:-100])
        self.assertEqual(self.inspect("RealArt/truncated.bmp")["status"], "corrupt_or_unreadable")

    def test_duplicates_cross_labels_and_statistics(self):
        original = self.root / "AiArtData" / "a.png"
        Image.new("RGB", (10, 20), "blue").save(original)
        shutil.copyfile(original, self.root / "AiArtData" / "b.png")
        shutil.copyfile(original, self.root / "RealArt" / "c.png")
        rows = [self.inspect(p) for p in ("AiArtData/a.png", "AiArtData/b.png", "RealArt/c.png")]
        counts, numbers, groups = summarize(rows)
        self.assertEqual(len(groups), 1)
        self.assertEqual(counts[-1]["duplicate_excess_within_class"], 2)
        self.assertTrue(all(row["cross_class_duplicate"] for row in rows))
        self.assertEqual(next(row for row in numbers if row["class"] == "ALL" and row["metric"] == "width")["median"], 10)

    def test_extensionless_image_and_unmapped_label(self):
        Image.new("RGB", (3, 4)).save(self.root / "image_without_extension", format="PNG")
        row = self.inspect("image_without_extension")
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["class"], "UNMAPPED")


if __name__ == "__main__":
    unittest.main(verbosity=2)
