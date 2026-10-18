import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from omniembed.demo_import import ImportValidationError, materialize, preview


JPEG = b"\xff\xd8\xff\xe0" + b"small-jpeg-fixture"
MP4 = b"\x00\x00\x00\x18ftypmp42" + b"small-video-fixture"


class DemoImportTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="demo-import-test-")
        self.base = Path(self.temporary.name)
        self.rows = [
            {"photo_id": "100", "keyword_top": "小猫", "keywords": ["小猫", "猫咪"],
             "cover_file": "covers/100.jpg", "video_file": "videos/100.mp4", "cover_status": "OK", "video_status": "OK",
             "click_cnt_total": 12, "kw_detail": [{"keyword": "小猫", "click_cnt": 12, "click_dv": 8}]},
            {"photo_id": "101", "keyword_top": "小狗", "keywords": ["小狗"],
             "cover_file": "covers/101.jpg", "video_file": "videos/101.mp4", "video_status": "FAIL"},
        ]
        self.metadata = {"source": {"private_table": "internal.source.only"}, "summary": {"total": 2}, "records": self.rows}

    def tearDown(self):
        self.temporary.cleanup()

    def archive(self, extra=(), metadata=None, name="input.tar"):
        path = self.base / name
        data = {"export/metadata.json": json.dumps(metadata or self.metadata, ensure_ascii=False).encode(),
                "export/metadata.csv": b"photo_id,keyword_top\n100,cat\n",
                "export/covers/100.jpg": JPEG, "export/covers/101.jpg": JPEG,
                "export/videos/100.mp4": MP4,
                "export/gallery.html": b"<script>throw 'must never serve';</script>",
                "export/fetch.py": b"raise Exception('must never execute')",
                "export/query.sql": b"DROP TABLE anything;",
                "export/manifest.csv": b"path\n/data_agent/private/never-follow.mp4\n"}
        with tarfile.open(path, "w") as tar:
            for member_name, content in data.items():
                info = tarfile.TarInfo(member_name)
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))
            for info, content in extra:
                tar.addfile(info, io.BytesIO(content) if content is not None else None)
        return path

    def test_preview_and_materialize_missing_video_preserves_row(self):
        path = self.archive()
        before = path.read_bytes()
        inspected = preview(path)
        self.assertEqual((inspected["total_assets"], inspected["cover_count"], inspected["video_count"], inspected["missing_video_count"]), (2, 2, 1, 1))
        self.assertNotIn("kw_detail", json.dumps(inspected["sample"]))
        self.assertNotIn("internal.source.only", json.dumps(inspected))
        self.assertTrue(any("不是人工相关性标签" in w for w in inspected["warnings"]))
        progress = []
        result = materialize(path, self.base / "sources", progress=progress.append)
        self.assertEqual(len(result["assets"]), 2)
        first, second = result["assets"]
        self.assertEqual(first["title"], "搜索词：小猫")
        self.assertEqual(first["title_source"], "search_keyword")
        self.assertEqual(first["metadata"]["kw_detail"][0]["click_cnt"], 12)
        self.assertTrue(Path(second["cover_path"]).is_file())
        self.assertIsNone(second["video_path"])
        self.assertEqual(first["cover_path"], second["cover_path"])
        folder = self.base / "sources" / result["dataset"]["source_sha256"]
        self.assertEqual(len(list((folder / "media").iterdir())), 2)
        self.assertFalse(list(folder.rglob("*.html")))
        self.assertFalse(list(folder.rglob("*.py")))
        self.assertFalse(list(folder.rglob("*.sql")))
        self.assertFalse(list(folder.rglob("manifest.csv")))
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(progress[-1]["phase"], "complete")
        self.assertEqual(materialize(path, self.base / "sources"), result)

    def test_directory_input_copies_to_destination(self):
        source = self.base / "folder"
        (source / "covers").mkdir(parents=True)
        (source / "videos").mkdir()
        (source / "metadata.json").write_text(json.dumps(self.metadata))
        (source / "covers/100.jpg").write_bytes(JPEG)
        (source / "covers/101.jpg").write_bytes(JPEG)
        (source / "videos/100.mp4").write_bytes(MP4)
        result = materialize(source, self.base / "sources")
        self.assertEqual(result["dataset"]["source_type"], "directory")
        copied = Path(result["assets"][0]["video_path"])
        self.assertIn((self.base / "sources").resolve(), copied.parents)
        self.assertEqual(copied.read_bytes(), MP4)
        (source / "videos/100.mp4").write_bytes(b"changed original")
        self.assertEqual(copied.read_bytes(), MP4)
        with self.assertRaises(ImportValidationError):
            materialize(source, source / "unsafe-output")

    def test_duplicate_records_removed_but_conflicts_rejected(self):
        duplicate = {**self.metadata, "records": self.rows + [dict(self.rows[0])]}
        path = self.archive(metadata=duplicate)
        self.assertEqual(preview(path)["duplicate_records_removed"], 1)
        self.assertEqual(len(materialize(path, self.base / "sources")["assets"]), 2)
        conflict = {**self.metadata, "records": self.rows + [{**self.rows[0], "keyword_top": "conflict"}]}
        with self.assertRaises(ImportValidationError):
            preview(self.archive(metadata=conflict, name="conflict.tar"))

    def test_member_paths_links_duplicates_and_special_types_rejected(self):
        for index, member in enumerate(["../escape.jpg", "/absolute.jpg", "C:\\escape.jpg", "export/metadata.json"]):
            with self.subTest(path=member):
                info = tarfile.TarInfo(member)
                info.size = 1
                with self.assertRaises(ImportValidationError):
                    preview(self.archive(extra=[(info, b"x")], name="bad-path-{}.tar".format(index)))
        for index, kind in enumerate([tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.CHRTYPE, tarfile.FIFOTYPE]):
            with self.subTest(kind=kind):
                info = tarfile.TarInfo("export/ignored-link")
                info.type, info.linkname = kind, "/etc/passwd"
                with self.assertRaises(ImportValidationError):
                    preview(self.archive(extra=[(info, None)], name="bad-type-{}.tar".format(index)))

    def test_unsafe_metadata_reference_is_not_followed(self):
        data = {**self.metadata, "records": [{**self.rows[0], "cover_file": "/etc/passwd", "video_file": "../../secret.mp4"}]}
        result = materialize(self.archive(metadata=data), self.base / "sources")
        self.assertIsNone(result["assets"][0]["cover_path"])
        self.assertIsNone(result["assets"][0]["video_path"])
        self.assertTrue(any("不安全" in warning for warning in result["assets"][0]["warnings"]))

    def test_cancel_and_fake_media_leave_no_partial_dataset(self):
        path = self.archive()
        with self.assertRaises(InterruptedError):
            materialize(path, self.base / "sources", cancelled=lambda: True)
        self.assertFalse((self.base / "sources").exists())
        with patch("omniembed.demo_import._magic", return_value=False):
            with self.assertRaises(ImportValidationError):
                materialize(path, self.base / "sources")
        self.assertEqual(list((self.base / "sources").iterdir()), [])

    def test_directory_links_and_hardlinks_rejected(self):
        source = self.base / "linked"
        source.mkdir()
        (source / "metadata.json").write_text(json.dumps(self.metadata))
        (source / "link").symlink_to("/etc/passwd")
        with self.assertRaises(ImportValidationError):
            preview(source)
        (source / "link").unlink()
        os.link(source / "metadata.json", source / "hardlink.json")
        with self.assertRaises(ImportValidationError):
            preview(source)

    def test_csv_fallback_is_data_only(self):
        source = self.base / "csv-folder"
        (source / "covers").mkdir(parents=True)
        (source / "covers/1.jpg").write_bytes(JPEG)
        (source / "metadata.csv").write_text('photo_id,keyword_top,cover_file,video_file,keywords\n1,cat,covers/1.jpg,videos/missing.mp4,"[""cat""]"\n')
        result = materialize(source, self.base / "sources")
        self.assertEqual(result["assets"][0]["keywords"], ["cat"])
        self.assertIsNone(result["assets"][0]["video_path"])
        self.assertEqual(result["dataset"]["summary"]["missing_video_count"], 1)

    def test_size_and_entry_limits_rejected_before_payload_read(self):
        path = self.archive()
        with patch("omniembed.demo_import.MAX_ENTRIES", 2):
            with self.assertRaises(ImportValidationError):
                preview(path)
        with patch("omniembed.demo_import.MAX_TOTAL_BYTES", 1):
            with self.assertRaises(ImportValidationError):
                preview(path)
        with patch("omniembed.demo_import.MAX_METADATA_BYTES", 1):
            with self.assertRaises(ImportValidationError):
                preview(path)


if __name__ == "__main__":
    unittest.main()
