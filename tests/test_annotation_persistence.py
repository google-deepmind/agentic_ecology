# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Exercise annotation persistence with synthetic databases and real files."""

from concurrent.futures import ThreadPoolExecutor
import errno
import http.client
import http.server
import importlib.util
import json
import os
from pathlib import Path
import stat
import tempfile
import threading
import unittest
from unittest import mock

SERVER_PATH = (
    Path(__file__).resolve().parents[1]
    / "skills"
    / "agentic-ecology-ui"
    / "assets"
    / "server.py"
)
spec = importlib.util.spec_from_file_location("ecology_ui_server", SERVER_PATH)
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


class AnnotationPersistenceTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.path = self.root / "birds.json"
        self.data = {
            "items": [
                {
                    "id": 1,
                    "title": "Original item",
                    "media_type": "audio",
                    "media_url": "/media/item.wav",
                    "query_sim": 0.5,
                }
            ],
            "annotations": {"1": {"bird": "positive", "wind": "negative"}},
        }
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        self.adapter = server.JSONDatabaseAdapter(str(self.root), str(self.root))
        self.adapter.get_labels("birds")
        self.original_bytes = self.path.read_bytes()

    def assert_original(self):
        self.assertEqual(self.path.read_bytes(), self.original_bytes)
        self.assertEqual(self.adapter.loaded_databases["birds"], self.data)
        self.assertEqual(set(self.root.iterdir()), {self.path})

    def fail_mid_write(self, payload, file, **kwargs):
        file.write('{"items": [')
        file.flush()
        raise OSError(errno.ENOSPC, "synthetic full disk")

    def test_partial_write_failure_preserves_file_and_cached_annotations(self):
        for annotation in (
            server.AnnotationValue.NEGATIVE,
            server.AnnotationValue.UNCERTAIN,
        ):
            with self.subTest(annotation=annotation):
                with mock.patch.object(
                    server.json, "dump", side_effect=self.fail_mid_write
                ):
                    with self.assertRaises(OSError) as caught:
                        self.adapter.annotate("birds", 1, "bird", annotation)
                self.assertEqual(caught.exception.errno, errno.ENOSPC)
                self.assert_original()

    def test_failed_new_annotation_is_not_added_to_the_cache(self):
        with mock.patch.object(server.json, "dump", side_effect=self.fail_mid_write):
            with self.assertRaises(OSError):
                self.adapter.annotate(
                    "birds", 7, "new label", server.AnnotationValue.POSITIVE
                )
        self.assert_original()

    def test_serialization_failure_preserves_the_last_good_database(self):
        with mock.patch.object(
            server.json, "dump", side_effect=TypeError("synthetic serialization error")
        ):
            with self.assertRaisesRegex(TypeError, "synthetic serialization error"):
                self.adapter.annotate(
                    "birds", 1, "bird", server.AnnotationValue.NEGATIVE
                )
        self.assert_original()

    def test_failed_replacement_preserves_file_and_cached_annotations(self):
        with mock.patch.object(
            server.os, "replace", side_effect=PermissionError("synthetic replace error")
        ):
            with self.assertRaisesRegex(PermissionError, "synthetic replace error"):
                self.adapter.annotate(
                    "birds", 1, "bird", server.AnnotationValue.UNCERTAIN
                )
        self.assert_original()

    def test_replacement_receives_complete_json_after_the_temporary_file_closes(self):
        replace = os.replace
        original_mode = stat.S_IMODE(self.path.stat().st_mode)
        observed = []

        def inspect_replace(source, destination):
            self.assertEqual(Path(destination), self.path)
            self.assertEqual(Path(source).parent, self.root)
            self.assertNotEqual(Path(source), self.path)
            self.assertEqual(self.path.read_bytes(), self.original_bytes)
            payload = json.loads(Path(source).read_text(encoding="utf-8"))
            self.assertEqual(payload["annotations"]["1"]["bird"], "negative")
            self.assertEqual(payload["items"], self.data["items"])
            observed.append(source)
            return replace(source, destination)

        with mock.patch.object(server.os, "replace", side_effect=inspect_replace):
            self.adapter.annotate("birds", 1, "bird", server.AnnotationValue.NEGATIVE)
        self.assertEqual(len(observed), 1)
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), original_mode)
        self.assertEqual(set(self.root.iterdir()), {self.path})
        reloaded = server.JSONDatabaseAdapter(str(self.root), str(self.root))
        self.assertEqual(
            reloaded.search("birds", server.SearchMode.QUERY, "bird", "")["results"][0][
                "annotation"
            ],
            "negative",
        )

    def test_successful_retry_does_not_commit_a_failed_annotation(self):
        with mock.patch.object(server.json, "dump", side_effect=self.fail_mid_write):
            with self.assertRaises(OSError):
                self.adapter.annotate(
                    "birds", 7, "failed", server.AnnotationValue.POSITIVE
                )
        self.adapter.annotate("birds", 1, "bird", server.AnnotationValue.NEGATIVE)
        saved = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertNotIn("7", saved["annotations"])
        self.assertEqual(
            saved["annotations"]["1"], {"bird": "negative", "wind": "negative"}
        )
        self.assertEqual(saved, self.adapter.loaded_databases["birds"])

    def test_failed_save_restores_an_originally_absent_annotations_key(self):
        data = {"items": self.data["items"]}
        self.path.write_text(json.dumps(data), encoding="utf-8")
        adapter = server.JSONDatabaseAdapter(str(self.root), str(self.root))
        original = self.path.read_bytes()
        with mock.patch.object(server.json, "dump", side_effect=self.fail_mid_write):
            with self.assertRaises(OSError):
                adapter.annotate("birds", 1, "bird", server.AnnotationValue.POSITIVE)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(adapter.loaded_databases["birds"], data)
        self.assertEqual(set(self.root.iterdir()), {self.path})

    def test_unloaded_database_save_does_not_create_a_file(self):
        adapter = server.JSONDatabaseAdapter(str(self.root), str(self.root))
        adapter._save_database("unused")
        self.assert_original()

    def test_clearing_a_label_preserves_other_items_and_labels(self):
        self.adapter.annotate("birds", 1, "bird", server.AnnotationValue.UNCERTAIN)
        saved = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(saved["annotations"], {"1": {"wind": "negative"}})
        self.assertEqual(saved["items"], self.data["items"])
        self.adapter.annotate("birds", 1, "wind", server.AnnotationValue.UNCERTAIN)
        self.assertEqual(json.loads(self.path.read_text())["annotations"], {})

    def test_concurrent_annotations_remain_serialized_and_reloadable(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(
                    self.adapter.annotate,
                    "birds",
                    i,
                    "bird",
                    server.AnnotationValue.POSITIVE,
                )
                for i in range(2, 18)
            ]
            for future in futures:
                future.result(timeout=10)
        saved = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(len(saved["annotations"]), 17)
        self.assertEqual(saved["items"], self.data["items"])
        self.assertEqual(saved, self.adapter.loaded_databases["birds"])
        self.assertEqual(set(self.root.iterdir()), {self.path})

    def test_http_failure_leaves_the_persisted_annotation_intact(self):
        handler = type(
            "QuietHandler", (server.Handler,), {"log_message": lambda *args: None}
        )
        httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        with mock.patch.object(server, "ACTIVE_ADAPTER", self.adapter):
            thread.start()
            try:
                conn = http.client.HTTPConnection(*httpd.server_address, timeout=5)
                self.addCleanup(conn.close)
                body = json.dumps(
                    {
                        "database": "birds",
                        "item_id": 1,
                        "label": "bird",
                        "annotation": "negative",
                    }
                )
                with mock.patch.object(
                    server.json, "dump", side_effect=self.fail_mid_write
                ):
                    conn.request(
                        "POST",
                        "/api/annotation",
                        body,
                        {"Content-Type": "application/json"},
                    )
                    response = conn.getresponse()
                    self.assertEqual(response.status, 500)
                    self.assertIn(
                        "synthetic full disk", json.loads(response.read())["error"]
                    )
                self.assert_original()
            finally:
                httpd.shutdown()
                thread.join(timeout=5)
                httpd.server_close()


if __name__ == "__main__":
    unittest.main()
