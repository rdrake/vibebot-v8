"""Service images: saving, download, generation, draw context/rewrite, SSRF protection."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import pytest
from llm.service import LLMService

from .conftest import make_completion_response

if TYPE_CHECKING:
    from unittest.mock import Mock

    from pytest_mock import MockerFixture


class TestImageSaving:
    """Tests for save_image_to_http and _save_image_bytes functionality."""

    @pytest.fixture(autouse=True)
    def setup(self, make_service, mocker: MockerFixture) -> None:
        """Set up test fixtures."""
        self.mocker = mocker
        self.service, self.mock_plugin = make_service(
            httpRoot="/tmp/test_llm_images",
            httpUrlBase="https://example.com/llm",
            fileCleanupAge=24,
            fileCleanupMax=1000,
        )

    # --- save_image_to_http tests ---

    def test_save_image_to_http_success(self, tmp_path: object) -> None:
        """GIVEN valid base64 image WHEN saving THEN returns URL."""
        import base64

        # Mock config to use temp directory
        self.mock_plugin.registryValue = self.mocker.Mock(
            side_effect=lambda key, channel=None: {
                "httpRoot": str(tmp_path),
                "httpUrlBase": "https://example.com/llm",
                "fileCleanupAge": 24,
                "fileCleanupMax": 1000,
            }.get(key)
        )

        # Create simple PNG-like data
        image_data = b"\x89PNG\r\n\x1a\n" + b"fake image data"
        b64_data = base64.b64encode(image_data).decode()

        result = self.service.save_image_to_http(b64_data)

        assert result is not None
        assert result.startswith("https://example.com/llm/img_")
        assert result.endswith(".png")

    def test_save_image_to_http_custom_extension(self, tmp_path: object) -> None:
        """GIVEN custom extension WHEN saving THEN uses that extension."""
        import base64

        self.mock_plugin.registryValue = self.mocker.Mock(
            side_effect=lambda key, channel=None: {
                "httpRoot": str(tmp_path),
                "httpUrlBase": "https://example.com/llm",
                "fileCleanupAge": 24,
                "fileCleanupMax": 1000,
            }.get(key)
        )

        image_data = b"fake jpeg data"
        b64_data = base64.b64encode(image_data).decode()

        result = self.service.save_image_to_http(b64_data, extension="jpg")

        assert result is not None
        assert result.endswith(".jpg")

    def test_save_image_to_http_invalid_base64(self) -> None:
        """GIVEN invalid base64 WHEN saving THEN returns None and logs error."""
        result = self.service.save_image_to_http("not valid base64!!!")

        # Error is logged via service's own logger (not plugin.log)
        assert result is None

    # --- _save_image_bytes tests ---

    def test_save_image_bytes_success(self, tmp_path: object) -> None:
        """GIVEN valid image bytes WHEN saving THEN returns URL."""
        self.mock_plugin.registryValue = self.mocker.Mock(
            side_effect=lambda key, channel=None: {
                "httpRoot": str(tmp_path),
                "httpUrlBase": "https://example.com/llm",
            }.get(key)
        )

        image_data = b"\x89PNG\r\n\x1a\n" + b"fake image data"
        result = self.service._save_image_bytes(image_data)

        assert result is not None
        assert result.startswith("https://example.com/llm/img_")
        assert result.endswith(".png")

    def test_save_image_bytes_custom_extension(self, tmp_path: object) -> None:
        """GIVEN custom extension WHEN saving THEN uses that extension."""
        self.mock_plugin.registryValue = self.mocker.Mock(
            side_effect=lambda key, channel=None: {
                "httpRoot": str(tmp_path),
                "httpUrlBase": "https://example.com/llm",
            }.get(key)
        )

        result = self.service._save_image_bytes(b"fake jpeg data", extension="jpg")

        assert result is not None
        assert result.endswith(".jpg")

    def test_save_image_bytes_magic_bytes_override_extension(self, tmp_path: object) -> None:
        """GIVEN JPEG magic bytes but extension='png' WHEN saving THEN uses jpg."""
        self.mock_plugin.registryValue = self.mocker.Mock(
            side_effect=lambda key, channel=None: {
                "httpRoot": str(tmp_path),
                "httpUrlBase": "https://example.com/llm",
            }.get(key)
        )

        jpeg_data = b"\xff\xd8\xff\xe0" + b"fake jpeg payload"
        result = self.service._save_image_bytes(jpeg_data, extension="png")

        assert result is not None
        assert result.endswith(".jpg")

    # --- _detect_image_format tests ---

    def test_detect_image_format_png(self) -> None:
        """GIVEN PNG magic bytes THEN returns 'png'."""
        assert self.service._detect_image_format(b"\x89PNG\r\n\x1a\ndata") == "png"

    def test_detect_image_format_jpeg(self) -> None:
        """GIVEN JPEG magic bytes THEN returns 'jpg'."""
        assert self.service._detect_image_format(b"\xff\xd8\xff\xe0data") == "jpg"

    def test_detect_image_format_webp(self) -> None:
        """GIVEN WebP magic bytes THEN returns 'webp'."""
        assert self.service._detect_image_format(b"RIFF\x00\x00\x00\x00WEBPdata") == "webp"

    def test_detect_image_format_gif(self) -> None:
        """GIVEN GIF magic bytes THEN returns 'gif'."""
        assert self.service._detect_image_format(b"GIF89adata") == "gif"

    def test_detect_image_format_unknown(self) -> None:
        """GIVEN unknown bytes THEN returns None."""
        assert self.service._detect_image_format(b"unknown data") is None

    def test_convert_png_to_jpeg(self) -> None:
        """GIVEN a real PNG image WHEN converting THEN returns JPEG bytes."""
        from io import BytesIO

        from PIL import Image

        # Create a real 1x1 red PNG
        img = Image.new("RGB", (1, 1), color=(255, 0, 0))
        buf = BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        jpeg_bytes, ext = self.service._convert_png_to_jpeg(png_bytes)
        assert ext == "jpg"
        assert jpeg_bytes[:3] == b"\xff\xd8\xff"

    def test_convert_png_to_jpeg_rgba(self) -> None:
        """GIVEN RGBA PNG WHEN converting THEN strips alpha and returns JPEG."""
        from io import BytesIO

        from PIL import Image

        img = Image.new("RGBA", (1, 1), color=(255, 0, 0, 128))
        buf = BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        jpeg_bytes, ext = self.service._convert_png_to_jpeg(png_bytes)
        assert ext == "jpg"
        assert jpeg_bytes[:3] == b"\xff\xd8\xff"

    def test_convert_invalid_png_falls_back(self) -> None:
        """GIVEN invalid PNG data WHEN converting THEN falls back to original."""
        bad_data = b"\x89PNG\r\n\x1a\ngarbage"
        result_bytes, ext = self.service._convert_png_to_jpeg(bad_data)
        assert ext == "png"
        assert result_bytes == bad_data

    def test_save_real_png_becomes_jpeg(self, tmp_path: object) -> None:
        """GIVEN a real PNG image WHEN saving THEN file is saved as JPEG."""
        from io import BytesIO
        from pathlib import Path

        from PIL import Image

        self.mock_plugin.registryValue = self.mocker.Mock(
            side_effect=lambda key, channel=None: {
                "httpRoot": str(tmp_path),
                "httpUrlBase": "https://example.com/llm",
            }.get(key)
        )

        img = Image.new("RGB", (1, 1), color=(0, 128, 255))
        buf = BytesIO()
        img.save(buf, format="PNG")

        result = self.service._save_image_bytes(buf.getvalue())
        assert result is not None
        assert result.endswith(".jpg")

        jpg_files = list(Path(str(tmp_path)).glob("img_*.jpg"))
        assert len(jpg_files) == 1

    def test_save_image_bytes_writes_file(self, tmp_path: object) -> None:
        """GIVEN image bytes WHEN saving THEN file exists on disk."""
        from pathlib import Path

        self.mock_plugin.registryValue = self.mocker.Mock(
            side_effect=lambda key, channel=None: {
                "httpRoot": str(tmp_path),
                "httpUrlBase": "https://example.com/llm",
            }.get(key)
        )

        image_data = b"\x89PNG\r\n\x1a\nfake"
        self.service._save_image_bytes(image_data)

        png_files = list(Path(str(tmp_path)).glob("img_*.png"))
        assert len(png_files) == 1
        assert png_files[0].read_bytes() == image_data


class TestDownloadAndSaveImage:
    """Tests for _download_and_save_image functionality."""

    @pytest.fixture(autouse=True)
    def setup(self, make_service, mocker: MockerFixture) -> None:
        """Set up test fixtures."""
        self.mocker = mocker
        self.service, self.mock_plugin = make_service(
            httpRoot="/tmp/test_llm_images",
            httpUrlBase="https://example.com/llm",
            drawTimeout=60,
            timeout=30,
            fileCleanupAge=24,
            fileCleanupMax=1000,
        )

    def test_download_success(self) -> None:
        """GIVEN valid image URL WHEN downloading THEN returns local URL."""
        mock_resp = self.mocker.Mock()
        mock_resp.read.return_value = b"\x89PNG\r\n\x1a\nfake"
        mock_resp.headers = {"Content-Type": "image/png"}
        mock_resp.__enter__ = self.mocker.Mock(return_value=mock_resp)
        mock_resp.__exit__ = self.mocker.Mock(return_value=False)

        mock_opener = self.mocker.Mock()
        mock_opener.open.return_value = mock_resp
        self.mocker.patch("urllib.request.build_opener", return_value=mock_opener)
        mock_save = self.mocker.patch.object(
            self.service,
            "_save_image_bytes",
            return_value="https://example.com/llm/img_abc.png",
        )
        result = self.service._download_and_save_image("https://provider.com/img.png")

        assert result == "https://example.com/llm/img_abc.png"
        mock_save.assert_called_once_with(b"\x89PNG\r\n\x1a\nfake", "png")

    def test_download_jpeg_content_type(self) -> None:
        """GIVEN JPEG content type WHEN downloading THEN uses jpg extension."""
        mock_resp = self.mocker.Mock()
        mock_resp.read.return_value = b"fake jpeg"
        mock_resp.headers = {"Content-Type": "image/jpeg"}
        mock_resp.__enter__ = self.mocker.Mock(return_value=mock_resp)
        mock_resp.__exit__ = self.mocker.Mock(return_value=False)

        mock_opener = self.mocker.Mock()
        mock_opener.open.return_value = mock_resp
        self.mocker.patch("urllib.request.build_opener", return_value=mock_opener)
        mock_save = self.mocker.patch.object(self.service, "_save_image_bytes", return_value="url")
        self.service._download_and_save_image("https://provider.com/img")

        mock_save.assert_called_once_with(b"fake jpeg", "jpg")

    def test_download_infers_extension_from_url(self) -> None:
        """GIVEN no content type WHEN URL has extension THEN infers from URL."""
        mock_resp = self.mocker.Mock()
        mock_resp.read.return_value = b"fake webp"
        mock_resp.headers = {"Content-Type": "application/octet-stream"}
        mock_resp.__enter__ = self.mocker.Mock(return_value=mock_resp)
        mock_resp.__exit__ = self.mocker.Mock(return_value=False)

        mock_opener = self.mocker.Mock()
        mock_opener.open.return_value = mock_resp
        self.mocker.patch("urllib.request.build_opener", return_value=mock_opener)
        mock_save = self.mocker.patch.object(self.service, "_save_image_bytes", return_value="url")
        self.service._download_and_save_image("https://provider.com/img.webp")

        mock_save.assert_called_once_with(b"fake webp", "webp")

    def test_download_defaults_to_png(self) -> None:
        """GIVEN no content type and no URL extension WHEN downloading THEN defaults to png."""
        mock_resp = self.mocker.Mock()
        mock_resp.read.return_value = b"mystery image"
        mock_resp.headers = {"Content-Type": ""}
        mock_resp.__enter__ = self.mocker.Mock(return_value=mock_resp)
        mock_resp.__exit__ = self.mocker.Mock(return_value=False)

        mock_opener = self.mocker.Mock()
        mock_opener.open.return_value = mock_resp
        self.mocker.patch("urllib.request.build_opener", return_value=mock_opener)
        mock_save = self.mocker.patch.object(self.service, "_save_image_bytes", return_value="url")
        self.service._download_and_save_image("https://provider.com/generate?id=123")

        mock_save.assert_called_once_with(b"mystery image", "png")

    def test_download_too_large(self) -> None:
        """GIVEN image exceeds 20 MB WHEN downloading THEN returns None."""
        mock_resp = self.mocker.Mock()
        mock_resp.read.return_value = b"x" * (20 * 1024 * 1024 + 1)
        mock_resp.headers = {"Content-Type": "image/png"}
        mock_resp.__enter__ = self.mocker.Mock(return_value=mock_resp)
        mock_resp.__exit__ = self.mocker.Mock(return_value=False)

        mock_opener = self.mocker.Mock()
        mock_opener.open.return_value = mock_resp
        self.mocker.patch("urllib.request.build_opener", return_value=mock_opener)
        result = self.service._download_and_save_image("https://provider.com/huge.png")

        assert result is None

    def test_download_network_error(self) -> None:
        """GIVEN network error WHEN downloading THEN returns None."""
        import urllib.error

        mock_opener = self.mocker.Mock()
        mock_opener.open.side_effect = urllib.error.URLError("connection refused")
        self.mocker.patch("urllib.request.build_opener", return_value=mock_opener)
        result = self.service._download_and_save_image("https://provider.com/img.png")

        assert result is None

    def test_download_rejects_non_http_scheme(self) -> None:
        """GIVEN file:// URL WHEN downloading THEN refuses without fetching."""
        mock_build = self.mocker.patch("urllib.request.build_opener")
        result = self.service._download_and_save_image("file:///etc/passwd")

        assert result is None
        mock_build.assert_not_called()

    def test_download_rejects_loopback_literal(self) -> None:
        """GIVEN 127.0.0.1 URL WHEN downloading THEN refuses without fetching."""
        mock_build = self.mocker.patch("urllib.request.build_opener")
        result = self.service._download_and_save_image("http://127.0.0.1/img.png")

        assert result is None
        mock_build.assert_not_called()

    def test_download_rejects_private_literal(self) -> None:
        """GIVEN 192.168.x.x URL WHEN downloading THEN refuses without fetching."""
        mock_build = self.mocker.patch("urllib.request.build_opener")
        result = self.service._download_and_save_image("http://192.168.1.1/img.png")

        assert result is None
        mock_build.assert_not_called()

    def test_download_rejects_link_local_literal(self) -> None:
        """GIVEN 169.254.x.x URL WHEN downloading THEN refuses without fetching."""
        mock_build = self.mocker.patch("urllib.request.build_opener")
        result = self.service._download_and_save_image("http://169.254.169.254/latest")

        assert result is None
        mock_build.assert_not_called()

    def test_download_disables_redirects(self) -> None:
        """GIVEN download path WHEN building opener THEN installs a no-redirect handler."""
        import urllib.request

        captured: dict[str, object] = {}
        real_build = urllib.request.build_opener

        def capture_build(*handlers: object) -> object:
            captured["handlers"] = handlers
            return real_build(*handlers)

        self.mocker.patch("urllib.request.build_opener", side_effect=capture_build)
        # Force network call to bail without actually fetching
        self.mocker.patch.object(self.service, "_save_image_bytes", return_value=None)

        # We don't care if the open call fails — only that build_opener was
        # called with a HTTPRedirectHandler subclass that vetoes redirects.
        self.service._download_and_save_image("https://nonexistent.invalid/img.png")

        handlers = captured.get("handlers", ())
        assert any(
            isinstance(h, urllib.request.HTTPRedirectHandler)
            and h.redirect_request(None, None, None, None, None) is None  # type: ignore[arg-type]
            for h in handlers  # type: ignore[union-attr]
        )


class TestImageGenerationWithBase64:
    """Tests for image_generation with base64 handling and typing indicators."""

    @pytest.fixture(autouse=True)
    def setup(self, make_service, mocker: MockerFixture) -> None:
        """Set up test fixtures."""
        self.mocker = mocker
        self.service, self.mock_plugin = make_service(
            imageApiKey="test-api-key",
            imageModel="gemini/imagen-4.0-generate-001",
            timeout=30,
            maxPromptLength=10000,
            httpRoot="/tmp/test",
            httpUrlBase="https://example.com/llm",
            fileCleanupAge=24,
            fileCleanupMax=1000,
            drawAutoRewriteMax=0,
        )

    def _make_mock_irc(self, capabilities: set | None = None) -> Mock:
        """Create mock IRC with capability negotiation."""
        irc = self.mocker.Mock()
        irc.state = self.mocker.Mock()
        irc.state.capabilities_ack = capabilities or {"message-tags"}
        irc.queueMsg = self.mocker.Mock()
        return irc

    def _make_mock_msg(self, channel: str = "#test") -> Mock:
        """Create mock message."""
        msg = self.mocker.Mock()
        msg.args = (channel,)
        return msg

    def test_image_generation_with_url_response(self) -> None:
        """GIVEN provider returns URL WHEN generating THEN downloads and returns local URL."""
        mock_response = self.mocker.Mock()
        mock_response.data = [self.mocker.Mock(url="https://provider.com/image.png", b64_json=None)]

        self.mocker.patch("llm.service.litellm.image_generation", return_value=mock_response)
        mock_download = self.mocker.patch.object(
            self.service,
            "_download_and_save_image",
            return_value="https://example.com/llm/img_abc123.png",
        )
        result = self.service.image_generation("a cat")

        mock_download.assert_called_once_with("https://provider.com/image.png")
        assert result.content == "https://example.com/llm/img_abc123.png"

    def test_image_generation_url_download_failure_falls_back(self) -> None:
        """GIVEN provider returns URL and download fails WHEN generating THEN falls back to provider URL."""
        mock_response = self.mocker.Mock()
        mock_response.data = [self.mocker.Mock(url="https://provider.com/image.png", b64_json=None)]

        self.mocker.patch("llm.service.litellm.image_generation", return_value=mock_response)
        self.mocker.patch.object(self.service, "_download_and_save_image", return_value=None)
        result = self.service.image_generation("a cat")

        assert result.content == "https://provider.com/image.png"

    def test_image_generation_with_base64_response(self, tmp_path: object) -> None:
        """GIVEN provider returns base64 WHEN generating THEN saves and returns URL."""
        import base64

        self.mock_plugin.registryValue = self.mocker.Mock(
            side_effect=lambda key, channel=None: {
                "imageApiKey": "test-api-key",
                "imageModel": "gemini/imagen",
                "timeout": 30,
                "maxPromptLength": 10000,
                "httpRoot": str(tmp_path),
                "httpUrlBase": "https://example.com/llm",
                "fileCleanupAge": 24,
                "fileCleanupMax": 1000,
            }.get(key)
        )

        image_data = b"\x89PNG\r\n\x1a\nfake image"
        b64_data = base64.b64encode(image_data).decode()

        mock_response = self.mocker.Mock()
        mock_response.data = [self.mocker.Mock(url=None, b64_json=b64_data)]

        self.mocker.patch("llm.service.litellm.image_generation", return_value=mock_response)
        result = self.service.image_generation("a cat")

        assert result.content.startswith("https://example.com/llm/img_")
        assert result.content.endswith(".png")

    def test_image_generation_sends_typing_indicator(self) -> None:
        """GIVEN irc context WHEN generating THEN sends typing indicators."""
        irc = self._make_mock_irc()
        msg = self._make_mock_msg()

        mock_response = self.mocker.Mock()
        mock_response.data = [self.mocker.Mock(url="https://example.com/image.png", b64_json=None)]

        self.mocker.patch("llm.service.litellm.image_generation", return_value=mock_response)
        self.mocker.patch.object(self.service, "_download_and_save_image", return_value=None)
        self.service.image_generation("a cat", irc=irc, msg=msg)

        # Should have called queueMsg twice - once for active, once for done
        assert irc.queueMsg.call_count == 2

        # First call should be typing=active
        first_msg = irc.queueMsg.call_args_list[0][0][0]
        assert first_msg.server_tags == {"+typing": "active"}

        # Second call should be typing=done
        second_msg = irc.queueMsg.call_args_list[1][0][0]
        assert second_msg.server_tags == {"+typing": "done"}

    def test_image_generation_sends_done_on_error(self) -> None:
        """GIVEN error during generation WHEN generating THEN still sends done indicator."""
        irc = self._make_mock_irc()
        msg = self._make_mock_msg()

        self.mocker.patch(
            "llm.service.litellm.image_generation", side_effect=Exception("API error")
        )
        result = self.service.image_generation("a cat", irc=irc, msg=msg)

        assert "Error" in result.content

        # Should still send typing=done in finally block
        assert irc.queueMsg.call_count == 2
        second_msg = irc.queueMsg.call_args_list[1][0][0]
        assert second_msg.server_tags == {"+typing": "done"}

    def test_image_generation_no_data_in_response(self) -> None:
        """GIVEN empty response WHEN generating THEN returns content filter error."""
        mock_response = self.mocker.Mock()
        mock_response.data = []

        self.mocker.patch("llm.service.litellm.image_generation", return_value=mock_response)
        result = self.service.image_generation("a cat")

        assert "No image generated" in result.content
        assert "content safety filters" in result.content

    def test_image_generation_without_irc_context(self) -> None:
        """GIVEN no irc context WHEN generating THEN works without typing indicators."""
        mock_response = self.mocker.Mock()
        mock_response.data = [self.mocker.Mock(url="https://example.com/image.png", b64_json=None)]

        self.mocker.patch("llm.service.litellm.image_generation", return_value=mock_response)
        self.mocker.patch.object(
            self.service,
            "_download_and_save_image",
            return_value="https://example.com/llm/img_local.png",
        )
        result = self.service.image_generation("a cat")

        assert result.content == "https://example.com/llm/img_local.png"


class TestCleanupWithImages:
    """Tests for _cleanup_old_files with image extensions."""

    @pytest.fixture(autouse=True)
    def setup(self, make_service) -> None:
        """Set up test fixtures."""
        self.service, self.mock_plugin = make_service(
            fileCleanupAge=24,
            fileCleanupMax=1000,
        )

    def test_cleanup_collects_image_files(self, tmp_path: object) -> None:
        """GIVEN image files exist WHEN cleanup runs THEN collects them."""
        from pathlib import Path

        # Create test files of various types
        (Path(str(tmp_path)) / "code_abc.html").write_text("code")
        (Path(str(tmp_path)) / "img_def.png").write_bytes(b"png")
        (Path(str(tmp_path)) / "img_ghi.jpg").write_bytes(b"jpg")
        (Path(str(tmp_path)) / "img_jkl.jpeg").write_bytes(b"jpeg")
        (Path(str(tmp_path)) / "img_mno.webp").write_bytes(b"webp")
        (Path(str(tmp_path)) / "other.txt").write_text("ignored")

        # Set max_files to 0 to force cleanup of all
        self.service._cleanup_old_files(str(tmp_path), max_age_hours=0, max_files=0)

        # All recognized files should be deleted, txt should remain
        assert not (Path(str(tmp_path)) / "code_abc.html").exists()
        assert not (Path(str(tmp_path)) / "img_def.png").exists()
        assert not (Path(str(tmp_path)) / "img_ghi.jpg").exists()
        assert not (Path(str(tmp_path)) / "img_jkl.jpeg").exists()
        assert not (Path(str(tmp_path)) / "img_mno.webp").exists()
        assert (Path(str(tmp_path)) / "other.txt").exists()


class TestCleanupLock:
    """Test that _cleanup_old_files uses a lock for thread safety (Fix 5)."""

    @pytest.fixture(autouse=True)
    def setup(self, make_service) -> None:
        """Set up test fixtures."""
        self.service, self.mock_plugin = make_service(
            fileCleanupAge=24,
            fileCleanupMax=1000,
        )

    def test_cleanup_lock_exists(self) -> None:
        """GIVEN service WHEN initialized THEN _cleanup_lock exists."""
        assert hasattr(self.service, "_cleanup_lock")

    def test_cleanup_serializes_concurrent_calls(self, tmp_path: object) -> None:
        """GIVEN concurrent cleanup calls WHEN running THEN lock prevents races."""
        from pathlib import Path

        # Create a test file
        (Path(str(tmp_path)) / "img_test.png").write_bytes(b"png")

        errors: list[Exception] = []

        def run_cleanup() -> None:
            try:
                self.service._cleanup_old_files(str(tmp_path), max_age_hours=0, max_files=0)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=run_cleanup) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestHTTPFileManagement:
    """Tests for HTTP file storage, URL generation, and cleanup."""

    @pytest.fixture(autouse=True)
    def setup(self, make_service, mocker: MockerFixture) -> None:
        """Set up test fixtures."""
        self.mocker = mocker
        self.service, self.mock_plugin = make_service()

    def test_get_http_paths_localhost_fallback(self) -> None:
        """GIVEN no httpRoot/httpUrlBase and no publicUrl WHEN get_http_paths called THEN falls back to localhost with port."""
        self.mock_plugin.registryValue = self.mocker.Mock(
            side_effect=lambda key, *a, **kw: {"httpRoot": "", "httpUrlBase": ""}.get(key, "")
        )
        mock_conf = self.mocker.patch("llm.service.conf")
        mock_conf.supybot.directories.data.web.dirize.return_value = "/tmp/web"
        mock_conf.supybot.servers.http.publicUrl.return_value = ""
        mock_conf.supybot.servers.http.port.return_value = 8080

        http_root, url_base = self.service.get_http_paths()

        assert http_root == "/tmp/web"
        assert "localhost:8080" in url_base

    def test_save_code_to_http_oserror_returns_none(self) -> None:
        """GIVEN mkdir raises OSError WHEN save_code_to_http called THEN returns None."""
        self.mocker.patch.object(
            self.service,
            "get_http_paths",
            return_value=("/nonexistent/path", "http://x"),
        )
        self.mocker.patch("llm.service.Path.mkdir", side_effect=OSError("disk full"))

        result = self.service.save_code_to_http("# hello world")

        assert result is None

    def test_cleanup_old_files_deletes_old_preserves_new(self, tmp_path: object) -> None:
        """GIVEN old and new files WHEN _cleanup_old_files called THEN deletes old, keeps new."""
        import os
        import time
        from pathlib import Path

        dir_path = Path(str(tmp_path))
        old_file = dir_path / "old_code.html"
        new_file = dir_path / "new_code.html"
        old_file.write_text("old")
        new_file.write_text("new")

        # Backdate old file by 25 hours
        old_mtime = time.time() - (25 * 3600)
        os.utime(str(old_file), (old_mtime, old_mtime))

        self.service._cleanup_old_files(str(dir_path), max_age_hours=24, max_files=100)

        assert not old_file.exists()
        assert new_file.exists()

    def test_cleanup_old_files_caps_recent_files(self, tmp_path: object) -> None:
        """GIVEN 5 recent files WHEN max_files=2 THEN only 2 newest remain."""
        import time
        from pathlib import Path

        dir_path = Path(str(tmp_path))
        files = []
        for i in range(5):
            f = dir_path / f"code_{i}.html"
            f.write_text(f"content {i}")
            # Stagger mtimes so ordering is deterministic
            import os

            mtime = time.time() - (10 * (4 - i))  # oldest first
            os.utime(str(f), (mtime, mtime))
            files.append(f)

        self.service._cleanup_old_files(str(dir_path), max_age_hours=9999, max_files=2)

        remaining = list(dir_path.glob("*.html"))
        assert len(remaining) == 2

    def test_cleanup_old_files_nonexistent_dir_no_error(self) -> None:
        """GIVEN nonexistent directory WHEN _cleanup_old_files called THEN no error raised."""
        self.service._cleanup_old_files("/nonexistent/path", max_age_hours=24, max_files=100)


class TestDrawContext:
    """Tests for context integration in image generation."""

    @pytest.fixture(autouse=True)
    def setup(self, make_service, mocker: MockerFixture) -> None:
        """Set up test fixtures."""
        self.mocker = mocker
        self.service, self.mock_plugin = make_service(
            imageApiKey="test-api-key",
            imageModel="gemini/imagen",
            timeout=30,
            maxPromptLength=10000,
            drawAutoRewriteMax=0,
        )

    def test_image_generation_uses_raw_prompt(self) -> None:
        """GIVEN a prompt WHEN generating image THEN uses prompt as-is."""
        prompt_used = []

        def capture_prompt(**kwargs):
            prompt_used.append(kwargs.get("prompt", ""))
            mock_response = self.mocker.Mock()
            mock_response.data = [
                self.mocker.Mock(url="https://example.com/img.png", b64_json=None)
            ]
            return mock_response

        self.mocker.patch("llm.service.litellm.image_generation", side_effect=capture_prompt)
        self.mocker.patch.object(self.service, "_download_and_save_image", return_value=None)
        self.service.image_generation("a sunset")

        assert prompt_used[0] == "a sunset"


class TestImageUrlSsrfProtection:
    """Tests for SSRF protection in image URL validation."""

    @pytest.fixture(autouse=True)
    def setup(self, make_service, mocker: MockerFixture) -> None:
        """Set up test fixtures."""
        self.mocker = mocker
        self.service, self.mock_plugin = make_service()

    def test_blocks_localhost(self) -> None:
        """GIVEN localhost URL WHEN validated THEN rejected."""
        assert self.service.validate_image_url("http://localhost/image.png") is False
        assert self.service.validate_image_url("http://127.0.0.1/image.png") is False

    def test_blocks_private_ranges(self) -> None:
        """GIVEN private IP range URLs WHEN validated THEN rejected."""
        assert self.service.validate_image_url("http://192.168.1.1/image.png") is False
        assert self.service.validate_image_url("http://10.0.0.1/image.png") is False
        assert self.service.validate_image_url("http://172.16.0.1/image.png") is False

    def test_blocks_metadata_endpoints(self) -> None:
        """GIVEN cloud metadata endpoint WHEN validated THEN rejected."""
        assert self.service.validate_image_url("http://169.254.169.254/image.png") is False

    def test_allows_public_urls(self) -> None:
        """GIVEN public URL WHEN validated THEN accepted."""
        # Note: This test requires DNS resolution, so we mock the resolver
        self.mocker.patch.object(self.service, "_resolves_to_public", return_value=True)
        assert self.service.validate_image_url("https://example.com/image.png") is True

    def test_resolver_fails_closed(self) -> None:
        """GIVEN DNS resolution failure WHEN checking host THEN blocked."""
        assert (
            self.service._resolves_to_public(
                "http://definitely-not-a-valid-hostname-12345.invalid/x.png"
            )
            is False
        )


class TestDrawAutoRewrite:
    """Tests for automatic prompt rewriting on content safety failures."""

    @pytest.fixture(autouse=True)
    def setup(self, make_service, mocker: MockerFixture) -> None:
        """Set up test fixtures."""
        self.mocker = mocker
        self.service, self.mock_plugin = make_service(
            imageApiKey="test-draw-key",
            imageModel="vertex_ai/imagen-4.0-generate-001",
            assistantApiKey="test-ask-key",
            assistantModel="gemini/gemini-flash-latest",
            timeout=30,
            maxPromptLength=10000,
            httpRoot="/tmp/test",
            httpUrlBase="https://example.com/llm",
            drawAutoRewriteMax=3,
        )
        self.config_values = {
            "imageApiKey": "test-draw-key",
            "imageModel": "vertex_ai/imagen-4.0-generate-001",
            "assistantApiKey": "test-ask-key",
            "assistantModel": "gemini/gemini-flash-latest",
            "timeout": 30,
            "maxPromptLength": 10000,
            "httpRoot": "/tmp/test",
            "httpUrlBase": "https://example.com/llm",
            "drawAutoRewriteMax": 3,
        }

    def _make_success_response(self, url: str = "https://example.com/img.png") -> Mock:
        """Create a mock successful image generation response."""
        response = self.mocker.Mock()
        response.data = [self.mocker.Mock(url=url, b64_json=None)]
        response.usage = self.mocker.Mock(prompt_tokens=5, completion_tokens=0)
        return response

    def _make_empty_response(self) -> Mock:
        """Create a mock empty (content-blocked) image generation response."""
        response = self.mocker.Mock()
        response.data = []
        response.usage = self.mocker.Mock(prompt_tokens=5, completion_tokens=0)
        return response

    def _make_rewrite_response(self, rewritten: str = "a safe cat") -> Mock:
        """Create a mock completion response for prompt rewriting."""
        return make_completion_response(rewritten, prompt_tokens=20, completion_tokens=10)

    def test_auto_rewrite_on_empty_data_succeeds(self) -> None:
        """GIVEN empty response data WHEN auto-rewrite enabled THEN retries with rewritten prompt."""
        empty_resp = self._make_empty_response()
        success_resp = self._make_success_response()
        rewrite_resp = self._make_rewrite_response("a friendly cat")

        self.mocker.patch(
            "llm.service.litellm.image_generation", side_effect=[empty_resp, success_resp]
        )
        self.mocker.patch("llm.service.litellm.completion", return_value=rewrite_resp)
        self.mocker.patch("llm.service.litellm.completion_cost", return_value=0.01)
        self.mocker.patch.object(
            self.service,
            "_download_and_save_image",
            return_value="https://example.com/llm/img_local.png",
        )
        result = self.service.image_generation("a dangerous cat")

        assert result.content == "https://example.com/llm/img_local.png"
        assert result.rewritten_prompt == "a friendly cat"

    def test_auto_rewrite_on_content_policy_error_succeeds(self) -> None:
        """GIVEN ContentPolicyViolationError WHEN auto-rewrite enabled THEN retries."""
        import litellm as litellm_module

        rewrite_resp = self._make_rewrite_response("a safe prompt")
        success_resp = self._make_success_response()

        self.mocker.patch(
            "llm.service.litellm.image_generation",
            side_effect=[
                litellm_module.ContentPolicyViolationError(
                    message="blocked", model="imagen", llm_provider="vertex_ai"
                ),
                success_resp,
            ],
        )
        self.mocker.patch("llm.service.litellm.completion", return_value=rewrite_resp)
        self.mocker.patch("llm.service.litellm.completion_cost", return_value=0.01)
        self.mocker.patch.object(
            self.service,
            "_download_and_save_image",
            return_value="https://example.com/llm/img_local.png",
        )
        result = self.service.image_generation("bad prompt")

        assert result.content == "https://example.com/llm/img_local.png"
        assert result.rewritten_prompt == "a safe prompt"

    def test_auto_rewrite_multiple_retries_succeeds_on_third(self) -> None:
        """GIVEN multiple blocks WHEN retrying THEN succeeds on later attempt."""
        empty_resp = self._make_empty_response()
        success_resp = self._make_success_response()
        rewrite1 = self._make_rewrite_response("rewrite v1")
        rewrite2 = self._make_rewrite_response("rewrite v2")

        self.mocker.patch(
            "llm.service.litellm.image_generation",
            side_effect=[empty_resp, empty_resp, success_resp],
        )
        self.mocker.patch(
            "llm.service.litellm.completion",
            side_effect=[rewrite1, rewrite2],
        )
        self.mocker.patch("llm.service.litellm.completion_cost", return_value=0.001)
        self.mocker.patch.object(
            self.service,
            "_download_and_save_image",
            return_value="https://example.com/llm/img_local.png",
        )
        result = self.service.image_generation("test prompt")

        assert result.content == "https://example.com/llm/img_local.png"
        assert result.rewritten_prompt == "rewrite v2"

    def test_auto_rewrite_exhausts_all_retries(self) -> None:
        """GIVEN all retries fail WHEN max reached THEN returns error with attempt count."""
        empty_resp = self._make_empty_response()
        rewrite1 = self._make_rewrite_response("rewrite v1")
        rewrite2 = self._make_rewrite_response("rewrite v2")
        rewrite3 = self._make_rewrite_response("rewrite v3")

        self.mocker.patch(
            "llm.service.litellm.image_generation",
            side_effect=[empty_resp, empty_resp, empty_resp, empty_resp],
        )
        self.mocker.patch(
            "llm.service.litellm.completion",
            side_effect=[rewrite1, rewrite2, rewrite3],
        )
        self.mocker.patch("llm.service.litellm.completion_cost", return_value=0.001)
        result = self.service.image_generation("test prompt")

        assert "Error" in result.content
        assert "3 rewrite attempt" in result.content

    def test_auto_rewrite_disabled_when_max_zero(self) -> None:
        """GIVEN drawAutoRewriteMax=0 WHEN content blocked THEN no rewrite attempted."""
        self.config_values["drawAutoRewriteMax"] = 0
        self.mock_plugin.registryValue = self.mocker.Mock(
            side_effect=lambda key, channel=None: self.config_values.get(key)
        )
        empty_resp = self._make_empty_response()

        self.mocker.patch("llm.service.litellm.image_generation", return_value=empty_resp)
        mock_completion = self.mocker.patch("llm.service.litellm.completion")
        self.mocker.patch("llm.service.litellm.completion_cost", return_value=0.0)
        result = self.service.image_generation("test prompt")

        assert "content safety filters" in result.content
        mock_completion.assert_not_called()

    def test_auto_rewrite_llm_failure_falls_back(self) -> None:
        """GIVEN rewrite LLM fails WHEN retrying THEN falls back to error message."""
        empty_resp = self._make_empty_response()

        self.mocker.patch("llm.service.litellm.image_generation", return_value=empty_resp)
        self.mocker.patch(
            "llm.service.litellm.completion",
            side_effect=Exception("LLM unavailable"),
        )
        self.mocker.patch("llm.service.litellm.completion_cost", return_value=0.0)
        result = self.service.image_generation("test prompt")

        assert "Error" in result.content

    def test_auto_rewrite_skipped_when_ask_key_missing(self) -> None:
        """GIVEN assistantApiKey not configured WHEN content blocked THEN skips rewrite."""
        self.config_values["assistantApiKey"] = ""
        self.mock_plugin.registryValue = self.mocker.Mock(
            side_effect=lambda key, channel=None: self.config_values.get(key)
        )
        empty_resp = self._make_empty_response()

        self.mocker.patch("llm.service.litellm.image_generation", return_value=empty_resp)
        mock_completion = self.mocker.patch("llm.service.litellm.completion")
        self.mocker.patch("llm.service.litellm.completion_cost", return_value=0.0)
        result = self.service.image_generation("test prompt")

        assert "Error" in result.content
        mock_completion.assert_not_called()

    def test_auto_rewrite_aggregates_costs(self) -> None:
        """GIVEN successful rewrite WHEN costs tracked THEN aggregated in result."""
        empty_resp = self._make_empty_response()
        success_resp = self._make_success_response()
        rewrite_resp = self._make_rewrite_response("safe prompt")

        self.mocker.patch(
            "llm.service.litellm.image_generation", side_effect=[empty_resp, success_resp]
        )
        self.mocker.patch("llm.service.litellm.completion", return_value=rewrite_resp)
        self.mocker.patch("llm.service.litellm.completion_cost", return_value=0.005)
        self.mocker.patch.object(self.service, "_download_and_save_image", return_value=None)
        result = self.service.image_generation("test prompt")

        # Should include both rewrite and generation costs
        assert result.prompt_tokens > 0
        assert result.cost > 0

    def test_non_content_error_does_not_trigger_rewrite(self) -> None:
        """GIVEN timeout error WHEN generating THEN no rewrite attempted."""
        import litellm as litellm_module

        self.mocker.patch(
            "llm.service.litellm.image_generation",
            side_effect=litellm_module.Timeout(
                message="timed out", model="imagen", llm_provider="vertex_ai"
            ),
        )
        mock_completion = self.mocker.patch("llm.service.litellm.completion")
        result = self.service.image_generation("test prompt")

        assert "timed out" in result.content.lower()
        mock_completion.assert_not_called()

    def test_auth_error_does_not_trigger_rewrite(self) -> None:
        """GIVEN authentication error WHEN generating THEN no rewrite attempted."""
        import litellm as litellm_module

        self.mocker.patch(
            "llm.service.litellm.image_generation",
            side_effect=litellm_module.AuthenticationError(
                message="invalid key", model="imagen", llm_provider="vertex_ai"
            ),
        )
        mock_completion = self.mocker.patch("llm.service.litellm.completion")
        result = self.service.image_generation("test prompt")

        assert "Invalid API key" in result.content
        mock_completion.assert_not_called()

    def test_prior_rewrites_passed_to_subsequent_attempts(self) -> None:
        """GIVEN multiple rewrite attempts WHEN calling rewriter THEN prior history passed."""
        self.config_values["drawAutoRewriteMax"] = 2
        self.mock_plugin.registryValue = self.mocker.Mock(
            side_effect=lambda key, channel=None: self.config_values.get(key)
        )
        empty_resp = self._make_empty_response()
        rewrite1 = self._make_rewrite_response("rewrite v1")
        rewrite2 = self._make_rewrite_response("rewrite v2")

        self.mocker.patch(
            "llm.service.litellm.image_generation",
            side_effect=[empty_resp, empty_resp, empty_resp],
        )
        mock_completion = self.mocker.patch(
            "llm.service.litellm.completion",
            side_effect=[rewrite1, rewrite2],
        )
        self.mocker.patch("llm.service.litellm.completion_cost", return_value=0.001)
        self.service.image_generation("test prompt")

        # Second rewrite call should include prior_rewrites in the user message
        assert mock_completion.call_count == 2
        second_call_messages = mock_completion.call_args_list[1][1]["messages"]
        user_msg = second_call_messages[1]["content"]
        assert "rewrite v1" in user_msg
        assert "Previous rewrite attempts" in user_msg

    def test_rewritten_prompt_not_set_on_first_success(self) -> None:
        """GIVEN first attempt succeeds WHEN no rewrite needed THEN rewritten_prompt is None."""
        success_resp = self._make_success_response()

        self.mocker.patch("llm.service.litellm.image_generation", return_value=success_resp)
        self.mocker.patch("llm.service.litellm.completion_cost", return_value=0.01)
        self.mocker.patch.object(self.service, "_download_and_save_image", return_value=None)
        result = self.service.image_generation("a cat")

        assert result.rewritten_prompt is None

    def test_auto_rewrite_on_bad_request_moderation_blocked(self) -> None:
        """GIVEN BadRequestError with moderation_blocked WHEN auto-rewrite enabled THEN retries."""
        import litellm as litellm_module

        rewrite_resp = self._make_rewrite_response("a safe prompt")
        success_resp = self._make_success_response()

        self.mocker.patch(
            "llm.service.litellm.image_generation",
            side_effect=[
                litellm_module.BadRequestError(
                    message=(
                        "OpenAIException - Error code: 400 - {'error': {'code': "
                        "'moderation_blocked'}}"
                    ),
                    model="imagen",
                    llm_provider="vertex_ai",
                ),
                success_resp,
            ],
        )
        self.mocker.patch("llm.service.litellm.completion", return_value=rewrite_resp)
        self.mocker.patch("llm.service.litellm.completion_cost", return_value=0.01)
        self.mocker.patch.object(
            self.service,
            "_download_and_save_image",
            return_value="https://example.com/llm/img_local.png",
        )
        result = self.service.image_generation("bad prompt")

        assert result.content == "https://example.com/llm/img_local.png"
        assert result.rewritten_prompt == "a safe prompt"

    def test_non_moderation_bad_request_does_not_trigger_rewrite(self) -> None:
        """GIVEN BadRequestError without moderation keywords WHEN generating THEN no rewrite."""
        import litellm as litellm_module

        self.mocker.patch(
            "llm.service.litellm.image_generation",
            side_effect=litellm_module.BadRequestError(
                message="Invalid image size parameter",
                model="imagen",
                llm_provider="vertex_ai",
            ),
        )
        mock_completion = self.mocker.patch("llm.service.litellm.completion")
        result = self.service.image_generation("test prompt")

        assert "Error" in result.content
        mock_completion.assert_not_called()


class TestImageGenerationValidation:
    """Tests for image_generation() early validation error paths."""

    @pytest.fixture(autouse=True)
    def setup(self, make_service, mocker: MockerFixture) -> None:
        """Set up test fixtures."""
        self.mocker = mocker
        self.make_service = make_service
        self.service, self.mock_plugin = make_service()

    def test_image_generation_invalid_prompt(self) -> None:
        """GIVEN empty prompt WHEN image_generation called THEN returns error in result."""
        result = self.service.image_generation("")

        assert result.error is not None
        assert "Error" in result.content

    def test_image_generation_missing_draw_key(self) -> None:
        """GIVEN service with empty imageApiKey WHEN image_generation called THEN returns API key error."""
        service, _ = self.make_service(imageApiKey="")

        result = service.image_generation("A beautiful sunset")

        assert result.error is not None
        assert "API key" in result.content


class TestImageGenerationPaths:
    """Tests for image generation rewrite loop and error paths."""

    @pytest.fixture(autouse=True)
    def setup(self, make_service, mocker: MockerFixture) -> None:
        """Set up test fixtures."""
        self.mocker = mocker
        self.service, self.mock_plugin = make_service(
            imageApiKey="test-draw-key",
            imageModel="dall-e-3",
            assistantApiKey="test-ask-key",
            assistantModel="gemini/gemini-flash-latest",
            timeout=30,
            drawTimeout=30,
            maxPromptLength=10000,
            httpRoot="/tmp/test",
            httpUrlBase="https://example.com/llm",
            drawAutoRewriteMax=3,
        )

    def test_rewrite_empty_response(self) -> None:
        """GIVEN LLM returns empty content WHEN _rewrite_prompt_for_safety called THEN returns None tuple."""
        response = self.mocker.Mock()
        response.choices = [self.mocker.Mock(message=self.mocker.Mock(content=""))]

        self.mocker.patch("llm.service.litellm.completion", return_value=response)

        result = self.service._rewrite_prompt_for_safety("bad prompt", "blocked", [], "#chan")

        assert result == (None, 0, 0, 0.0)

    def test_xai_model_kwargs(self) -> None:
        """GIVEN xai model WHEN _attempt_image_generation called THEN passes extra kwargs."""
        mock_response = self.mocker.Mock()
        mock_response.data = [self.mocker.Mock(url="http://img.png", b64_json=None)]

        mock_img_gen = self.mocker.patch(
            "llm.service.litellm.image_generation", return_value=mock_response
        )
        self.mocker.patch.object(self.service, "_extract_usage", return_value=(0, 0, 0.0))
        self.mocker.patch.object(self.service, "_download_and_save_image", return_value=None)

        self.service._attempt_image_generation("cat", "xai/grok-2-image", 30)

        call_kwargs = mock_img_gen.call_args
        assert call_kwargs[1]["aspect_ratio"] == "9:16"

    def test_b64_json_save_failure(self) -> None:
        """GIVEN b64_json data but save fails WHEN _attempt_image_generation called THEN returns error."""
        image_data = self.mocker.Mock()
        image_data.url = None
        image_data.b64_json = "base64data"

        mock_response = self.mocker.Mock()
        mock_response.data = [image_data]

        self.mocker.patch("llm.service.litellm.image_generation", return_value=mock_response)
        self.mocker.patch.object(self.service, "_extract_usage", return_value=(0, 0, 0.0))
        self.mocker.patch.object(self.service, "save_image_to_http", return_value=None)

        result = self.service._attempt_image_generation("cat", "dall-e-3", 30)

        assert result is not None
        assert result.error is not None

    def test_timeout_not_stashed(self) -> None:
        """GIVEN image_generation times out and stashing fails WHEN called THEN returns error."""
        import litellm as litellm_module

        self.mocker.patch(
            "llm.service.litellm.image_generation",
            side_effect=litellm_module.Timeout(
                message="Request timed out", model="dall-e-3", llm_provider="openai"
            ),
        )
        self.mocker.patch.object(self.service, "_stash_timeout", return_value=False)

        result = self.service.image_generation("a cat")

        assert result.error is not None

    def test_non_content_error_in_rewrite_loop(self) -> None:
        """GIVEN first attempt blocked and retry raises non-content error WHEN generating THEN returns error."""
        self.mocker.patch.object(
            self.service,
            "_attempt_image_generation",
            side_effect=[None, RuntimeError("network")],
        )
        self.mocker.patch.object(
            self.service,
            "_rewrite_prompt_for_safety",
            return_value=("rewritten", 10, 5, 0.01),
        )
        self.mocker.patch.object(
            self.service,
            "_is_content_safety_error",
            return_value=False,
        )

        result = self.service.image_generation("a cat")

        assert result.error is not None

    def test_outer_exception_handler(self) -> None:
        """GIVEN unexpected error in validate_prompt WHEN image_generation called THEN returns graceful error."""
        self.mocker.patch.object(
            self.service,
            "validate_prompt",
            side_effect=RuntimeError("unexpected"),
        )

        result = self.service.image_generation("a cat")

        assert result.error is not None


class TestRetryImage:
    """Tests for _retry_image error paths."""

    @pytest.fixture(autouse=True)
    def setup(self, make_service, mocker: MockerFixture) -> None:
        """Set up service with mock plugin."""
        self.mocker = mocker
        self.service, self.mock_plugin = make_service()

    def _make_task(self, **overrides):
        """Create a PendingTaskRow with draw defaults."""
        from llm.persistence import PendingTaskRow

        defaults = {
            "id": 1,
            "task_type": "draw",
            "nick": "user",
            "reply_target": "#chan",
            "is_channel": 1,
            "prompt_preview": "test",
            "model": "dall-e-3",
            "request_data": '{"prompt": "cat"}',
            "submitted_at": 100.0,
            "expires_at": 200.0,
            "attempt_count": 0,
            "next_attempt_at": 100.0,
            "claimed_until": 0,
            "last_error": "",
            "delivery_state": "pending",
            "result_payload": "",
            "last_delivery_error": "",
            "delivery_attempt_count": 0,
            "origin_request_id": "",
            "account": None,
        }
        defaults.update(overrides)
        return PendingTaskRow(**defaults)

    def test_retry_image_malformed_data(self) -> None:
        """GIVEN request_data missing prompt key WHEN _retry_image called THEN returns failed_terminal with Malformed reason."""
        task = self._make_task()

        result = self.service._retry_image(task, {"not_prompt": "x"})

        assert result.status == "failed_terminal"
        assert "Malformed" in result.reason

    def test_retry_image_no_api_key(self) -> None:
        """GIVEN imageApiKey is empty WHEN _retry_image called THEN returns failed_terminal with API key reason."""
        from .conftest import make_registry_side_effect

        self.mock_plugin.registryValue = self.mocker.Mock(
            side_effect=make_registry_side_effect({"imageApiKey": ""})
        )
        service = LLMService(self.mock_plugin)
        task = self._make_task()

        result = service._retry_image(task, {"prompt": "cat"})

        assert result.status == "failed_terminal"
        assert "API key" in result.reason

    def test_retry_image_content_blocked(self) -> None:
        """GIVEN _attempt_image_generation returns None WHEN _retry_image called THEN returns failed_terminal with blocked reason."""
        task = self._make_task()
        self.mocker.patch.object(self.service, "_attempt_image_generation", return_value=None)

        result = self.service._retry_image(task, {"prompt": "cat"})

        assert result.status == "failed_terminal"
        assert "blocked" in result.reason
