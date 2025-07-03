"""Convert images to Minecraft tellraw commands with colored text."""

import gzip
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import cv2 as cv
import numpy as np
import numpy.typing as npt
import requests
from PIL import Image

ImageArray = npt.NDArray[np.uint8]


class ImageLoadError(Exception):
    """Raised when image loading fails."""


class ImageProcessError(Exception):
    """Raised when image processing fails."""


@dataclass(frozen=True, slots=True)
class MinecraftImageConfig:
    """Configuration for Minecraft image conversion."""

    width: int = 300
    height: int | None = None
    aspect_ratio_scale: float = 0.15  # Compensate for text character height
    minecraft_height_limit: int = 31
    character: str = "▏"
    url_size_limit: int = 20_000_000  # 20MB
    auto_crop: bool = True  # Enable auto-cropping of uniform borders
    crop_tolerance: int = 5  # Color difference tolerance for border detection


class MinecraftImage:
    """Handles image conversion to Minecraft text format."""

    config: MinecraftImageConfig
    _image: ImageArray

    def __init__(
        self,
        image: ImageArray,
        config: MinecraftImageConfig | None = None,
    ) -> None:
        self.config = config or MinecraftImageConfig()
        self._image = image
        if self.config.auto_crop:
            self._auto_crop_borders()
        self._resize()

    @classmethod
    def from_url(cls, url: str, **kwargs: Any) -> Self:
        """Load image from URL."""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            # Quick header checks
            if (size := response.headers.get("Content-Length")) and int(
                size,
            ) > 20_000_000:
                msg = f"Image too large: {int(size):,} bytes"
                raise ImageLoadError(msg)

            if (ct := response.headers.get("Content-Type", "")) and "image" not in ct:
                msg = f"Invalid content type: {ct}"
                raise ImageLoadError(msg)

            return cls.from_bytes(response.content, **kwargs)
        except requests.RequestException as e:
            msg = f"Failed to download: {e}"
            raise ImageLoadError(msg) from e

    @classmethod
    def from_file(cls, path: Path | str, **kwargs: Any) -> Self:
        """Load image from file."""
        return cls.from_bytes(Path(path).read_bytes(), **kwargs)

    @classmethod
    def from_bytes(cls, data: bytes, **kwargs: Any) -> Self:
        """Load image from bytes, with special handling for GIFs."""
        # Try to load with PIL first to handle GIFs and other formats
        try:
            pil_image = Image.open(io.BytesIO(data))

            # If it's a GIF or has multiple frames, get the first frame
            if hasattr(pil_image, "is_animated") and pil_image.is_animated:
                pil_image.seek(0)  # Ensure we're at the first frame

            # Convert to RGB if necessary (handles RGBA, P mode, etc.)
            if pil_image.mode != "RGB":
                # Handle transparency by compositing on white background
                if pil_image.mode in ("RGBA", "LA") or (
                    pil_image.mode == "P" and "transparency" in pil_image.info
                ):
                    background = Image.new("RGB", pil_image.size, (255, 255, 255))
                    background.paste(
                        pil_image,
                        mask=(
                            pil_image.split()[-1]
                            if pil_image.mode in ("RGBA", "LA")
                            else pil_image
                        ),
                    )
                    pil_image = background
                else:
                    pil_image = pil_image.convert("RGB")

            # Convert PIL image to OpenCV format (RGB to BGR)
            image = cv.cvtColor(np.array(pil_image), cv.COLOR_RGB2BGR)

        except Exception:
            # Fallback to OpenCV for non-GIF formats
            image = cv.imdecode(np.frombuffer(data, dtype=np.uint8), cv.IMREAD_COLOR)
            if image is None:
                msg = "Failed to decode image"
                raise ImageLoadError(msg)

        config = MinecraftImageConfig(**kwargs) if kwargs else None
        return cls(image, config)

    def _auto_crop_borders(self) -> None:
        """Automatically crop uniform color borders from the image."""
        if self._image.size == 0:
            return

        h, w = self._image.shape[:2]

        # Find the dominant border color (most common color in the corners)
        corners = [
            self._image[0, 0],
            self._image[0, w - 1],
            self._image[h - 1, 0],
            self._image[h - 1, w - 1],
        ]
        border_color = np.median(corners, axis=0).astype(np.uint8)

        # Create a mask of pixels that match the border color (within tolerance)
        diff = np.abs(self._image.astype(np.int16) - border_color.astype(np.int16))
        mask = np.all(diff <= self.config.crop_tolerance, axis=2)

        # Find crop boundaries by checking where non-border pixels start
        # Top border
        top = 0
        for i in range(h):
            if not np.all(mask[i, :]):
                top = i
                break

        # Bottom border
        bottom = h
        for i in range(h - 1, -1, -1):
            if not np.all(mask[i, :]):
                bottom = i + 1
                break

        # Left border
        left = 0
        for i in range(w):
            if not np.all(mask[:, i]):
                left = i
                break

        # Right border
        right = w
        for i in range(w - 1, -1, -1):
            if not np.all(mask[:, i]):
                right = i + 1
                break

        # Apply crop if we found borders to remove
        if top > 0 or bottom < h or left > 0 or right < w:
            # Ensure we don't crop everything
            if bottom > top and right > left:
                self._image = self._image[top:bottom, left:right]

    def _resize(self) -> None:
        """Resize image according to configuration."""
        h, w = self._image.shape[:2]
        new_w = min(w, self.config.width)

        if self.config.height is not None:
            new_h = self.config.height
        else:
            new_h = int(new_w / w * h * self.config.aspect_ratio_scale)
            new_h = min(max(new_h, 1), self.config.minecraft_height_limit)

        if new_w != w or new_h != h:
            self._image = cv.resize(
                self._image,
                (new_w, new_h),
                interpolation=cv.INTER_LANCZOS4,
            )

    def get_hex_data(self) -> str:
        """Get compacted hex color data."""
        # Convert BGR to RGB and directly to hex strings
        rgb = cv.cvtColor(self._image, cv.COLOR_BGR2RGB)
        return "".join(f"{r:02X}{g:02X}{b:02X}" for r, g, b in rgb.reshape(-1, 3))

    def _generate_content(self) -> list[dict[str, str]]:
        """Generate content array with proper line breaks and color run optimization."""
        hex_data = self.get_hex_data()
        _h, w = self._image.shape[:2]

        content = [{"text": "\n"}]  # First line has unequal spacing. Skip it.

        current_color = None
        current_text = ""

        def flush_run() -> None:
            nonlocal current_text
            if current_text:
                content.append({"text": current_text, "color": f"#{current_color}"})
                current_text = ""

        for i in range(0, len(hex_data), 6):
            pixel_index = i // 6

            # Check if we need a line break
            if pixel_index > 0 and pixel_index % w == 0:
                flush_run()
                content.append({"text": "\n"})

            color = hex_data[i : i + 6]

            if color == current_color:
                # Same color, add to current run
                current_text += self.config.character
            else:
                # Color changed, flush current run and start new one
                flush_run()
                current_text = self.config.character
                current_color = color

        # Flush final run
        flush_run()

        return content

    def to_hover_command(self) -> str:
        """Generate Minecraft tellraw command with hover text."""
        content = self._generate_content()

        command = {
            "text": "Hover to view image.",
            "hover_event": {"action": "show_text", "value": content},
        }
        return f"tellraw @a {json.dumps(command, separators=(',', ':'))}"

    def to_text_command(self) -> str:
        """Generate Minecraft tellraw command with direct text display."""
        content = self._generate_content()
        return f"tellraw @a {json.dumps(content, separators=(',', ':'))}"

    @property
    def dimensions(self) -> tuple[int, int]:
        """Get current image dimensions (width, height)."""
        h, w = self._image.shape[:2]
        return (w, h)

    def gzip_export(self) -> tuple[int, int, bytes]:
        """Process image bytes and return dimensions and compressed hex data.

        Used for API into Velocity plugin.
        """
        w, h = self.dimensions
        return (w, h, gzip.compress(self.get_hex_data().encode("utf-8")))


# Convenience functions
def write_command(command: str, path: Path | str = "tellraw.txt") -> None:
    """Write command to file."""
    Path(path).write_text(command, encoding="utf-8")


# Example usage
if __name__ == "__main__":
    # Simple usage
    url = input("Enter image URL: ")
    img = MinecraftImage.from_url(url)
    write_command(img.to_text_command())
    print(f"Saved command for {img.dimensions[0]}x{img.dimensions[1]} image")
