"""
PlaywrightComputer - Browser automation for Computer Use agent.

Based on Google ADK sample implementation.
Uses synchronous Playwright in a thread to avoid Windows asyncio subprocess issues.
"""

import asyncio
import atexit
import logging
import shutil
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass(frozen=True)
class BrowserConfig:
    """Configuration for browser automation behavior."""
    # Default screen size
    default_screen_size: Tuple[int, int] = (1280, 936)

    # Timeout settings (in milliseconds)
    navigation_timeout: int = 30000  # 30 seconds
    action_timeout: int = 10000  # 10 seconds

    # Scroll settings
    scroll_amount: int = 500  # pixels per scroll action

    # Wait times (in milliseconds)
    click_settle_time: int = 500  # time to wait after click
    hover_settle_time: int = 200  # time to wait after hover
    type_settle_time: int = 500  # time to wait after typing with enter


# Default configuration instance
BROWSER_CONFIG = BrowserConfig()

from google.adk.tools.computer_use.base_computer import (
    BaseComputer,
    ComputerEnvironment,
    ComputerState,
)
from playwright.sync_api import sync_playwright, BrowserContext, Page, Playwright


# Key mapping for Playwright
PLAYWRIGHT_KEY_MAP = {
    "backspace": "Backspace",
    "delete": "Delete",
    "enter": "Enter",
    "return": "Enter",
    "tab": "Tab",
    "escape": "Escape",
    "esc": "Escape",
    "up": "ArrowUp",
    "down": "ArrowDown",
    "left": "ArrowLeft",
    "right": "ArrowRight",
    "pageup": "PageUp",
    "pagedown": "PageDown",
    "home": "Home",
    "end": "End",
    "insert": "Insert",
    "space": " ",
    "ctrl": "Control",
    "control": "Control",
    "alt": "Alt",
    "shift": "Shift",
    "meta": "Meta",
    "cmd": "Meta",
    "command": "Meta",
    "win": "Meta",
    "f1": "F1",
    "f2": "F2",
    "f3": "F3",
    "f4": "F4",
    "f5": "F5",
    "f6": "F6",
    "f7": "F7",
    "f8": "F8",
    "f9": "F9",
    "f10": "F10",
    "f11": "F11",
    "f12": "F12",
}


class PlaywrightComputer(BaseComputer):
    """A browser automation computer using Playwright.

    Uses synchronous Playwright API running in a dedicated thread to avoid
    Windows asyncio subprocess compatibility issues.
    """

    def __init__(
        self,
        screen_size: Optional[Tuple[int, int]] = None,
        start_url: str = "https://www.google.com",
        search_engine_url: str = "https://www.google.com/search?q=",
        user_data_dir: Optional[str] = None,
        navigation_timeout: Optional[int] = None,
        action_timeout: Optional[int] = None,
        config: Optional[BrowserConfig] = None,
    ):
        cfg = config or BROWSER_CONFIG
        self._screen_size = screen_size or cfg.default_screen_size
        self._start_url = start_url
        self._search_engine_url = search_engine_url
        self._owns_user_data_dir = user_data_dir is None
        self._user_data_dir = user_data_dir or tempfile.mkdtemp()
        self._navigation_timeout = navigation_timeout or cfg.navigation_timeout
        self._action_timeout = action_timeout or cfg.action_timeout
        self._config = cfg

        logger.info(f"Initializing PlaywrightComputer with screen_size={self._screen_size}, "
                   f"navigation_timeout={self._navigation_timeout}ms, action_timeout={self._action_timeout}ms")

        self._playwright: Optional[Playwright] = None
        self._browser: Optional[BrowserContext] = None  # launch_persistent_context returns BrowserContext
        self._page: Optional[Page] = None
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="playwright")
        self._closed = False

        # Register cleanup on exit
        atexit.register(self._cleanup_sync)

    async def _run_in_thread(self, func, *args, **kwargs):
        """Run a synchronous function in the playwright thread from async context."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, lambda: func(*args, **kwargs))

    def _cleanup_sync(self):
        """Synchronous cleanup for atexit handler."""
        if self._closed:
            return
        self._closed = True
        logger.debug("Starting browser cleanup")

        with self._lock:
            try:
                if self._browser:
                    self._browser.close()
                    self._browser = None
                    logger.debug("Browser context closed")
            except Exception as e:
                logger.debug(f"Browser cleanup error (non-fatal): {e}")

            try:
                if self._playwright:
                    self._playwright.stop()
                    self._playwright = None
                    logger.debug("Playwright stopped")
            except Exception as e:
                logger.debug(f"Playwright cleanup error (non-fatal): {e}")

            self._page = None

            # Clean up temp directory if we created it
            if self._owns_user_data_dir and self._user_data_dir:
                try:
                    shutil.rmtree(self._user_data_dir, ignore_errors=True)
                    logger.debug(f"Temp directory removed: {self._user_data_dir}")
                except Exception as e:
                    logger.debug(f"Temp directory cleanup error (non-fatal): {e}")

    def _ensure_browser_sync(self):
        """Ensure browser is initialized (synchronous version)."""
        with self._lock:
            if self._playwright is None:
                logger.info("Starting Playwright")
                self._playwright = sync_playwright().start()

            if self._browser is None:
                logger.info("Launching browser context")
                self._browser = self._playwright.chromium.launch_persistent_context(
                    user_data_dir=self._user_data_dir,
                    headless=False,
                    viewport={"width": self._screen_size[0], "height": self._screen_size[1]},
                    args=["--disable-blink-features=AutomationControlled"],
                )

            if self._page is None or self._page.is_closed():
                if self._browser.pages:
                    self._page = self._browser.pages[0]
                else:
                    self._page = self._browser.new_page()
                logger.info(f"Navigating to start URL: {self._start_url}")
                self._page.goto(self._start_url, timeout=self._navigation_timeout)
                self._page.wait_for_load_state("domcontentloaded", timeout=self._navigation_timeout)

    def _get_state_sync(self) -> ComputerState:
        """Get current state synchronously."""
        screenshot_bytes = self._page.screenshot(type="png", timeout=self._action_timeout)
        return ComputerState(
            screenshot=screenshot_bytes,
            url=self._page.url,
        )

    async def screen_size(self) -> Tuple[int, int]:
        """Returns the screen size of the environment."""
        return self._screen_size

    async def environment(self) -> ComputerEnvironment:
        """Returns the environment type."""
        return ComputerEnvironment.ENVIRONMENT_BROWSER

    async def initialize(self) -> None:
        """Initialize the computer (open browser)."""
        await self._run_in_thread(self._ensure_browser_sync)

    async def open_web_browser(self) -> ComputerState:
        """Opens the web browser."""
        def _open():
            self._ensure_browser_sync()
            return self._get_state_sync()
        return await self._run_in_thread(_open)

    async def current_state(self) -> ComputerState:
        """Get current browser state with screenshot."""
        def _get_state():
            self._ensure_browser_sync()
            return self._get_state_sync()
        return await self._run_in_thread(_get_state)

    async def navigate(self, url: str) -> ComputerState:
        """Navigate to a URL."""
        def _navigate(url):
            self._ensure_browser_sync()
            logger.debug(f"Navigating to: {url}")
            self._page.goto(url, timeout=self._navigation_timeout)
            self._page.wait_for_load_state("domcontentloaded", timeout=self._navigation_timeout)
            return self._get_state_sync()
        return await self._run_in_thread(_navigate, url)

    async def go_back(self) -> ComputerState:
        """Go back in browser history."""
        def _go_back():
            self._ensure_browser_sync()
            logger.debug("Going back in history")
            self._page.go_back(timeout=self._navigation_timeout)
            self._page.wait_for_load_state("domcontentloaded", timeout=self._navigation_timeout)
            return self._get_state_sync()
        return await self._run_in_thread(_go_back)

    async def go_forward(self) -> ComputerState:
        """Go forward in browser history."""
        def _go_forward():
            self._ensure_browser_sync()
            logger.debug("Going forward in history")
            self._page.go_forward(timeout=self._navigation_timeout)
            self._page.wait_for_load_state("domcontentloaded", timeout=self._navigation_timeout)
            return self._get_state_sync()
        return await self._run_in_thread(_go_forward)

    async def search(self) -> ComputerState:
        """Navigate to the search engine home page."""
        def _search():
            self._ensure_browser_sync()
            # Navigate to search engine home (without query)
            base_url = self._search_engine_url.split("?")[0].removesuffix("/search")
            logger.debug(f"Navigating to search engine: {base_url}")
            self._page.goto(base_url or "https://www.google.com", timeout=self._navigation_timeout)
            self._page.wait_for_load_state("domcontentloaded", timeout=self._navigation_timeout)
            return self._get_state_sync()
        return await self._run_in_thread(_search)

    async def click_at(self, x: int, y: int) -> ComputerState:
        """Click at coordinates."""
        def _click(x, y):
            self._ensure_browser_sync()
            self._page.mouse.click(x, y)
            self._page.wait_for_timeout(self._config.click_settle_time)
            return self._get_state_sync()
        return await self._run_in_thread(_click, x, y)

    async def hover_at(self, x: int, y: int) -> ComputerState:
        """Hover at coordinates."""
        def _hover(x, y):
            self._ensure_browser_sync()
            self._page.mouse.move(x, y)
            self._page.wait_for_timeout(self._config.hover_settle_time)
            return self._get_state_sync()
        return await self._run_in_thread(_hover, x, y)

    async def drag_and_drop(
        self, x: int, y: int, destination_x: int, destination_y: int
    ) -> ComputerState:
        """Drag from (x, y) to (destination_x, destination_y)."""
        def _drag(x, y, destination_x, destination_y):
            self._ensure_browser_sync()
            self._page.mouse.move(x, y)
            self._page.mouse.down()
            self._page.mouse.move(destination_x, destination_y)
            self._page.mouse.up()
            return self._get_state_sync()
        return await self._run_in_thread(_drag, x, y, destination_x, destination_y)

    async def type_text_at(
        self,
        x: int,
        y: int,
        text: str,
        press_enter: bool = True,
        clear_before_typing: bool = True,
    ) -> ComputerState:
        """Type text at coordinates."""
        def _type(x, y, text, press_enter, clear_before_typing):
            self._ensure_browser_sync()
            self._page.mouse.click(x, y)

            if clear_before_typing:
                self._page.keyboard.press("Control+a")
                self._page.keyboard.press("Backspace")

            self._page.keyboard.type(text)

            if press_enter:
                self._page.keyboard.press("Enter")
                self._page.wait_for_timeout(self._config.type_settle_time)

            return self._get_state_sync()
        return await self._run_in_thread(_type, x, y, text, press_enter, clear_before_typing)

    async def scroll_document(
        self, direction: Literal["up", "down", "left", "right"]
    ) -> ComputerState:
        """Scroll the document."""
        def _scroll(direction):
            self._ensure_browser_sync()
            amount = self._config.scroll_amount

            if direction == "up":
                self._page.evaluate(f"window.scrollBy(0, -{amount})")
            elif direction == "down":
                self._page.evaluate(f"window.scrollBy(0, {amount})")
            elif direction == "left":
                self._page.evaluate(f"window.scrollBy(-{amount}, 0)")
            elif direction == "right":
                self._page.evaluate(f"window.scrollBy({amount}, 0)")

            return self._get_state_sync()
        return await self._run_in_thread(_scroll, direction)

    async def scroll_at(
        self,
        x: int,
        y: int,
        direction: Literal["up", "down", "left", "right"],
        magnitude: int,
    ) -> ComputerState:
        """Scroll at specific coordinates by magnitude."""
        def _scroll_at(x, y, direction, magnitude):
            self._ensure_browser_sync()
            self._page.mouse.move(x, y)

            delta_x, delta_y = 0, 0
            if direction == "up":
                delta_y = -magnitude
            elif direction == "down":
                delta_y = magnitude
            elif direction == "left":
                delta_x = -magnitude
            elif direction == "right":
                delta_x = magnitude

            self._page.mouse.wheel(delta_x, delta_y)
            return self._get_state_sync()
        return await self._run_in_thread(_scroll_at, x, y, direction, magnitude)

    async def wait(self, seconds: int) -> ComputerState:
        """Wait for n seconds."""
        def _wait(seconds):
            self._ensure_browser_sync()
            self._page.wait_for_timeout(seconds * 1000)
            return self._get_state_sync()
        return await self._run_in_thread(_wait, seconds)

    async def key_combination(self, keys: list[str]) -> ComputerState:
        """Press a key combination."""
        def _key_combo(keys):
            self._ensure_browser_sync()

            # Map keys to Playwright format
            mapped_keys = []
            for key in keys:
                mapped_key = PLAYWRIGHT_KEY_MAP.get(key.lower(), key)
                mapped_keys.append(mapped_key)

            # Build key combination string
            key_combo = "+".join(mapped_keys)
            self._page.keyboard.press(key_combo)
            return self._get_state_sync()
        return await self._run_in_thread(_key_combo, keys)

    async def close(self) -> None:
        """Close the browser and clean up resources."""
        if self._closed:
            return

        try:
            await self._run_in_thread(self._cleanup_sync)
        except Exception:
            # If running in thread fails, do sync cleanup directly
            self._cleanup_sync()

        # Unregister atexit handler since we've already cleaned up
        try:
            atexit.unregister(self._cleanup_sync)
        except Exception:
            pass

        # Shutdown executor and wait for pending tasks
        self._executor.shutdown(wait=True)
