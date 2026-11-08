"""Save actual Windows terminal pixels for this task; does not send UI input."""
import ctypes
import argparse
from ctypes import wintypes
from pathlib import Path
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--task', choices=['1.1', '1.2'], default='1.1')
args = parser.parse_args()

user32 = ctypes.WinDLL("user32")
gdi32 = ctypes.WinDLL("gdi32")
user32.GetWindowDC.restype = wintypes.HDC
gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
gdi32.CreateCompatibleDC.restype = wintypes.HDC
gdi32.CreateCompatibleBitmap.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]
gdi32.CreateCompatibleBitmap.restype = wintypes.HBITMAP
gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HANDLE]
gdi32.SelectObject.restype = wintypes.HANDLE
user32.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
gdi32.DeleteDC.argtypes = [wintypes.HDC]
gdi32.DeleteObject.argtypes = [wintypes.HANDLE]
gdi32.GetDIBits.argtypes = [wintypes.HDC, wintypes.HBITMAP, wintypes.UINT,
                          wintypes.UINT, ctypes.c_void_p, ctypes.c_void_p, wintypes.UINT]


class BitmapInfo(ctypes.Structure):
    _fields_ = [("size", wintypes.DWORD), ("width", wintypes.LONG),
                ("height", wintypes.LONG), ("planes", wintypes.WORD),
                ("bits", wintypes.WORD), ("compression", wintypes.DWORD),
                ("imagesize", wintypes.DWORD), ("x", wintypes.LONG),
                ("y", wintypes.LONG), ("used", wintypes.DWORD),
                ("important", wintypes.DWORD)]


@ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
def capture(hwnd, _):
    title = ctypes.create_unicode_buffer(512)
    user32.GetWindowTextW(hwnd, title, 512)
    if not title.value.startswith(f"23L-0885_A2_Ops - Task {args.task} - "):
        return True
    rect = wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    width, height = rect.right - rect.left, rect.bottom - rect.top
    dc = user32.GetWindowDC(hwnd)
    mem = gdi32.CreateCompatibleDC(dc)
    bitmap = gdi32.CreateCompatibleBitmap(dc, width, height)
    previous = gdi32.SelectObject(mem, bitmap)
    try:
        if not user32.PrintWindow(hwnd, mem, 2):
            raise RuntimeError("Window capture failed")
        info = BitmapInfo()
        info.size = ctypes.sizeof(BitmapInfo)
        info.width, info.height = width, -height
        info.planes, info.bits = 1, 32
        buffer = ctypes.create_string_buffer(width * height * 4)
        gdi32.GetDIBits(mem, bitmap, 0, height, buffer, ctypes.byref(info), 0)
        section = title.value.rsplit(" - ", 1)[-1]
        output = Path(__file__).resolve().parents[1] / "evidence" / f"task_{args.task.replace('.', '_')}" / f"{section}.png"
        Image.frombuffer("RGB", (width, height), buffer, "raw", "BGRX", 0, 1).save(output)
        print(f"Captured: {output} ({width} x {height})")
    finally:
        gdi32.SelectObject(mem, previous)
        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(mem)
        user32.ReleaseDC(hwnd, dc)
    return True


user32.EnumWindows(capture, 0)
