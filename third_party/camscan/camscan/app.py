"""
This is an application used for scanning documents using a camera connected to
your computer, like your webcam. This module specifically implements the GUI
part of the application, as well as the code used to handle, post process, and
export the captured images.
"""

from datetime import datetime
import concurrent.futures
import functools
import logging
import os
import re
import math
import typing as t

import customtkinter as ctk
import cv2
import numpy as np
import PIL
import tkinter as tk
from tkinter import filedialog, messagebox

from camscan import postprocessing, widgets
from camscan.camera import Camera
from camscan import scanner
from camscan import __version__
import utils

logging.basicConfig(
    format="%(asctime)s.%(msecs)03d [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
)

# Define the window title
WINDOW_TITLE = f"Сканер VIAR {__version__}"

# Define the initial application window size
WINDOW_WIDTH = 1536
WINDOW_HEIGHT = 864

# Define the to wait before updating the camera feed (20ms)
CAMERA_FEED_WAIT_MS = 50
PREVIEW_MAX_WIDTH = 1280
PREVIEW_MAX_HEIGHT = 900

# Define constants related to the styling of widgets in the GUI
LEFT_MENU_PAD_X = 20
LEFT_MENU_PAD_Y = 5
RIGHT_MENU_PAD_X = 10
RIGHT_MENU_PAD_Y = 5
LEFT_MENU_PACK_KWARGS = dict(padx=LEFT_MENU_PAD_X, pady=LEFT_MENU_PAD_Y)
RIGHT_MENU_PACK_KWARGS = dict(padx=RIGHT_MENU_PAD_X, pady=RIGHT_MENU_PAD_Y)

# Keybind used to capture images with the cameras
CAPTURE_KEYBIND = "<space>"

# Specify supported file formats when exporting images as separate files.
# See the OpenCV documentation for more information on the supported file types:
# https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html
EXPORT_SEPARATE_FILE_TYPES = [
    "png",
    "bmp",
    "dib",
    "jpeg",
    "jpg",
    "jpe",
    "jp2",
    "webp",
    "pbm",
    "pgm",
    "ppm",
    "pxm",
    "pnm",
    "sr",
    "ras",
    "tiff",
    "tif",
    "exr",
    "hdr",
    "pic",
]

# Specify supported file formats when exporting images as a single merged file
EXPORT_MERGED_FILE_TYPES = [
    "pdf",
]

# Specify the supported postprocessing functions for the captured images
POSTPROCESSING_OPTIONS = {
    "Без обработки": postprocessing.dummy,
    "Коррекция белого": postprocessing.document_color,
    "Повышенная резкость": postprocessing.sharpen,
    "Оттенки серого": postprocessing.document_gray,
    "Чёрно-белый": postprocessing.black_and_white,
}


def process_image(
    image: cv2.Mat,
    option: str,
    bw_threshold: int = 50,
) -> cv2.Mat:
    """Apply a named processing mode, including the adjustable B/W threshold."""
    function = POSTPROCESSING_OPTIONS[option]
    if function is postprocessing.black_and_white:
        return function(image, threshold_strength=bw_threshold)
    return function(image)

# Define the list of pre-defined camera resolutions. In addition to these, the
# user can also enter custom resolutions manually.
RESOLUTIONS = [
    "2592x1944",
    "2048x1536",
    "1600x1200",
    "1280x1024",
    "640x480",
]

# Collection of tooltip strings shown for various widgets
TOOLTIPS = {
    # Left panel
    "camera_configuration": (
        "Open camera configuration for selecting camera and resolution"
    ),
    "camera_driver_settings": (
        "Open camera driver settings dialog (determined by the selected camera)"
    ),
    "postprocessing": "Set the postprocessing effect applied to the captured images",
    "system_appearance": "Set the user interface appearance of the application",
    "system_ui_scaling": "Set the user interface scale of the application",
    "free_capture_mode": (
        "Ignore the document detection algorithm and capture the entire image"
    ),
    "two_page_mode": "Split the captured image into equal left and right parts",
    "capture": (
        f"Capture an image and save to the captures pane (key bind {CAPTURE_KEYBIND})"
    ),
    "export_separate": "Export captures as separate files in a directory",
    "export_merged": "Export captures as a single merged file",
    # Right panel
    "select_all": "Select or deselect all captures",
    "delete": "Delete the selected captures",
    # Camera Configuration Window
    "camera_index": (
        "Select a camera by choosing its device index. Update this list with available"
        " devices using the camera identification button."
    ),
    "identify_cameras": (
        "Identify available cameras on the system and populate the camera index list"
    ),
    "camera_resolution": "Set the camera resolution from a preset list of resolutions",
    "custom_camera_resolution": (
        "Set a custom camera resolution using a string on the form <width>x<height>"
    ),
}


def opencv_to_pil_image(
    image: cv2.Mat,
    width: int = None,
    height: int = None,
) -> PIL.Image:
    """
    Given an OpenCV image, convert to to a PIL image. The function also supports
    resizing the image while keeping its original aspect ratio.
    :param image: The input OpenCV image
    :param width: Optional width to scale the image to
    :param width: Optional height to scale the image to
    :return: The image converted to a PIL image
    """
    if len(image.shape) == 2:
        converted = image
    else:
        converted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return PIL.Image.fromarray(
        utils.resize_with_aspect_ratio(
            image=converted,
            width=width,
            height=height,
        )
    )


def opencv_to_ctk_image(
    image: cv2.Mat,
    width: int = None,
    height: int = None,
) -> ctk.CTkImage:
    """
    Given an OpenCV image, convert to to a CTkImage. The function also supports
    resizing the image while keeping its original aspect ratio.
    :param image: The input OpenCV image
    :param width: Optional width to scale the image to
    :param width: Optional height to scale the image to
    :return: The image converted to a CTkImage
    """
    pil_image = opencv_to_pil_image(image=image, width=width, height=height)
    return ctk.CTkImage(
        pil_image,
        size=(pil_image.width, pil_image.height),
    )


class CaptureEntry:
    """
    Helper class for keeping track of the captured images. This class both
    contains the original OpenCV image, as well as the GUI element called an
    'Entry' which is comprised of several underlying widgets.
    :param image: The original OpenCV image capture from the camera
    :param name: A name given to the image which is displayed in the Entry
    :param index: The index number shown in the Entry
    :param master: The parent widget containing the Entry
    :param move_entry: A function used to move this entry up or down in the list
    """

    def __init__(
        self,
        image: cv2.Mat,
        name: str,
        index: int,
        master: ctk.CTkBaseClass,
        move_entry: t.Callable,
    ):
        self.var_selected = tk.IntVar(value=0)
        self.original_image = image.copy()
        self.current_image = image.copy()
        self.name = name
        self.frame = ctk.CTkFrame(master=master)
        self.frame.grid(row=index, column=0, padx=5, pady=5, sticky="nsew")

        self.move_up_button = ctk.CTkButton(
            master=self.frame,
            text="🔼",
            width=24,
            height=24,
            font=ctk.CTkFont(size=24),
            fg_color="transparent",
            command=functools.partial(move_entry, self, -1),
        )
        self.selection_checkbox = ctk.CTkCheckBox(
            master=self.frame,
            text=None,
            checkbox_width=24,
            checkbox_height=24,
            width=24,
            height=24,
            variable=self.var_selected,
        )
        self.move_down_button = ctk.CTkButton(
            master=self.frame,
            text="🔽",
            width=24,
            height=24,
            font=ctk.CTkFont(size=24),
            fg_color="transparent",
            command=functools.partial(move_entry, self, 1),
        )
        self.image_widget = ctk.CTkButton(
            master=self.frame,
            fg_color="transparent",
            text=None,
            command=self.open_image_viewer_window,
        )
        self.rotation_frame = ctk.CTkFrame(
            master=self.frame,
            fg_color="transparent",
        )
        self.rotate_left_button = ctk.CTkButton(
            master=self.rotation_frame,
            text="↶ 90°",
            width=82,
            command=functools.partial(self.rotate, clockwise=False),
        )
        self.rotate_right_button = ctk.CTkButton(
            master=self.rotation_frame,
            text="↷ 90°",
            width=82,
            command=functools.partial(self.rotate, clockwise=True),
        )
        self.index_label = ctk.CTkLabel(
            master=self.frame,
            text=str(index),
        )
        self.name_label = ctk.CTkLabel(
            master=self.frame,
            text=self.name,
        )
        self.index_label.grid(row=0, column=0, padx=5, pady=(5, 0), sticky="nsew")
        self.name_label.grid(row=0, column=1, padx=5, pady=(5, 0), sticky="nsew")
        self.move_up_button.grid(row=1, column=0, padx=5, pady=5)
        # The checkbox needs slightly more padding on the left to be aligned
        # with the up/down buttons in the same column
        self.selection_checkbox.grid(row=2, column=0, padx=(12, 5), pady=5)
        self.move_down_button.grid(row=3, column=0, padx=5, pady=5)
        self.image_widget.grid(
            row=1, column=1, rowspan=3, padx=5, pady=(0, 5), sticky="nsew"
        )
        self.rotation_frame.grid(row=4, column=1, padx=5, pady=(0, 5))
        self.rotate_left_button.pack(side=tk.LEFT, padx=(0, 4))
        self.rotate_right_button.pack(side=tk.LEFT, padx=(4, 0))

        self.set_current_image(image=image)

    def set_current_image(self, image: cv2.Mat):
        """
        Update the current displayed image of this Entry. This will not modify
        the original OpenCV image stored in this object. This will also update
        the smaller thumbnail picture used in the Entry.
        :param image: The new OpenCV image to set the current displayed image to
        """
        self.current_image = image.copy()
        thumbnail_image = opencv_to_ctk_image(image=image, width=230, height=400)
        self.image_widget.photo = thumbnail_image
        self.image_widget.configure(image=thumbnail_image)

    def rotate(self, clockwise: bool):
        """Rotate this scanned page by 90 degrees and refresh its thumbnail."""
        rotation = (
            cv2.ROTATE_90_CLOCKWISE
            if clockwise
            else cv2.ROTATE_90_COUNTERCLOCKWISE
        )
        self.original_image = cv2.rotate(self.original_image, rotation)
        rotated_current = cv2.rotate(self.current_image, rotation)
        self.set_current_image(rotated_current)

    def open_image_viewer_window(self):
        """
        Open an image viewer window displaying the current image of this Entry.
        """
        window = ctk.CTkToplevel()
        window.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        window.title(self.name)

        frame_widget = ctk.CTkFrame(master=window)
        image_widget = ctk.CTkLabel(master=frame_widget, text=None)

        # The current window size. Used to keep track of when it changes
        current_size = [0, 0]

        def _resize_image():
            """Resize the image to fill up the frame in the window"""
            max_width = frame_widget.winfo_width()
            max_height = frame_widget.winfo_height()

            # At startup, this area might be of size zero. If so, try later
            if not (max_width > 1 and max_height > 1):
                return

            # Convert the OpenCV image to a CTkImage to display in the widget
            new_image = opencv_to_ctk_image(
                image=self.current_image, width=max_width, height=max_height
            )
            image_widget.photo = new_image
            image_widget.configure(image=new_image)

        def _on_window_resize(event):
            # We need to make sure that the only widget that is allowed to
            # trigger the image resizing is the window itself. Otherwise, when
            # we update the image size, the image widget itself will generate
            # a 'Configure' event which will trigger this function again. That
            # will lead to an endless stream of events.
            if event.widget == window:
                return
            # We are only interested in updating the image size if the window
            # changes size. This event is also triggered when the window moves,
            # so we keep track of the current window size and compare to the new
            # one and update only if it changes.
            new_size = [event.width, event.height]
            if current_size == new_size:
                return
            # Modify values inplace to keep the reference intact
            current_size[:] = new_size[:]
            # Make sure that the containing frame widget has been updated
            # since it is that size which determines the maximum size of
            # the displayed image inside. In some cases, like when the user
            # expands the window to full-screen, this frame might not have
            # enough time to update before we attempt to update time image
            # within. Then it is not possible to fully expand the image to
            # the entire frame size. By manually calling the update here we
            # can ensure that the frame is the maximum size first.
            frame_widget.update()
            _resize_image()

        # Pack the widgets
        frame_widget.pack(fill=ctk.BOTH, expand=True)
        image_widget.pack()
        # Bind an event to when the window changes size to resize the image
        window.bind("<Configure>", _on_window_resize)
        # Make sure this window is on top of the main window
        window.lift()
        window.attributes("-topmost", True)
        # It can take some time for the window to set up its widgets and get its
        # proper size. Therefore, wait a little bit before displaying the image.
        window.after(ms=100, func=_resize_image)


class CamScanApp(ctk.CTk):
    """
    Application class for CamScan. This defines a CTk Window object containing
    the entire GUI of the application, as well as supporting code.

    Example usage:
        app = CameraScannerApp()
        app.mainloop()
    """

    def __init__(self):
        super().__init__()

        self.camera = Camera()
        self.entries = []
        self.var_postprocessing_option = tk.StringVar(
            value=list(POSTPROCESSING_OPTIONS.keys())[0]
        )
        self.var_two_page_mode = tk.IntVar(value=0)
        self.var_free_capture_mode = tk.IntVar(value=0)
        self.var_auto_portrait = tk.IntVar(value=1)
        self.var_bw_threshold = tk.IntVar(value=50)
        self.var_select_all_captures = tk.IntVar(value=0)
        self._preview_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="viar-preview",
        )
        self._preview_future = None
        self._last_submitted_frame_sequence = -1
        self._latest_capture = (None, None, None)
        self._bw_threshold_after_id = None
        self.var_merged_captures_file_type = tk.StringVar(
            value=EXPORT_MERGED_FILE_TYPES[0]
        )
        self.var_separate_captures_file_type = tk.StringVar(
            value=EXPORT_SEPARATE_FILE_TYPES[0]
        )
        self.var_select_all_captures = tk.IntVar(value=0)

        # configure window
        self.title(WINDOW_TITLE)
        self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.protocol("WM_DELETE_WINDOW", self.close_event)

        # Configure the grid layout
        self.grid_columnconfigure((0, 2), weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Configure the left sidebar
        self.left_sidebar_frame = ctk.CTkScrollableFrame(
            self,
            width=220,
            corner_radius=0,
        )

        # Add a label to the top of the sidebar
        self.left_sidebar_title_label = ctk.CTkLabel(
            self.left_sidebar_frame,
            text="Сканирование",
            font=ctk.CTkFont(size=20, weight="bold"),
        )

        # Add a button for the camera settings
        self.camera_settings_label = ctk.CTkLabel(
            self.left_sidebar_frame, text="Камера:", anchor="w"
        )
        self.configure_camera_button = ctk.CTkButton(
            self.left_sidebar_frame,
            text="Настроить камеру",
            command=self.configure_camera_event,
        )
        self.camera_settings_button = ctk.CTkButton(
            self.left_sidebar_frame,
            text="Параметры драйвера",
            command=self.camera.show_settings,
        )

        # Add a menu for the color settings
        self.postprocessing_menu_label = ctk.CTkLabel(
            self.left_sidebar_frame, text="Обработка:", anchor="w"
        )
        self.postprocessing_option_menu = ctk.CTkOptionMenu(
            self.left_sidebar_frame,
            values=list(POSTPROCESSING_OPTIONS.keys()),
            command=self.change_postprocessing_event,
            variable=self.var_postprocessing_option,
        )
        self.bw_threshold_label = ctk.CTkLabel(
            self.left_sidebar_frame,
            text="Сохранение бледного текста: 50%",
            anchor="w",
        )
        self.bw_threshold_slider = ctk.CTkSlider(
            self.left_sidebar_frame,
            from_=0,
            to=100,
            number_of_steps=100,
            variable=self.var_bw_threshold,
            command=self.change_bw_threshold_event,
        )
        self.bw_threshold_slider.configure(state="disabled")

        # Add a menu for the application UI appearance
        self.appearance_mode_label = ctk.CTkLabel(
            self.left_sidebar_frame, text="Оформление:", anchor="w"
        )
        self.appearance_mode_option_menu = ctk.CTkOptionMenu(
            self.left_sidebar_frame,
            values=["System", "Dark", "Light"],
            command=change_ui_appearance_event,
        )
        self.appearance_mode_option_menu.set("System")

        # Add a menu for the application UI scaling
        self.scaling_label = ctk.CTkLabel(
            self.left_sidebar_frame, text="Масштаб интерфейса:", anchor="w"
        )
        self.scaling_option_menu = ctk.CTkOptionMenu(
            self.left_sidebar_frame,
            values=["80%", "90%", "100%", "110%", "120%"],
            command=change_ui_scaling_event,
        )
        self.scaling_option_menu.set("100%")

        # Add a button for capturing the screen
        self.capture_image_label = ctk.CTkLabel(
            self.left_sidebar_frame, text="Режим:", anchor="w"
        )
        self.two_page_setting_check_box = ctk.CTkCheckBox(
            self.left_sidebar_frame,
            text="Разворот: две страницы",
            variable=self.var_two_page_mode,
        )
        self.free_capture_setting_check_box = ctk.CTkCheckBox(
            self.left_sidebar_frame,
            text="Вся область без обрезки",
            variable=self.var_free_capture_mode,
        )
        self.auto_portrait_check_box = ctk.CTkCheckBox(
            self.left_sidebar_frame,
            text="Автоповорот одиночного листа",
            variable=self.var_auto_portrait,
        )
        self.capture_image_button = ctk.CTkButton(
            self.left_sidebar_frame,
            text="Сканировать",
            command=self.capture_image,
        )
        self.camera_status_label = ctk.CTkLabel(
            self.left_sidebar_frame,
            text="Подключение камеры…",
            anchor="w",
        )

        # Add a menu for exporting separate captures
        self.export_separate_captures_label = ctk.CTkLabel(
            self.left_sidebar_frame, text="Сохранить изображения:", anchor="w"
        )
        self.export_separate_captures_option_menu = ctk.CTkComboBox(
            master=self.left_sidebar_frame,
            values=sorted(EXPORT_SEPARATE_FILE_TYPES),
            variable=self.var_separate_captures_file_type,
            state="readonly",
        )
        self.export_separate_captures_button = ctk.CTkButton(
            master=self.left_sidebar_frame,
            text="Сохранить отдельно",
            command=self.export_separate_captures,
        )

        # Add a menu for exporting merged captures
        self.export_merged_captures_label = ctk.CTkLabel(
            self.left_sidebar_frame, text="Готовый документ:", anchor="w"
        )
        self.export_merged_captures_option_menu = ctk.CTkComboBox(
            master=self.left_sidebar_frame,
            values=sorted(EXPORT_MERGED_FILE_TYPES),
            variable=self.var_merged_captures_file_type,
            state="readonly",
        )
        self.export_merged_captures_button = ctk.CTkButton(
            master=self.left_sidebar_frame,
            text="Сохранить PDF",
            command=self.export_merged_captures,
        )

        # Organize left menu items
        self.left_sidebar_title_label.pack(padx=LEFT_MENU_PAD_X, pady=20)
        self.camera_settings_label.pack(**LEFT_MENU_PACK_KWARGS)
        self.configure_camera_button.pack(**LEFT_MENU_PACK_KWARGS)
        self.camera_settings_button.pack(**LEFT_MENU_PACK_KWARGS)
        self.postprocessing_menu_label.pack(**LEFT_MENU_PACK_KWARGS)
        self.postprocessing_option_menu.pack(**LEFT_MENU_PACK_KWARGS)
        self.bw_threshold_label.pack(**LEFT_MENU_PACK_KWARGS)
        self.bw_threshold_slider.pack(**LEFT_MENU_PACK_KWARGS)
        self.appearance_mode_label.pack(**LEFT_MENU_PACK_KWARGS)
        self.appearance_mode_option_menu.pack(**LEFT_MENU_PACK_KWARGS)
        self.scaling_label.pack(**LEFT_MENU_PACK_KWARGS)
        self.scaling_option_menu.pack(**LEFT_MENU_PACK_KWARGS)
        self.capture_image_label.pack(**LEFT_MENU_PACK_KWARGS)
        self.free_capture_setting_check_box.pack(**LEFT_MENU_PACK_KWARGS)
        self.auto_portrait_check_box.pack(**LEFT_MENU_PACK_KWARGS)
        self.two_page_setting_check_box.pack(**LEFT_MENU_PACK_KWARGS)
        self.capture_image_button.pack(**LEFT_MENU_PACK_KWARGS)
        self.camera_status_label.pack(**LEFT_MENU_PACK_KWARGS)
        self.export_separate_captures_label.pack(**LEFT_MENU_PACK_KWARGS)
        self.export_separate_captures_option_menu.pack(**LEFT_MENU_PACK_KWARGS)
        self.export_separate_captures_button.pack(**LEFT_MENU_PACK_KWARGS)
        self.export_merged_captures_label.pack(**LEFT_MENU_PACK_KWARGS)
        self.export_merged_captures_option_menu.pack(**LEFT_MENU_PACK_KWARGS)
        self.export_merged_captures_button.pack(**LEFT_MENU_PACK_KWARGS)

        # Configure the central widget showing the camera feed
        self.camera_image_widget = ctk.CTkLabel(self, text=None, padx=0, pady=0)
        self.camera_image_label = ctk.CTkLabel(
            self,
            text="Камера не найдена",
            font=ctk.CTkFont(size=20, weight="bold"),
            padx=0,
            pady=0,
        )

        # Configure the right sidebar
        self.right_sidebar_frame = ctk.CTkFrame(self, corner_radius=0)
        self.right_sidebar_frame.grid_rowconfigure((0, 1), weight=0)
        self.right_sidebar_frame.grid_rowconfigure(2, weight=1)

        # Add a label to the top of the sidebar
        self.right_sidebar_title_label = ctk.CTkLabel(
            self.right_sidebar_frame,
            text="Страницы",
            font=ctk.CTkFont(size=20, weight="bold"),
        )

        # Create scrollable frame for the captures
        self.scrollable_frame = ctk.CTkScrollableFrame(
            master=self.right_sidebar_frame,
            width=320,
        )
        self.scrollable_frame.grid_columnconfigure(0, weight=1)

        # Add widgets for selcting all captures and deleting
        self.select_all_captures_check_box = ctk.CTkCheckBox(
            self.right_sidebar_frame,
            text="Выбрать все",
            command=self.select_all_entries,
            variable=self.var_select_all_captures,
        )

        self.delete_captures_button = ctk.CTkButton(
            master=self.right_sidebar_frame,
            text="🗑",
            width=24,
            height=24,
            font=ctk.CTkFont(size=24),
            fg_color="transparent",
            text_color=("#b42318", "#ff7b72"),
            hover_color=("#fee4e2", "#5c1a1a"),
            command=self.delete_selected_entries,
        )

        # Organize right menu items
        self.right_sidebar_title_label.grid(
            row=0, column=0, columnspan=2, padx=LEFT_MENU_PAD_X, pady=20
        )
        self.select_all_captures_check_box.grid(
            row=1, column=0, **RIGHT_MENU_PACK_KWARGS
        )
        self.delete_captures_button.grid(row=1, column=1, **RIGHT_MENU_PACK_KWARGS)
        self.scrollable_frame.grid(
            row=2, column=0, columnspan=2, sticky="nsew", **RIGHT_MENU_PACK_KWARGS
        )

        # Organize main frames
        self.left_sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.camera_image_widget.grid(row=0, column=1, sticky="nsew")
        self.camera_image_label.grid(row=0, column=1, sticky="nsew")
        self.camera_image_widget.lift()
        self.right_sidebar_frame.grid(row=0, column=2, rowspan=4, sticky="nsew")

        # Tooltips
        # Left menu
        widgets.Tooltip(
            widget=self.configure_camera_button,
            text=TOOLTIPS["camera_configuration"],
        )
        widgets.Tooltip(
            widget=self.camera_settings_button,
            text=TOOLTIPS["camera_driver_settings"],
        )
        widgets.Tooltip(
            widget=self.postprocessing_option_menu,
            text=TOOLTIPS["postprocessing"],
        )
        widgets.Tooltip(
            widget=self.bw_threshold_slider,
            text=(
                "Увеличьте значение, если пропадают тонкие буквы. "
                "Уменьшите, если появляется лишний чёрный шум."
            ),
        )
        widgets.Tooltip(
            widget=self.appearance_mode_option_menu,
            text=TOOLTIPS["system_appearance"],
        )
        widgets.Tooltip(
            widget=self.scaling_option_menu,
            text=TOOLTIPS["system_ui_scaling"],
        )
        widgets.Tooltip(
            widget=self.free_capture_setting_check_box,
            text=TOOLTIPS["free_capture_mode"],
        )
        widgets.Tooltip(
            widget=self.two_page_setting_check_box,
            text=TOOLTIPS["two_page_mode"],
        )
        widgets.Tooltip(
            widget=self.capture_image_button,
            text=TOOLTIPS["capture"],
        )
        widgets.Tooltip(
            widget=self.export_separate_captures_button,
            text=TOOLTIPS["export_separate"],
        )
        widgets.Tooltip(
            widget=self.export_merged_captures_button,
            text=TOOLTIPS["export_merged"],
        )
        # Right menu
        widgets.Tooltip(
            widget=self.select_all_captures_check_box,
            text=TOOLTIPS["select_all"],
        )
        widgets.Tooltip(
            widget=self.delete_captures_button,
            text=TOOLTIPS["delete"],
        )

        # Hotkeys
        self.bind(sequence=CAPTURE_KEYBIND, func=lambda _: self.capture_image())

        self.show_frame()

    def close_event(self):
        """Release the V4L2 stream before closing the application."""
        self.camera.close()
        self._preview_executor.shutdown(wait=False, cancel_futures=True)
        self.destroy()

    def capture(self) -> tuple[cv2.Mat, cv2.Mat, np.ndarray]:
        """
        Capture an image from the camera and run the document detection
        algorithm on the resulting image.
        :return:
            A tuple consisting of the raw image, the extracted warped image, and
            a numpy array describing the contours of the found document. If the
            video capture could not read a frame successfully, return None.
        """
        cached_raw, cached_warped, cached_contour = self._latest_capture
        if cached_raw is not None:
            return (
                cached_raw.copy(),
                None if cached_warped is None else cached_warped.copy(),
                None if cached_contour is None else cached_contour.copy(),
            )

        img_capture = self.camera.capture()
        if img_capture is not None:
            scan_result = scanner.main(img_capture)
            return img_capture, scan_result.warped, scan_result.contour
        return (None, None, None)

    @staticmethod
    def _process_preview_frame(
        raw_image: cv2.Mat,
        postprocessing_option: str,
        bw_threshold: int,
        free_capture_mode: bool,
    ):
        """Run expensive detection and preview processing outside Tk's thread."""
        scan_result = scanner.main(raw_image)
        image = process_image(
            raw_image,
            option=postprocessing_option,
            bw_threshold=bw_threshold,
        )
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if not free_capture_mode and scan_result.contour is not None:
            image = utils.draw_contour(
                image=image,
                contour=scan_result.contour,
                thickness=4,
            )
        image = utils.resize_with_aspect_ratio(
            image,
            width=PREVIEW_MAX_WIDTH,
            height=PREVIEW_MAX_HEIGHT,
        )
        return (
            image,
            raw_image,
            scan_result.warped,
            scan_result.contour,
        )

    def show_frame(self):
        """
        Poll completed background work and update Tk without doing OpenCV's
        expensive detection and filtering in the GUI thread.
        """
        max_width = self.camera_image_widget.winfo_width()
        max_height = self.camera_image_widget.winfo_height()
        if not (max_width > 1 and max_height > 1):
            self.after(ms=CAMERA_FEED_WAIT_MS, func=self.show_frame)
            return

        raw_image, frame_sequence = self.camera.capture_with_sequence()
        if raw_image is not None:
            remaining = self.camera.warmup_remaining
            if remaining > 0:
                self.camera_status_label.configure(
                    text=f"Прогрев камеры: {math.ceil(remaining)} с",
                    text_color=("#9a6700", "#f2cc60"),
                )
                self.capture_image_button.configure(state="disabled")
            else:
                self.camera_status_label.configure(
                    text="Камера готова",
                    text_color=("#1a7f37", "#56d364"),
                )
                self.capture_image_button.configure(state="normal")
        else:
            self.camera_image_label.lift()
            self.camera_status_label.configure(
                text="VIAR-500A не найдена",
                text_color=("#cf222e", "#ff7b72"),
            )
            self.capture_image_button.configure(state="disabled")

        if self._preview_future is not None and self._preview_future.done():
            try:
                image, capture_raw, warped, contour = self._preview_future.result()
                self._latest_capture = (capture_raw, warped, contour)
                image = opencv_to_ctk_image(
                    image=image,
                    width=max_width,
                    height=max_height,
                )
                self.camera_image_widget.photo = image
                self.camera_image_widget.configure(image=image)
                self.camera_image_widget.lift()
            except Exception:
                logging.exception("Preview processing failed")
            finally:
                self._preview_future = None

        if (
            raw_image is not None
            and self._preview_future is None
            and frame_sequence != self._last_submitted_frame_sequence
        ):
            self._last_submitted_frame_sequence = frame_sequence
            self._preview_future = self._preview_executor.submit(
                self._process_preview_frame,
                raw_image,
                self.var_postprocessing_option.get(),
                self.var_bw_threshold.get(),
                bool(self.var_free_capture_mode.get()),
            )

        self.after(ms=CAMERA_FEED_WAIT_MS, func=self.show_frame)

    def capture_image(self):
        """
        Capture an image using the camera.
        """
        if not self.camera.is_warmed_up:
            messagebox.showwarning(
                title="Камера прогревается",
                message=(
                    "Подождите ещё "
                    f"{math.ceil(self.camera.warmup_remaining)} с: "
                    "камера настраивает экспозицию и баланс белого."
                ),
            )
            return

        full_image, warped_image, _ = self.capture()

        # If we are using Free Capture mode, use the full uncropped image
        if self.var_free_capture_mode.get():
            if full_image is not None:
                image = full_image
            else:
                messagebox.showerror(
                    title="Ошибка",
                    message="Не удалось получить изображение с камеры.",
                )
                return
        # Otherwise, use the warped cropped extracted image
        elif warped_image is not None:
            image = warped_image
        else:
            messagebox.showerror(
                title="Лист не найден",
                message=(
                    "Не удалось найти границы листа. Положите его на тёмную "
                    "поверхность либо включите «Вся область без обрезки»."
                ),
            )
            return

        # A single A4 sheet is normally portrait.  If it was placed sideways,
        # turn the extracted page after perspective correction.  A book spread
        # and free-area capture deliberately keep their landscape orientation.
        if (
            self.var_auto_portrait.get()
            and not self.var_two_page_mode.get()
            and not self.var_free_capture_mode.get()
            and image.shape[1] > image.shape[0]
        ):
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Give the capture a name using a timestamp string
        timestamp_str = datetime.now().strftime(r"%Y%m%d_%H%M%S_%f")

        # If we are using two-page mode, cut the image into left and right parts
        if self.var_two_page_mode.get():
            cutoff_width = image.shape[1] // 2
            left_image = image[:, :cutoff_width]
            right_image = image[:, cutoff_width:]
            new_entries = [
                CaptureEntry(
                    master=self.scrollable_frame,
                    image=left_image,
                    name=f"{timestamp_str}_1",
                    index=len(self.entries) + 1,
                    move_entry=self.move_entry,
                ),
                CaptureEntry(
                    master=self.scrollable_frame,
                    image=right_image,
                    name=f"{timestamp_str}_2",
                    index=len(self.entries) + 2,
                    move_entry=self.move_entry,
                ),
            ]
        # Otherwise, take the entire image and as as an entry
        else:
            new_entries = [
                CaptureEntry(
                    master=self.scrollable_frame,
                    image=image,
                    name=timestamp_str,
                    index=len(self.entries) + 1,
                    move_entry=self.move_entry,
                )
            ]

        # If a postprocessing function is selected, apply it to the new images
        self.apply_postprocessing(entries=new_entries)
        self.entries += new_entries

        # Update the scrollable frame with the entries and move it to the bottom
        self.scrollable_frame.update()
        self.scrollable_frame._parent_canvas.yview_moveto(1.0)

    def move_entry(self, entry: CaptureEntry, distance: int):
        """
        Move an entry in the capture list either up or down by some distance.
        :param entry: The CaptureEntry to move
        :param distance: The move distance (-1 to move up, or +1 to move down)
        """
        # Find the current index 'i' of the entry and the destination index 'j'
        i = self.entries.index(entry)
        j = i + distance

        # If the destination index is out of range, skip the operation
        if j < 0 or j >= len(self.entries):
            return

        # Get the current grid rows of the entries. This is not really needed
        # since the indices i and j should be the same as the grid row
        i_grid_row = self.entries[i].frame.grid_info()["row"]
        j_grid_row = self.entries[j].frame.grid_info()["row"]

        # Switch grid positions
        logging.debug(f"Switching entries in rows {i_grid_row} and {j_grid_row}")
        self.entries[i].frame.grid(row=j_grid_row)
        self.entries[j].frame.grid(row=i_grid_row)

        # Switch index labels
        self.entries[i].index_label.configure(text=str(j + 1))
        self.entries[j].index_label.configure(text=str(i + 1))

        # Switch the locations of the entries in the list
        self.entries[i], self.entries[j] = self.entries[j], self.entries[i]

    def select_all_entries(self):
        """
        Select or deselect all current capture entries.
        """
        # Depending on the state of the checkbox, select or deselect all entries
        select = self.var_select_all_captures.get()
        for entry in self.entries:
            if select:
                entry.selection_checkbox.select()
            else:
                entry.selection_checkbox.deselect()

    def delete_selected_entries(self):
        """
        Delete all the currently selected capture entries.
        """
        # Select the entries based on the state of their checkbox variable
        entries_to_delete = [e for e in self.entries if e.var_selected.get()]
        logging.debug(f"Removing {len(entries_to_delete)} entries")

        # For each such entry, destroy its frame and remove from the list
        for entry in entries_to_delete:
            entry.frame.destroy()
            self.entries.remove(entry)

        # After deletion, update the grid positions of the remaining entries
        for i, entry in enumerate(self.entries):
            entry.frame.grid(row=i)

        # There is some peculiar behavior of the scrollbar in the scrollable
        # frame when all entries are deleted at once. If there are enough
        # entries (around 5+) to make the scrollbar active, and it is scrolled
        # all the way to the bottom, it will not correctly update its allowed
        # range of scrolling when the entries are deleted. Instead, it will
        # still be scrolled all the way to the bottom, with the scrollable frame
        # being completely empty. After testing, it seems that one (hacky)
        # solution to this is to do the following:
        # - Add back a widget in the grid (a dummy frame in this solution)
        # - Move the scroll all the way back up to the top (yview_moveto)
        # - Call the update function on the scrollable frame
        # - Delete the dummy frame after it is no longer needed.
        # By adding this dummy widget, it seems to make the update of the
        # scrollable frame also update the scrollbar to the correct range.
        # Without it, this does not work!
        if len(self.entries) == 0:
            dummy_frame = ctk.CTkFrame(master=self.scrollable_frame)
            dummy_frame.grid(row=0, column=0)
            self.scrollable_frame._parent_canvas.yview_moveto(0.0)
            self.scrollable_frame.update()
            dummy_frame.destroy()

        # Uncheck the checkbox for selecting all entries
        self.select_all_captures_check_box.deselect()

    def export_merged_captures(self):
        """
        Export all the current captures as a single merged file.
        """
        # Get the currently select file type to export as
        file_type = self.var_merged_captures_file_type.get()

        n = len(self.entries)

        # If there are no captures, show a message box and return
        if n == 0:
            messagebox.showerror(
                title="Нет страниц",
                message="Сначала отсканируйте хотя бы одну страницу.",
            )
            return

        # Create the name of the output file as a timestamp string
        timestamp_str = datetime.now().strftime(r"%Y%m%d_%H%M%S")
        initialfile = f"VIAR_{timestamp_str}.{file_type}"

        # Bring up a dialog asking for the output file path
        file_path = filedialog.asksaveasfilename(
            initialfile=initialfile,
            defaultextension=".pdf",
            filetypes=[("PDF Documents", "*.pdf"), ("All Files", "*.*")],
        )

        # If no output file was chosen (e.g. dialog cancelled), return
        if not file_path:
            return

        # Convert the captured OpenCV images to PIL images
        images = [opencv_to_pil_image(entry.current_image) for entry in self.entries]

        # The PIL save functionality requires that we initiate it from a single
        # image, then append the remaining images as function parameter
        first_image = images[0]
        remaining_images = images[1:]
        first_image.save(
            file_path,
            save_all=True,
            append_images=remaining_images,
            resolution=300.0,
        )

        # Show a message box indicating to the user that the export succeeded
        messagebox.showinfo(
            title="PDF сохранён",
            message=f"Страниц: {n}\nФайл: {file_path}",
        )

    def export_separate_captures(self):
        """
        Export all the current captures as separate files in a directory.
        """
        # Get the currently select file type to export as
        file_type = self.var_separate_captures_file_type.get()

        n = len(self.entries)

        # If there are no captures, show a message box and return
        if n == 0:
            messagebox.showerror(
                title="Нет страниц",
                message="Сначала отсканируйте хотя бы одну страницу.",
            )
            return

        # Bring up a dialog asking for the output directory path
        file_dialog_dir = filedialog.askdirectory()

        # If no output directory was chosen (e.g. dialog cancelled), return
        if not file_dialog_dir:
            return

        # Create the name of the output directory as a timestamp string
        timestamp_str = datetime.now().strftime(r"%Y%m%d_%H%M%S")
        output_dir = f"{file_dialog_dir}/VIAR_{timestamp_str}"
        os.makedirs(output_dir, exist_ok=True)

        # For each capture, write the image to the output directory
        for i, entry in enumerate(self.entries, start=1):
            cv2.imwrite(
                filename=f"{output_dir}/{i}_{entry.name}.{file_type}",
                img=entry.current_image,
            )

        # Show a message box indicating to the user that the export succeeded
        messagebox.showinfo(
            title="Изображения сохранены",
            message=f"Файлов: {n}\nПапка: {output_dir}",
        )

    def change_postprocessing_event(self, *args):
        """
        Handle the event when the chose postprocessing function changes.
        When it does, apply it to all current capture entries.
        """
        is_black_and_white = (
            self.var_postprocessing_option.get() == "Чёрно-белый"
        )
        self.bw_threshold_slider.configure(
            state="normal" if is_black_and_white else "disabled"
        )
        self._last_submitted_frame_sequence = -1
        self.apply_postprocessing(entries=self.entries)

    def change_bw_threshold_event(self, value):
        """Update the B/W text-density threshold, debouncing page reprocessing."""
        threshold = int(round(float(value)))
        self.var_bw_threshold.set(threshold)
        self.bw_threshold_label.configure(
            text=f"Сохранение бледного текста: {threshold}%"
        )
        self._last_submitted_frame_sequence = -1

        if self.var_postprocessing_option.get() != "Чёрно-белый":
            return
        if self._bw_threshold_after_id is not None:
            self.after_cancel(self._bw_threshold_after_id)
        self._bw_threshold_after_id = self.after(
            180,
            self._apply_debounced_bw_threshold,
        )

    def _apply_debounced_bw_threshold(self):
        self._bw_threshold_after_id = None
        self.apply_postprocessing(entries=self.entries)

    def apply_postprocessing(self, entries: list[CaptureEntry]):
        """
        Apply currently chosen postprocessing function to given capture entries.
        :param entries: The capture entries to apply the postprocessing to
        """
        postprocessing_option = self.var_postprocessing_option.get()
        for entry in entries:
            new_image = process_image(
                entry.original_image,
                option=postprocessing_option,
                bw_threshold=self.var_bw_threshold.get(),
            )
            entry.set_current_image(image=new_image)

    def configure_camera_event(self):
        """
        Handle the event for configuring the camera. This is done by opening a
        separate window with the available configuration.
        """

        def _set_camera_index(index: int):
            """Callback for changing the camera device index"""
            self.camera.set_index(index=int(index))

        def _update_available_camera_indices():
            """Callback for updating the available camera device indices"""
            camera_indices = self.camera.get_available_device_indices()
            camera_index_combobox.configure(values=list(map(str, camera_indices)))
            if camera_indices:
                camera_index_combobox.set(value=str(camera_indices[0]))
                _set_camera_index(camera_indices[0])

        def _set_camera_resolution(resolution_string: str):
            """Set the camera resolution from a resolution string"""
            regex = re.compile(r"^(\d+)x(\d+)$")
            matches = regex.findall(resolution_string)
            if matches:
                resolution = (int(matches[0][0]), int(matches[0][1]))
                self.camera.set_resolution(resolution=resolution)
            else:
                messagebox.showerror(
                    title="Ошибка",
                    message=(
                        "The resolution string must be on the form '<width>x<height>'"
                    ),
                )

        def _set_camera_rotation(rotation_string: str):
            rotations = {
                "Без поворота": 0,
                "90° вправо": 90,
                "180°": 180,
                "90° влево": 270,
            }
            self.camera.set_rotation(rotations[rotation_string])

        # Create a new top-level window for the camera configuration
        window = ctk.CTkToplevel()
        window.resizable(width=False, height=False)
        window.title("Настройка камеры")

        # Define the variables
        possible_camera_indices = list(map(str, range(10)))
        current_resolution_string = "x".join(map(str, self.camera.resolution))
        var_camera_index = tk.StringVar(value=str(self.camera.index))
        var_camera_resolution = tk.StringVar(value=current_resolution_string)
        var_custom_camera_resolution = tk.StringVar(value=current_resolution_string)
        rotation_names = {
            0: "Без поворота",
            90: "90° вправо",
            180: "180°",
            270: "90° влево",
        }
        var_camera_rotation = tk.StringVar(
            value=rotation_names[self.camera.rotation]
        )

        # Define the widgets
        camera_index_label = ctk.CTkLabel(
            master=window,
            text="Номер камеры:",
        )
        camera_index_combobox = ctk.CTkOptionMenu(
            master=window,
            values=possible_camera_indices,
            command=_set_camera_index,
            state="readonly",
            variable=var_camera_index,
        )
        find_camera_indices_button = ctk.CTkButton(
            master=window,
            text="Найти камеры",
            command=_update_available_camera_indices,
        )
        camera_resolution_label = ctk.CTkLabel(
            master=window,
            text="Разрешение:",
        )
        camera_resolution_combobox = ctk.CTkOptionMenu(
            master=window,
            values=RESOLUTIONS,
            command=_set_camera_resolution,
            variable=var_camera_resolution,
        )
        custom_camera_resolution_label = ctk.CTkLabel(
            master=window,
            text="Другое разрешение:",
        )
        custom_camera_resolution_entry = ctk.CTkEntry(
            master=window, textvariable=var_custom_camera_resolution
        )

        custom_camera_resolution_button = ctk.CTkButton(
            master=window,
            text="Применить разрешение",
            command=lambda: _set_camera_resolution(
                var_custom_camera_resolution.get()
            ),
        )
        camera_rotation_label = ctk.CTkLabel(
            master=window,
            text="Ориентация изображения:",
        )
        camera_rotation_combobox = ctk.CTkOptionMenu(
            master=window,
            values=list(rotation_names.values()),
            command=_set_camera_rotation,
            variable=var_camera_rotation,
        )

        # Pack the widgets
        pack_kwargs = dict(padx=10, pady=5)
        camera_index_label.pack(padx=10, pady=(20, 5))
        find_camera_indices_button.pack(**pack_kwargs)
        camera_index_combobox.pack(**pack_kwargs)
        camera_resolution_label.pack(**pack_kwargs)
        camera_resolution_combobox.pack(**pack_kwargs)
        camera_rotation_label.pack(**pack_kwargs)
        camera_rotation_combobox.pack(**pack_kwargs)
        custom_camera_resolution_label.pack(**pack_kwargs)
        custom_camera_resolution_entry.pack(**pack_kwargs)
        custom_camera_resolution_button.pack(padx=10, pady=(5, 20))

        # Add tooltips
        widgets.Tooltip(
            widget=camera_index_combobox,
            text=TOOLTIPS["camera_index"],
        )
        widgets.Tooltip(
            widget=find_camera_indices_button,
            text=TOOLTIPS["identify_cameras"],
        )
        widgets.Tooltip(
            widget=camera_resolution_combobox,
            text=TOOLTIPS["camera_resolution"],
        )
        widgets.Tooltip(
            widget=custom_camera_resolution_button,
            text=TOOLTIPS["custom_camera_resolution"],
        )

        # Make sure this window is on top of the main window
        # We could simply just set topmost to True and leave it at that, but
        # that will prevent the Tooltips from working properly. We can instead
        # set it to topmost temporarily, use grab_set to set focus, and then
        # set topmost back to False. This brings the window to the front.
        # From the documentation it seems that using .lift(aboveThis=self) would
        # work, but I was not able to make that work.
        window.attributes("-topmost", True)
        window.grab_set()
        window.attributes("-topmost", False)


def change_ui_appearance_event(new_appearance_mode: str):
    """
    Handle the event to update the application appearance.
    :param new_appearance_mode: The appearance mode (System, Dark, Light)
    """
    ctk.set_appearance_mode(new_appearance_mode)


def change_ui_scaling_event(new_scaling: str):
    """
    Handle the event to update the application UI scale.
    :param new_scaling: The new scaling string on the form XX%
    """
    new_scaling_float = int(new_scaling.replace("%", "")) / 100
    ctk.set_widget_scaling(new_scaling_float)


if __name__ == "__main__":
    app = CamScanApp()
    app.mainloop()
