from pathlib import Path
from PIL import Image, ImageOps, ImageDraw, ImageFont


#
"""

IMAGE_PATHS = [
    r"Synthetic Experiments/Plots/1_magic_numbers_results_std_kde_scaled.png",
    r"Synthetic Experiments/Plots/1_magic_numbers_results_center_distance_kde_scaled.png",
    r"Synthetic Experiments/Plots/1_magic_numbers_results_gauss_threshold_kde_scaled.png",
    r"Synthetic Experiments/Plots/1_magic_numbers_results_overlap_threshold_kde_scaled.png",
    r"Synthetic Experiments/Plots/1_magic_numbers_results_num_samples_kde_scaled.png",
    r"Synthetic Experiments/Plots/1_magic_numbers_results_contamination_rate_kde_scaled.png",
]

OUTPUT_PATH = r"Synthetic Experiments/Plots/kde_1_mn_combined.png"
"""

IMAGE_PATHS = [
    r"Synthetic Experiments/Plots/2_magic_numbers_results_std_kde_scaled.png",
    r"Synthetic Experiments/Plots/2_magic_numbers_results_center_distance_kde_scaled.png",
    r"Synthetic Experiments/Plots/2_magic_numbers_results_gauss_threshold_kde_scaled.png",
    r"Synthetic Experiments/Plots/2_magic_numbers_results_overlap_threshold_kde_scaled.png",
    r"Synthetic Experiments/Plots/2_magic_numbers_results_num_samples_kde_scaled.png",
    r"Synthetic Experiments/Plots/2_magic_numbers_results_contamination_rate_kde_scaled.png",
]

OUTPUT_PATH = r"Synthetic Experiments/Plots/kde_2_mn_combined.png"

# ------------------------------------------------------------
# LAYOUT
# ------------------------------------------------------------
# Number of columns in the collage.
#
# 2 = recommended for your Elsevier single-column figure
#
COLUMNS = 3

# Space between columns (pixels)
HORIZONTAL_GAP = 50

# Space between rows (pixels)
VERTICAL_GAP = 50

# Outer margin around the whole collage
MARGIN = 80


OUTPUT_WIDTH = 3600


# ------------------------------------------------------------
# PANEL LABELS
# ------------------------------------------------------------
SHOW_LABELS = True

# Distance of the label from the top-left corner of each panel
LABEL_MARGIN_X = 25
LABEL_MARGIN_Y = 20

# Label font size
LABEL_FONT_SIZE = 55


# ------------------------------------------------------------
# APPEARANCE
# ------------------------------------------------------------
BACKGROUND = "white"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_image(path):
    """Load an image and convert it to RGBA."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Image not found:\n{path}"
        )

    return Image.open(path).convert("RGBA")


def get_font(size):
    """Find a suitable font on macOS, Linux, or Windows."""

    font_candidates = [
        # macOS
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",

        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",

        # Windows
        "C:/Windows/Fonts/arial.ttf",
    ]

    for font_path in font_candidates:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size)

    # Fallback
    return ImageFont.load_default()


def fit_image(image, max_width, max_height):
    """
    Resize an image so that it fits inside the specified box
    while preserving its original aspect ratio.
    """

    return ImageOps.contain(
        image,
        (max_width, max_height),
        method=Image.Resampling.LANCZOS
    )


def generate_panel_labels(n):
    """
    Generate labels:
    (a), (b), (c), ...
    """

    labels = []

    for i in range(n):
        if i < 26:
            labels.append(f"({chr(97 + i)})")
        else:
            labels.append(f"({i + 1})")

    return labels


# ============================================================
# MAIN FUNCTION
# ============================================================

def create_collage():

    # --------------------------------------------------------
    # Validate configuration
    # --------------------------------------------------------

    if len(IMAGE_PATHS) == 0:
        raise ValueError("IMAGE_PATHS is empty.")

    if COLUMNS < 1:
        raise ValueError("COLUMNS must be at least 1.")

    # --------------------------------------------------------
    # Load images
    # --------------------------------------------------------

    images = [
        load_image(path)
        for path in IMAGE_PATHS
    ]

    number_of_images = len(images)

    # --------------------------------------------------------
    # Determine grid dimensions
    # --------------------------------------------------------

    rows = (
        number_of_images + COLUMNS - 1
    ) // COLUMNS

    # --------------------------------------------------------
    # Determine panel width
    # --------------------------------------------------------

    panel_width = (
        OUTPUT_WIDTH
        - 2 * MARGIN
        - (COLUMNS - 1) * HORIZONTAL_GAP
    ) // COLUMNS

    # --------------------------------------------------------
    # Determine panel height
    # --------------------------------------------------------
    # Calculate a common panel height based on the average
    # aspect ratio of all input images.

    aspect_ratios = [
        image.width / image.height
        for image in images
    ]

    average_aspect_ratio = (
        sum(aspect_ratios)
        / len(aspect_ratios)
    )

    panel_height = int(
        panel_width / average_aspect_ratio
    )

    # --------------------------------------------------------
    # Canvas dimensions
    # --------------------------------------------------------

    output_height = (
        2 * MARGIN
        + rows * panel_height
        + (rows - 1) * VERTICAL_GAP
    )

    canvas = Image.new(
        "RGBA",
        (OUTPUT_WIDTH, output_height),
        BACKGROUND
    )

    draw = ImageDraw.Draw(canvas)

    if SHOW_LABELS:
        label_font = get_font(LABEL_FONT_SIZE)
        labels = generate_panel_labels(number_of_images)

    # --------------------------------------------------------
    # Place images
    # --------------------------------------------------------

    for i, image in enumerate(images):

        row = i // COLUMNS
        col = i % COLUMNS

        # ----------------------------------------------------
        # Standard position
        # ----------------------------------------------------

        x = (
            MARGIN
            + col * (
                panel_width
                + HORIZONTAL_GAP
            )
        )

        y = (
            MARGIN
            + row * (
                panel_height
                + VERTICAL_GAP
            )
        )

        # ----------------------------------------------------
        # Special handling for odd final row
        # ----------------------------------------------------
        # Example:
        #
        # 7 images, 2 columns:
        #
        # A B
        # C D
        # E F
        #   G
        #
        if (
            i == number_of_images - 1
            and number_of_images % COLUMNS != 0
        ):
            remaining = number_of_images % COLUMNS

            # Center the final image in the row
            x = (
                MARGIN
                + (
                    OUTPUT_WIDTH
                    - 2 * MARGIN
                    - panel_width
                ) // 2
            )

        # ----------------------------------------------------
        # Resize image
        # ----------------------------------------------------

        resized = fit_image(
            image,
            panel_width,
            panel_height
        )

        # Center image inside its panel area
        paste_x = (
            x
            + (panel_width - resized.width) // 2
        )

        paste_y = (
            y
            + (panel_height - resized.height) // 2
        )

        # ----------------------------------------------------
        # Paste image
        # ----------------------------------------------------

        canvas.alpha_composite(
            resized,
            (paste_x, paste_y)
        )

        # ----------------------------------------------------
        # Add panel label
        # ----------------------------------------------------

        if SHOW_LABELS:

            draw.text(
                (
                    paste_x + LABEL_MARGIN_X,
                    paste_y + LABEL_MARGIN_Y
                ),
                labels[i],
                font=label_font,
                fill="black"
            )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = Path(OUTPUT_PATH)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Convert RGBA -> RGB
    canvas_rgb = Image.new(
        "RGB",
        canvas.size,
        BACKGROUND
    )

    canvas_rgb.paste(
        canvas,
        mask=canvas.getchannel("A")
    )

    canvas_rgb.save(
        output_path,
        format="PNG",
        dpi=(300, 300),
        optimize=True
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print("Collage created successfully.")
    print(f"Output: {output_path}")
    print(
        f"Images: {number_of_images}"
    )
    print(
        f"Layout: {rows} rows × {COLUMNS} columns"
    )
    print(
        f"Size: {canvas_rgb.width} × {canvas_rgb.height} px"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    create_collage()