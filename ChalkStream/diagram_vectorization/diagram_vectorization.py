"""
Board Stroke Vectorization — Module v0.2
Steps implemented (so far):
  1) Skeletonize a binary "ink mask" (1=ink, 0=background) using Zhang–Suen thinning
  2) Trace the skeleton into polylines (graph edges between endpoints/junctions)
  2.5) Refine line processing (first pass):
       - Simplify traced polylines with Ramer–Douglas–Peucker (RDP)
       - Detect "nearly straight" polylines, fit a line (PCA), and angle-snap to 0°/90° (optional diagonal snap later)
       - Convert those into clean line segments for downstream "box should be a box" behavior

Also:
  - Exhaustive comments throughout
  - Show all figures at once (non-blocking show, then one final blocking show)
  - Timing measurements printed to terminal for each stage + total runtime

IMPORTANT:
  - This expects you already have a *binary foreground mask* (board rectified + foreground extracted).
  - For now, the demo uses a synthetic toy mask. Replace `mask = make_toy_mask()` with your real mask.
"""

import time
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import cv2


# -----------------------------
# Utility: high-resolution timing helper
# -----------------------------
class Timer:
    """
    Small helper to time stages and print readable timing results.
    Usage:
        t = Timer()
        t.mark("stage name")  # prints time since last mark
    """
    def __init__(self):
        self.t0 = time.perf_counter()
        self.last = self.t0

    def mark(self, label: str):
        now = time.perf_counter()
        dt = now - self.last
        total = now - self.t0
        print(f"[TIMING] {label:30s}  stage={dt*1000:9.2f} ms   total={total*1000:9.2f} ms")
        self.last = now

def ink_mask_from_roi_bgr(roi_bgr: np.ndarray) -> np.ndarray:
    """
    Convert a region crop (BGR) into a binary ink mask (0/1).
    This is meant for orchestrator integration (no disk I/O).
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    th = cv2.adaptiveThreshold(
        gray_blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        5
    )

    white_ratio = float(np.mean(th == 255))
    if white_ratio > 0.5:
        ink = (th == 0)
    else:
        ink = (th == 255)

    return ink.astype(np.uint8)  # 0/1


def polylines_to_diagram_content(polylines_yx):
    """
    Convert traced polylines into JSON-safe, region-local coordinates.
    """
    out = []

    for path in polylines_yx:
        poly = []
        for (y, x) in path:
            poly.append([float(x), float(y)])
        out.append(poly)

    return out

import cv2
import numpy as np

def mask_from_array(img_bgr: np.ndarray) -> np.ndarray:
    """
    Convert a cropped BGR image into a binary ink mask.
    Output:
      mask[y,x] = 1 for ink
      mask[y,x] = 0 for background
    """

    # 1) Convert to grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 2) Light blur to reduce noise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # 3) Adaptive threshold (robust to lighting)
    th = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        5
    )

    # 4) Invert so ink = 1
    # Heuristic: boards are mostly white
    if np.mean(th == 255) > 0.5:
        mask = (th == 0)
    else:
        mask = (th == 255)

    return mask.astype(np.uint8)



def load_mask_from_image(path: str) -> np.ndarray:
    """
    Loads an input image (jpg/png) and converts it into a *binary ink mask*:
      - mask[y,x] = 1 means "ink/stroke"
      - mask[y,x] = 0 means "background"

    This is a TEMPORARY stand-in for your real foreground extraction.
    You will likely replace this with your own extraction output later.

    Heuristic:
      - Convert to grayscale
      - Light blur to reduce sensor noise
      - Adaptive threshold to handle uneven lighting
      - Invert if needed so strokes become 1s

    Returns:
      - mask: uint8 array with values 0/1
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find file: {path}")

    # Read image as grayscale (0..255)
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Failed to load image (is it a valid jpg/png?): {path}")

    # Light blur to reduce noise before thresholding
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Adaptive threshold: good for uneven lighting (common on boards)
    # Output is 0 or 255.
    th = cv2.adaptiveThreshold(
        gray_blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,   # blockSize (must be odd)
        5     # C (tweak up/down depending on contrast)
    )
    

    # Decide whether to invert:
    # For most whiteboards, strokes are darker -> threshold gives strokes as 0.
    # We want strokes as 1, so invert when needed.
    # Heuristic: if most pixels are white (255), it's likely background=255 and strokes=0 -> invert to make strokes=1.
    white_ratio = float(np.mean(th == 255))

    if white_ratio > 0.5:
        # likely background is white; strokes are black (0)
        ink = (th == 0)
    else:
        # likely background is black; strokes are white (255)
        ink = (th == 255)

    mask = ink.astype(np.uint8)  # 0/1
    return mask



def skeletonize_fast_opencv_contrib(mask: np.ndarray) -> np.ndarray:
    """
    Fast skeletonization using OpenCV contrib thinning.
    Input:
      - mask: 2D array, ink pixels are nonzero
    Output:
      - skel: 2D uint8 array with values 0/1
    """
    import cv2

    # OpenCV thinning expects 0/255 uint8
    img_255 = ((mask > 0).astype(np.uint8) * 255)

    # Thinning is in cv2.ximgproc (opencv-contrib)
    # THINNING_ZHANGSUEN or THINNING_GUOHALL are both available in many builds.
    skel_255 = cv2.ximgproc.thinning(img_255, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

    # Convert back to 0/1
    return (skel_255 > 0).astype(np.uint8)


def skeletonize_fast_skimage(mask: np.ndarray) -> np.ndarray:
    """
    Fast skeletonization using scikit-image.
    Input:
      - mask: 2D array, ink pixels are nonzero
    Output:
      - skel: 2D uint8 array with values 0/1
    """
    from skimage.morphology import skeletonize

    # skeletonize() expects a boolean image where True = foreground
    skel_bool = skeletonize(mask > 0)

    # Convert to uint8 0/1 to match the rest of your pipeline
    return skel_bool.astype(np.uint8)


# -----------------------------
# Step 1: Skeletonization (Zhang–Suen thinning)
# -----------------------------
def zhang_suen_thinning(bin_img: np.ndarray) -> np.ndarray:
    """
    Zhang–Suen thinning algorithm.
    Input:
      - bin_img: 2D array with values 0/1 (or 0/255, any nonzero treated as 1)
    Output:
      - skeleton: 2D array with values 0/1 where strokes are 1px wide centerlines

    Notes:
      - This is a classic thinning method widely used to skeletonize binary shapes.
      - It iteratively removes boundary pixels while preserving connectivity.
    """
    # Ensure binary 0/1
    img = (bin_img > 0).astype(np.uint8)

    # Pad image with 0s to avoid boundary checks everywhere
    img = np.pad(img, ((1, 1), (1, 1)), mode="constant")

    def neighbors(y, x):
        """
        Returns the 8-neighborhood values P2..P9 in clockwise order:
            P9 P2 P3
            P8 P1 P4
            P7 P6 P5
        where P1 is the center pixel (y,x).
        """
        return [
            img[y - 1, x],     # P2
            img[y - 1, x + 1], # P3
            img[y, x + 1],     # P4
            img[y + 1, x + 1], # P5
            img[y + 1, x],     # P6
            img[y + 1, x - 1], # P7
            img[y, x - 1],     # P8
            img[y - 1, x - 1], # P9
        ]

    def transitions(nb):
        """
        Counts the number of 0->1 transitions in the ordered neighbor sequence:
        P2, P3, ..., P9, back to P2.

        For thinning, we require exactly 1 transition to ensure "simple point" removal
        that does not break connectivity.
        """
        seq = nb + [nb[0]]
        return sum((seq[i] == 0 and seq[i + 1] == 1) for i in range(8))

    changed = True
    while changed:
        changed = False

        # -------------------------
        # Sub-iteration 1:
        # mark pixels to remove based on Zhang–Suen conditions
        # -------------------------
        to_remove = []
        ys, xs = np.where(img == 1)
        for y, x in zip(ys, xs):
            # skip padded boundary pixels
            if y == 0 or x == 0 or y == img.shape[0] - 1 or x == img.shape[1] - 1:
                continue

            nb = neighbors(y, x)
            B = sum(nb)                 # number of nonzero neighbors
            A = transitions(nb)         # number of 0->1 transitions
            P2, P3, P4, P5, P6, P7, P8, P9 = nb

            # Zhang–Suen removal conditions for sub-iter 1:
            #  - 2 <= B <= 6
            #  - A == 1
            #  - P2 * P4 * P6 == 0
            #  - P4 * P6 * P8 == 0
            if (2 <= B <= 6 and A == 1 and
                P2 * P4 * P6 == 0 and
                P4 * P6 * P8 == 0):
                to_remove.append((y, x))

        if to_remove:
            for y, x in to_remove:
                img[y, x] = 0
            changed = True

        # -------------------------
        # Sub-iteration 2:
        # slightly different constraints to preserve topology
        # -------------------------
        to_remove = []
        ys, xs = np.where(img == 1)
        for y, x in zip(ys, xs):
            if y == 0 or x == 0 or y == img.shape[0] - 1 or x == img.shape[1] - 1:
                continue

            nb = neighbors(y, x)
            B = sum(nb)
            A = transitions(nb)
            P2, P3, P4, P5, P6, P7, P8, P9 = nb

            # Zhang–Suen conditions for sub-iter 2:
            #  - 2 <= B <= 6
            #  - A == 1
            #  - P2 * P4 * P8 == 0
            #  - P2 * P6 * P8 == 0
            if (2 <= B <= 6 and A == 1 and
                P2 * P4 * P8 == 0 and
                P2 * P6 * P8 == 0):
                to_remove.append((y, x))

        if to_remove:
            for y, x in to_remove:
                img[y, x] = 0
            changed = True

    # Unpad back to original shape
    return img[1:-1, 1:-1]


def prune_isolated_pixels(skel: np.ndarray, max_iters: int = 50) -> np.ndarray:
    """
    Minimal cleanup pass:
      - Removes truly isolated skeleton pixels (degree == 0).
    This reduces 'speck' artifacts that can create tiny polylines later.

    NOTE: This is NOT full spur pruning. We'll improve spur pruning later
          when we start caring more about diagram correctness/cleanliness.
    """
    out = (skel > 0).astype(np.uint8)
    H, W = out.shape
    neigh = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    for _ in range(max_iters):
        ys, xs = np.where(out == 1)
        remove = []
        for y, x in zip(ys, xs):
            deg = 0
            for dy, dx in neigh:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and out[ny, nx] == 1:
                    deg += 1
            if deg == 0:
                remove.append((y, x))
        if not remove:
            break
        for y, x in remove:
            out[y, x] = 0

    return out


# -----------------------------
# Step 2: Trace skeleton pixels into polylines (graph edges)
# -----------------------------
NEIGH8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

def pixel_degrees(skel: np.ndarray) -> np.ndarray:
    """
    Computes the 8-neighborhood degree (number of neighboring skeleton pixels) for each skeleton pixel.
    Degree meaning:
      - 0: isolated speck
      - 1: endpoint
      - 2: normal interior pixel on a path
      - >=3: junction (branch point)
    """
    sk = (skel > 0).astype(np.uint8)
    H, W = sk.shape
    deg = np.zeros_like(sk, dtype=np.uint8)

    ys, xs = np.where(sk == 1)
    for y, x in zip(ys, xs):
        d = 0
        for dy, dx in NEIGH8:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and sk[ny, nx] == 1:
                d += 1
        deg[y, x] = d

    return deg


def trace_polylines(skel: np.ndarray):
    """
    Converts a skeleton (0/1 image) into polylines by walking graph edges.

    Output:
      polylines: list of paths, each path is a list of (y,x) integer pixel coordinates
      nodes: set of (y,x) nodes (endpoints + junctions + isolated pixels)
    """
    sk = (skel > 0).astype(np.uint8)
    H, W = sk.shape

    deg = pixel_degrees(sk)

    # Nodes = all skeleton pixels whose degree != 2
    # (degree 2 pixels are just interior path pixels; degree !=2 pixels are endpoints/junctions)
    nodes = set(zip(*np.where((sk == 1) & (deg != 2))))

    def nbrs(p):
        """Return 8-connected skeleton neighbors of pixel p=(y,x)."""
        y, x = p
        out = []
        for dy, dx in NEIGH8:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and sk[ny, nx] == 1:
                out.append((ny, nx))
        return out

    visited_edges = set()  # stores directed edges (a,b) to avoid duplicating traces
    polylines = []

    # Trace all edges that start at a node and go to a neighbor.
    # Each trace continues through degree-2 pixels until reaching another node.
    for n in list(nodes):
        for nxt in nbrs(n):
            # Skip if this edge already traced (in either direction)
            if (n, nxt) in visited_edges or (nxt, n) in visited_edges:
                continue

            path = [n]
            prev = n
            cur = nxt
            visited_edges.add((prev, cur))

            while True:
                path.append(cur)

                # Stop condition: reached a node (endpoint or junction)
                if cur in nodes:
                    break

                # Continue condition: degree==2 => choose the neighbor that's not prev
                ns = nbrs(cur)
                if not ns:
                    break  # dead end (rare)

                if len(ns) == 1:
                    nxt2 = ns[0]
                else:
                    # Pick neighbor that isn't prev (assumes degree 2 interior)
                    nxt2 = ns[0] if ns[1] == prev else ns[1]

                prev, cur = cur, nxt2
                visited_edges.add((prev, cur))

            polylines.append(path)

    # Special case: handle pure cycles where every pixel has degree==2 (so there are no nodes).
    # Those won't be traced by the node-start logic above, so we detect leftover pixels and trace a loop.
    all_pixels = set(zip(*np.where(sk == 1)))
    used_pixels = set()
    for pl in polylines:
        used_pixels.update(pl)

    remaining = {p for p in (all_pixels - used_pixels) if deg[p] == 2}

    while remaining:
        start = next(iter(remaining))
        path = [start]
        prev = None
        cur = start

        while True:
            ns = nbrs(cur)
            if len(ns) < 2:
                break

            # If prev is None, pick either neighbor to get going; otherwise take the "other" neighbor
            if prev is None:
                nxt = ns[0]
            else:
                nxt = ns[0] if ns[1] == prev else ns[1]

            prev, cur = cur, nxt

            if cur == start:
                break  # loop closed

            path.append(cur)

            # Safety break (shouldn't happen, but prevents infinite loops)
            if len(path) > H * W:
                break

        for p in path:
            remaining.discard(p)

        polylines.append(path)

    return polylines, nodes


# -----------------------------
# Step 2.5: Refine line processing
#   - Simplify polylines with RDP
#   - Detect nearly-straight polylines, line-fit + angle-snap to 0/90
# -----------------------------
def point_line_distance(p, a, b) -> float:
    """
    Perpendicular distance from point p to line segment (in R^2).
    p, a, b are 2D points as (x,y) floats.

    Used by RDP simplification.
    """
    px, py = p
    ax, ay = a
    bx, by = b
    dx = bx - ax
    dy = by - ay

    # If a and b are the same point, return distance to that point
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)

    # Project point p onto the infinite line through a->b, using parameter t
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)

    # For distance to the infinite line (not segment), we do not clamp t.
    projx = ax + t * dx
    projy = ay + t * dy
    return math.hypot(px - projx, py - projy)


def rdp_simplify(points_xy: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Ramer–Douglas–Peucker polyline simplification.
    Input:
      - points_xy: Nx2 array (x,y)
      - epsilon: max deviation allowed (in pixels) before keeping a point
    Output:
      - simplified Nx2 array (x,y)

    This removes jitter and reduces point count while preserving corners.
    """
    if len(points_xy) <= 2:
        return points_xy

    a = points_xy[0]
    b = points_xy[-1]

    # Find the point farthest from the baseline a->b
    max_dist = -1.0
    idx = -1
    for i in range(1, len(points_xy) - 1):
        d = point_line_distance(points_xy[i], a, b)
        if d > max_dist:
            max_dist = d
            idx = i

    # If the farthest point exceeds epsilon, keep it and recurse; otherwise keep endpoints only
    if max_dist > epsilon:
        left = rdp_simplify(points_xy[: idx + 1], epsilon)
        right = rdp_simplify(points_xy[idx:], epsilon)
        return np.vstack([left[:-1], right])  # avoid duplicating the split point
    else:
        return np.vstack([a, b])


def polyline_length(points_xy: np.ndarray) -> float:
    """Sum of Euclidean distances along the polyline."""
    if len(points_xy) < 2:
        return 0.0
    diffs = points_xy[1:] - points_xy[:-1]
    return float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))


def fit_line_pca(points_xy: np.ndarray):
    """
    Fits a line direction using PCA.
    Returns:
      - centroid (cx,cy)
      - direction unit vector (dx,dy) (principal component)
    """
    pts = points_xy.astype(np.float64)
    c = pts.mean(axis=0)
    X = pts - c

    # Covariance matrix and eigen decomposition
    cov = (X.T @ X) / max(len(pts), 1)
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Principal direction = eigenvector with largest eigenvalue
    v = eigvecs[:, np.argmax(eigvals)]
    v = v / (np.linalg.norm(v) + 1e-12)
    return c, v


def angle_of_vec(v: np.ndarray) -> float:
    """Angle in degrees of vector v=(dx,dy) with respect to +x axis."""
    return math.degrees(math.atan2(v[1], v[0]))


def snap_angle_to_axis(angle_deg: float, tol_deg: float = 10.0):
    """
    Snap angle to nearest axis (0° or 90°) if within tolerance.
    Returns:
      snapped_angle_deg (or None if no snap)
      label ("H" or "V" or None)
    """
    # Normalize angle to [-90, 90] for easier axis comparison
    a = angle_deg
    while a <= -90:
        a += 180
    while a > 90:
        a -= 180

    # Compare to 0° (horizontal) and 90° (vertical)
    if abs(a - 0) <= tol_deg:
        return 0.0, "H"
    if abs(abs(a) - 90) <= tol_deg:
        return 90.0, "V"
    return None, None


def project_points_onto_line(points_xy: np.ndarray, c: np.ndarray, v_unit: np.ndarray):
    """
    Projects points onto line defined by centroid c and direction v_unit.
    Returns:
      t values for each point such that projection = c + t*v_unit
    """
    X = points_xy - c
    t = X @ v_unit
    return t


def snap_polyline_to_line_segment(points_xy: np.ndarray, tol_deg: float = 10.0):
    """
    Attempts to convert a polyline to a clean snapped line segment.

    Process:
      1) Fit a line direction with PCA
      2) If its angle is within tolerance of horizontal or vertical, snap it
      3) Convert polyline into a single segment spanning min->max along that direction
    Output:
      - segment as ((x1,y1),(x2,y2)) if snapped, else None
      - snap label ("H"/"V") if snapped, else None
    """
    c, v = fit_line_pca(points_xy)
    ang = angle_of_vec(v)

    snapped_ang, label = snap_angle_to_axis(ang, tol_deg=tol_deg)
    if label is None:
        return None, None

    # Force snapped direction unit vector exactly horizontal or vertical
    if label == "H":
        v_snap = np.array([1.0, 0.0], dtype=np.float64)
    else:
        v_snap = np.array([0.0, 1.0], dtype=np.float64)

    # Project points to get extents along the snapped direction
    t = project_points_onto_line(points_xy, c, v_snap)
    tmin, tmax = float(np.min(t)), float(np.max(t))

    p1 = c + tmin * v_snap
    p2 = c + tmax * v_snap

    return (tuple(p1.tolist()), tuple(p2.tolist())), label


def simplify_and_snap_polylines(polylines_yx, rdp_epsilon_px=2.0, min_len_px=20.0, snap_tol_deg=10.0):
    """
    Takes raw traced polylines (list of (y,x) integer pixels) and produces:
      - simplified polylines (float xy arrays) for visualization/storage
      - snapped line segments for those that are long enough and nearly axis-aligned

    Parameters:
      - rdp_epsilon_px: controls how aggressively jitter is removed
      - min_len_px: minimum simplified polyline length before we even try to snap
      - snap_tol_deg: angular tolerance for snapping to horizontal/vertical

    Returns:
      simplified_polylines_xy: list[np.ndarray Nx2]  (x,y points)
      snapped_segments: list[((x1,y1),(x2,y2), label)]
    """
    simplified = []
    snapped_segments = []

    for path_yx in polylines_yx:
        # Convert (y,x) pixel coords into float (x,y) for geometry ops and plotting
        pts_xy = np.array([(p[1], p[0]) for p in path_yx], dtype=np.float64)

        # Simplify
        pts_s = rdp_simplify(pts_xy, epsilon=rdp_epsilon_px)

        simplified.append(pts_s)

        # Skip snapping if too short (this also helps avoid snapping noise specks)
        if polyline_length(pts_s) < min_len_px:
            continue

        seg, label = snap_polyline_to_line_segment(pts_s, tol_deg=snap_tol_deg)
        if seg is not None:
            (x1, y1), (x2, y2) = seg
            snapped_segments.append(((x1, y1), (x2, y2), label))

    return simplified, snapped_segments


# -----------------------------
# Demo / Visualization Helpers
# -----------------------------
def make_toy_mask(H=220, W=320) -> np.ndarray:
    """
    Creates a synthetic binary mask (ink=1) with two rectangles and a connector line.
    This is ONLY for demo/testing.

    Replace this by loading your real extracted foreground mask.
    """
    mask = np.zeros((H, W), dtype=np.uint8)

    def draw_rect(img, x, y, w, h, t=6):
        # Draw a thick rectangle border by painting strips
        img[y:y+t, x:x+w] = 1
        img[y+h-t:y+h, x:x+w] = 1
        img[y:y+h, x:x+t] = 1
        img[y:y+h, x+w-t:x+w] = 1

    def draw_line(img, x0, y0, x1, y1, t=5):
        # Draw a thick line by sampling points and painting a small square at each point
        n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
        xs = np.linspace(x0, x1, n).astype(int)
        ys = np.linspace(y0, y1, n).astype(int)
        for x, y in zip(xs, ys):
            x0b, x1b = max(0, x - t//2), min(W, x + t//2 + 1)
            y0b, y1b = max(0, y - t//2), min(H, y + t//2 + 1)
            img[y0b:y1b, x0b:x1b] = 1

    draw_rect(mask, x=60, y=55, w=120, h=90, t=7)
    draw_line(mask, 180, 100, 260, 160, t=6)
    draw_rect(mask, x=235, y=145, w=70, h=50, t=6)

    # Add tiny random speck noise
    rng = np.random.default_rng(0)
    noise_pts = rng.integers([0, 0], [H, W], size=(90, 2))
    mask[noise_pts[:, 0], noise_pts[:, 1]] = (rng.random(90) < 0.12).astype(np.uint8)

    return mask


def show_all_figures_nonblocking():
    """
    We show each figure with block=False so you can see multiple windows at once.
    Then we call a final plt.show() (blocking) at the end of main to keep them open.

    NOTE: Some environments (e.g., notebooks) may handle windows differently,
          but this is the best "show all at once" approach in standard matplotlib.
    """
    plt.show(block=False)


# -----------------------------
# Main
# -----------------------------
def main():
    timer = Timer()

    # Requirements for mask:
    #   - 2D NumPy array (H,W)
    #   - ink pixels are >0 (we convert to 0/1 internally)
    #
    # 
    # If you run: python script.py input.jpg
    # we load that image. Otherwise we fall back to the toy demo.
    if len(sys.argv) >= 2:
        input_path = sys.argv[1]

        # Resolve path relative to the script location
        if not os.path.isabs(input_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            input_path = os.path.join(script_dir, input_path)

        mask = load_mask_from_image(input_path)
    else:
        mask = make_toy_mask()
    timer.mark("load/create mask")

    # -----------------------------
    # Step 1: Skeletonize
    # -----------------------------
    #skel = zhang_suen_thinning(mask)
    skel = skeletonize_fast_skimage(mask)
    #skel = skeletonize_fast_opencv_contrib(mask)

    timer.mark("step1: thinning")

    skel = prune_isolated_pixels(skel)
    timer.mark("step1: prune isolated")

    # -----------------------------
    # Step 2: Trace skeleton to polylines
    # -----------------------------
    polylines, nodes = trace_polylines(skel)
    timer.mark("step2: trace polylines")

    # -----------------------------
    # Step 2.5: Simplify + snap lines (first refinement pass)
    # -----------------------------
    simplified_polylines_xy, snapped_segments = simplify_and_snap_polylines(
        polylines_yx=polylines,
        rdp_epsilon_px=2.0,   # tweak: 1.0-3.0 typical
        min_len_px=20.0,      # tweak: raise to ignore more small fragments
        snap_tol_deg=10.0     # tweak: 5-15 degrees typical
    )
    timer.mark("step2.5: simplify + snap")

    # -----------------------------
    # Terminal summary
    # -----------------------------
    print("\n[SUMMARY]")
    print(f"Foreground pixels:       {int((mask > 0).sum())}")
    print(f"Skeleton pixels:         {int((skel > 0).sum())}")
    print(f"Node pixels:             {len(nodes)}")
    print(f"Raw polylines traced:    {len(polylines)}")
    print(f"Simplified polylines:    {len(simplified_polylines_xy)}")
    print(f"Snapped line segments:   {len(snapped_segments)}")
    timer.mark("summary printed")

    # -----------------------------
    # Visualization
    # Show all figures "at once" (multiple windows), without needing to close one to see next.
    # We do:
    #   plt.figure(); ...; plt.show(block=False)
    # for each figure, then final plt.show() at end to keep open.
    # -----------------------------

    # Figure 1: Input mask
    fig1 = plt.figure()
    plt.title("Input foreground mask (binary)")
    plt.imshow(mask, cmap="gray")
    plt.axis("off")
    show_all_figures_nonblocking()

    # Figure 2: Skeleton
    fig2 = plt.figure()
    plt.title("Step 1: skeleton (centerlines)")
    plt.imshow(skel, cmap="gray")
    plt.axis("off")
    show_all_figures_nonblocking()

    # Figure 3: Raw traced polylines overlay
    fig3 = plt.figure()
    plt.title(f"Step 2: traced polylines (raw) — count={len(polylines)}")
    plt.imshow(skel, cmap="gray")
    for path in polylines:
        ys = [p[0] for p in path]
        xs = [p[1] for p in path]
        plt.plot(xs, ys, linewidth=2)  # default matplotlib cycle colors
    plt.axis("off")
    show_all_figures_nonblocking()

    # Figure 4: Simplified polylines + snapped segments overlay
    fig4 = plt.figure()
    plt.title(f"Step 2.5: simplified polylines + snapped (H/V) segments — snapped={len(snapped_segments)}")
    plt.imshow(skel, cmap="gray")

    # Plot simplified polylines
    for pts in simplified_polylines_xy:
        plt.plot(pts[:, 0], pts[:, 1], linewidth=2)

    # Plot snapped segments thicker so you can see the 'clean' line replacements
    for (x1, y1), (x2, y2), label in snapped_segments:
        plt.plot([x1, x2], [y1, y2], linewidth=4)

    plt.axis("off")
    show_all_figures_nonblocking()

    timer.mark("figures created")

    # Final blocking show keeps all open.
    plt.show()
    timer.mark("plt.show() returned")


if __name__ == "__main__":
    main()
