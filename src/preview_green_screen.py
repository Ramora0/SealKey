"""Interactive green screen preview — press SPACE for a new one, Q/ESC to quit."""

import argparse

import cv2

from src.green_screen import generate_green_screen


def main():
    parser = argparse.ArgumentParser(description="Preview procedural green screens")
    parser.add_argument("--size", type=int, default=768, help="Image size (square)")
    parser.add_argument(
        "--profile",
        choices=["clean", "moderate", "messy"],
        default=None,
        help="Defect profile (default: random mix)",
    )
    args = parser.parse_args()

    window = "Green Screen Preview — SPACE: new  |  Q/ESC: quit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, args.size, args.size)

    seed = 0

    def show_new():
        nonlocal seed
        img = generate_green_screen(
            height=args.size, width=args.size, seed=seed, profile=args.profile
        )
        profile_label = args.profile or "mixed"
        print(f"[seed={seed}] profile={profile_label}")
        cv2.imshow(window, img)
        seed += 1

    show_new()

    try:
        while True:
            # Poll with a short timeout so Ctrl+C (KeyboardInterrupt) fires promptly.
            key = cv2.waitKey(50) & 0xFF
            if key == 255:
                continue
            if key in (ord("q"), 27):  # q or ESC
                break
            if key == ord(" "):
                show_new()
            # Window closed via OS close button
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
