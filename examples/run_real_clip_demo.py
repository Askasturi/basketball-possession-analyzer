
"""Run the demo pipeline on a real uploaded clip with mock overlay detections."""

from pathlib import Path

from run_demo import build_demo_pipeline


def main() -> None:
    """Run the existing demo pipeline on a real clip."""
    input_path = Path("data/clips/demo_clip_1.mp4")
    output_path = Path("outputs/demo_clip_1_annotated.mp4")

    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing input clip: {input_path}. "
            "Place your demo clip at data/clips/demo_clip_1.mp4"
        )

    pipeline = build_demo_pipeline()
    results = pipeline.analyze_video(
        input_path=input_path,
        output_path=output_path,
        start_frame=0,
        end_frame=5,
    )

    print(f"Analyzed {len(results)} frames from {input_path}")
    print(f"Wrote annotated video to {output_path}")


if __name__ == "__main__":
    main()
