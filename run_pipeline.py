from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.imagerie.dl_pipeline import train_evaluate_dl
from src.imagerie.ml_pipeline import build_feature_table, train_evaluate_ml
from src.imagerie.tfds_pipeline import load_tfds_pipeline

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plant disease detection/classification pipeline (TFDS)")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory")
    parser.add_argument("--run-dl", action="store_true", help="Run optional deep learning extension")
    parser.add_argument("--dl-epochs", type=int, default=8, help="Number of epochs for DL")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for TFDS")
    # Kept max-per-class only for taking subset of TFDS if desired, else ignore
    parser.add_argument("--max-samples", type=int, default=1000, help="Max images to take from TFDS for ML to avoid memory issues")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*50)
    print("🌱 AgriVision Pipeline Initialization 🌱")
    print("="*50)

    # 1. Load dataset from TFDS
    print("\n📦 [PHASE 1/4] Loading PlantVillage Dataset (TFDS)")
    print("--------------------------------------------------")
    train_ds, info = load_tfds_pipeline(
        batch_size=args.batch_size,
        image_size=(224, 224), # 224x224 if DL runs, ML is invariant to this basically
        split="train"
    )
    
    class_names = info.features['label'].names
    print(f"   ✓ Successfully loaded TFDS dataset stream.")
    print(f"   ✓ Total disease classes available: {len(class_names)}")
    if args.max_samples > 0:
         print(f"   ✓ Subsetting to max {args.max_samples} samples for Classical ML processing.")

    if args.max_samples > 0:
        # For ML pipeline, unbatch and take subset to avoid running out of RAM
        ml_ds = train_ds.unbatch().take(args.max_samples).batch(args.batch_size)
    else:
        ml_ds = train_ds

    # 2. Extract ML Features
    print("\n🔍 [PHASE 2/4] Classical Feature Extraction (Color & GLCM Texture)")
    print("----------------------------------------------------------------")
    X, y = build_feature_table(
        tfds_dataset=ml_ds,
        class_names=class_names,
        out_dir=out_dir
    )

    if len(X) == 0:
        raise ValueError("No valid images found.")

    print(f"   ✓ Extracted numerical feature arrays for {len(X)} images.")

    # 3. Train ML Models
    print("\n🧠 [PHASE 3/4] Training Classical ML Models (SVM & Random Forest)")
    print("-----------------------------------------------------------------")
    ml_summary = train_evaluate_ml(X, y, out_dir)

    summary = {
        "selected_classes": class_names,
        "num_samples": int(len(X)),
        "ml": ml_summary,
    }

    # 4. Optional DL Training
    if args.run_dl:
        print("\n🤖 [PHASE 4/4] Deep Learning Pipeline (MobileNetV2)")
        print("-------------------------------------------------")
        # Since TFDS 'plantvillage' only affords 'train', we'll manually split info if needed
        # Or dl_pipeline can handle split. We'll pass the base dataset.
        dl_summary = train_evaluate_dl(
            tfds_dataset=train_ds,
            class_names=class_names,
            out_dir=out_dir / "dl",
            image_size=(224, 224),
            epochs=args.dl_epochs,
        )
        summary["dl"] = dl_summary
    else:
        print("\n⏭️  [PHASE 4/4] Deep Learning Pipeline Skipped (No --run-dl flag)")

    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    
    print("\n✅ PIPELINE COMPLETE")
    print("==================================================")
    print("Run Summary:")
    print(json.dumps(summary, indent=2))
    print("==================================================")
    print(f"All artifacts saved to: {out_dir.absolute()}")
    print("You can now run 'streamlit run app.py' to explore the results!")


if __name__ == "__main__":
    main()
