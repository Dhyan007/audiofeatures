from pathlib import Path

import pandas as pd

from .extractor import extract_features


def extract_dataset(
    audio_folder: str,
    output_csv: str | None = None,
    window_size: float = 0.5
) -> pd.DataFrame:

    audio_folder = Path(audio_folder)

    if not audio_folder.exists():
        raise FileNotFoundError(f"{audio_folder} does not exist.")

    if not audio_folder.is_dir():
        raise NotADirectoryError(f"{audio_folder} is not a directory.")

    audio_extensions = {
        ".wav",
        ".mp3",
        ".flac",
        ".ogg",
        ".m4a"
    }

    all_features = []

    for audio_file in sorted(audio_folder.iterdir()):

        if audio_file.suffix.lower() not in audio_extensions:
            continue

        try:

            df = extract_features(
                audio_path=str(audio_file),
                window_size=window_size
            )

            df["song_id"] = audio_file.stem
            df["filename"] = audio_file.name

            all_features.append(df)

        except Exception as e:

            print(f"Skipping {audio_file.name}: {e}")

    if not all_features:
        return pd.DataFrame()

    dataset = pd.concat(
        all_features,
        ignore_index=True
    )

    if output_csv is not None:
        Path(output_csv).parent.mkdir(
            parents=True,
            exist_ok=True
        )
        dataset.to_csv(
            output_csv,
            index=False
        )

    return dataset