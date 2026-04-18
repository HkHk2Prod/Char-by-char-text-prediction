import urllib.request
from pathlib import Path

TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DEFUALT_RAW_DIR = Path("data/raw")
DEFAULT_FILENAME = "tiny_sahkespeare.txt"

def download_text(
        url: str = TINY_SHAKESPEARE_URL,
        dest_dir: Path = DEFUALT_RAW_DIR,
        filename: str = DEFAULT_FILENAME,
) -> None:
    """
        Downloads text from url. By default downloads tiny shakespeare
    """

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename

    if dest_path.exists():
        print(f'File already exists: {dest_path}')
        return

    urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, dest_path)

    print('Download successful.')

def main() -> None:
    download_text()

if __name__ == "__main__":
    main()