from icrawler.builtin import BingImageCrawler
from PIL import Image
import os

storage_dir = 'traffic cone'
target_size = (512, 512)

google_crawler = BingImageCrawler(storage={'root_dir': storage_dir}, downloader_threads = 4)
google_crawler.crawl(keyword='traffic cone on road', max_num=600)

for filename in os.listdir(storage_dir):
    filepath = os.path.join(storage_dir, filename)
    try:
        with Image.open(filepath) as img:
            img = img.convert('RGB')  # avoids issues with png transparency / palette modes
            img = img.resize(target_size)
            img.save(filepath)
    except Exception as e:
        print(f"Skipped {filename}: {e}")
