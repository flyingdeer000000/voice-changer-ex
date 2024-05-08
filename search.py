import os.path
import random

from musicdl import musicdl
from musicdl.modules import Downloader
from pydub import AudioSegment
from yt_dlp import YoutubeDL
import yt_dlp
from yt_dlp.utils import download_range_func
import json


def is_integer(string):
    if string.isdigit():
        return int(string)
    else:
        return 0


def is_numeric(string):
    if string.isdigit():
        return True
    if string.count('.') == 1:
        integer_part, decimal_part = string.split('.')
        if integer_part.isdigit() and decimal_part.isdigit():
            return True
    return False


def time_to_seconds(time_string):
    hours, minutes, seconds = map(lambda x: is_integer(x), time_string.split(':'))
    total_seconds = hours * 3600 + minutes * 60 + seconds
    return total_seconds


def size_to_int(size_string):
    prefix_size_str = size_string[:-2]  # 去除最后的单位部分，转换为浮点数
    if not is_numeric(prefix_size_str):
        return 5.1 * 1024 * 1024
    unit = size_string[-2:]  # 获取单位部分
    size = float(prefix_size_str)
    if unit == 'KB':
        size *= 1024  # 转换为字节
    elif unit == 'MB':
        size *= 1024 * 1024
    elif unit == 'GB':
        size *= 1024 * 1024 * 1024
    elif unit == 'TB':
        size *= 1024 * 1024 * 1024 * 1024

    return int(size)  # 转换为整数


def search_youtube(keywords):
    YDL_OPTIONS = {
        'format': 'bestaudio',
        # 'noplaylist': 'True',
        # 'proxy': 'http://127.0.0.1:8889',
    }
    with YoutubeDL(YDL_OPTIONS) as ydl:
        video = ydl.extract_info(f"ytsearch:{keywords}", download=False)['entries'][0:5]
        # video = ydl.extract_info(keywords, download=False)
    if len(video) > 0:
        ret = random.choice(video)
        return ydl.sanitize_info(ret)
    else:
        return None


def download_youtube(info, save_path):
    url = info['original_url']
    duration = info['duration']


    start_second = 0
    end_second = duration
    
    ydl_opts = {
        'format': 'm4a/bestaudio/best',
        'downloader': 'ffmpeg',
        'download_ranges': download_range_func(None, [(start_second, end_second)]),
        # ℹ️ See help(yt_dlp.postprocessor) for a list of available Postprocessors and their arguments
        'postprocessors': [{  # Extract audio using ffmpeg
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
        }],
        'outtmpl': save_path,
        # 'proxy': 'http://127.0.0.1:8889',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        # ℹ️ ydl.sanitize_info makes the info json-serializable
        ret_info = ydl.sanitize_info(info)
        ret_info['save_path'] = save_path
        return ret_info


def get_youtube(keywords, save_path):
    info = search_youtube(keywords)
    if info is None:
        return
    else:
        download_youtube(info, save_path)


def get_albums(keywords, config):
    target_srcs = [
        'kugou', 'kuwo', 'qqmusic', 'qianqian', 'fivesing',
        'netease', 'migu', 'joox', 'yiting',
    ]
    client = musicdl.musicdl(config=config)
    results = client.search(keywords, target_srcs)
    albums_set = set()
    valid_albums = []
    for albums in results.values():
        if len(albums) == 0:
            continue
        for album in albums:
            if album['songname'] in albums_set:
                continue
            if album['ext'] != 'mp3':
                continue
            if size_to_int(album['filesize']) > 5 * 1024 * 1024:
                continue
            if time_to_seconds(album['duration']) > 300:
                continue
            else:
                albums_set.add(album['songname'])
                valid_albums.append(album)
    return valid_albums


def get_random_spit(songinfo, save_path):
    d = Downloader(songinfo)
    d.start()
    song = AudioSegment.from_mp3(save_path)
    # pydub does things in milliseconds
    length = len(song)
    left_idx = length / 2 - 15 * 1000
    right_idx = length / 2 + 15 * 1000
    if left_idx < 0:
        left_idx = 0
    if right_idx > length:
        right_idx = length
    middle_30s = song[left_idx:right_idx]
    middle_30s.export(save_path, format="wav")
    return save_path


def download_random(keywords, config, save_path):
    albums = get_albums(keywords, config)
    if len(albums) == 0:
        return None
    album = random.choice(albums)
    get_random_spit(album, save_path=save_path)


if __name__ == '__main__':
    # config = {'logfilepath': 'musicdl.log', 'downloaded': 'downloaded', 'search_size_per_source': 5, 'proxies': {}}
    # infos = get_albums('李荣浩', config)
    # print(infos)
    info = search_youtube('李荣浩 模特')
    download_youtube(info, "downloaded/模特")
