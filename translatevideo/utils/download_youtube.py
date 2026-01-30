"""
TODO:
ytdlp의 인자들은 -h로 보여줄 수 있는데, 이건 argpase로 구현되고 init의 main 함수에 있음.
마찬가지로 init에서 parse_options라는 함수가 있음. 아래의 해석 함수에서 참고.

"""
import yt_dlp  # type: ignore
import yt_dlp.options  # type: ignore
from typing import Literal


def return_cli_to_api(ydl_opts: str) -> dict:
    create_parser = yt_dlp.options.create_parser

    def parse_patched_options(opts):
        patched_parser = create_parser()
        patched_parser.defaults.update({
            'ignoreerrors': False,
            'retries': 0,
            'fragment_retries': 0,
            'extract_flat': False,
            'concat_playlist': 'never',
            'update_self': False,
        })
        yt_dlp.options.create_parser = lambda: patched_parser
        try:
            return yt_dlp.parse_options(opts)
        finally:
            yt_dlp.options.create_parser = create_parser

    default_opts = parse_patched_options([]).ydl_opts

    def cli_to_api(opts, cli_defaults=False):
        opts = (yt_dlp.parse_options if cli_defaults else parse_patched_options)(
            opts).ydl_opts

        diff = {k: v for k, v in opts.items() if default_opts[k] != v}
        if 'postprocessors' in diff:
            diff['postprocessors'] = [pp for pp in diff['postprocessors']
                                      if pp not in default_opts['postprocessors']]
        return diff

    return cli_to_api(ydl_opts)


def download_youtube(urls: list[str],
                     ydl_opts: dict | str,
                     postprocessors: dict[yt_dlp.postprocessor.PostProcessor,
                                          Literal["pre_process",
                                                  "after_filter",
                                                  "video",
                                                  "before_dl",
                                                  "post_process",
                                                  "after_move",
                                                  "after_video",
                                                  "playlist"]] = None) -> Literal[0, 1]:
    if isinstance(ydl_opts, str):
        ydl_opts = return_cli_to_api(ydl_opts)

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        if postprocessors is not None:
            for pp, when in postprocessors.items():
                ydl.add_post_processor(pp, when)
        error_code: Literal[0, 1] = ydl.download(urls)
    return error_code
