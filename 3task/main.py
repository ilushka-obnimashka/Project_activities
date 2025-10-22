import os

import click

from video_processor import NoizeFilterProcessor


@click.command()
@click.argument("input_path", type=str)
@click.option(
    "--num_workers", "-n",
    type=int,
    default=2,
    show_default=True,
    help="The number of flows/processes."
)
def main(input_path: str, num_workers: int):
    try:
        processor = NoizeFilterProcessor(input_path, f"result_{os.path.basename(input_path)}", num_workers)
        processor.run()
    except Exception as e:
        print('12')
        print(e)


if __name__ == '__main__':
    main()
