import os

import click

from video_processor import VideoProcessor


@click.command()
@click.argument('input_path', default='0')
@click.argument('num_workers', default=1, type=int)
@click.option('--noise', 'noise_type', type=click.Choice(['salt_pepper', 'gaussian', 'none'], case_sensitive=False),
              default='salt_pepper', help='Type of noise to add.')
@click.option('--salt', 'salt_prob', type=float, default=0.02, help='Salt probability (for salt_pepper noise).')
@click.option('--pepper', 'pepper_prob', type=float, default=0.02, help='Pepper probability (for salt_pepper noise).')
@click.option('--mean', type=float, default=0, help='Mean (for gaussian noise).')
@click.option('--sigma', type=float, default=25, help='Sigma (for gaussian noise).')
def main(input_path, output_path, num_workers, noise_type, salt_prob, pepper_prob, mean, sigma):
    noise_params = {}
    if noise_type == 'salt_pepper':
        noise_params = {'salt_prob': salt_prob, 'pepper_prob': pepper_prob}
    elif noise_type == 'gaussian':
        noise_params = {'mean': mean, 'sigma': sigma}

    if noise_type == 'none':
        noise_type = None

    print(f"Starting with 1 worker process.")
    print(f"Noise type: {noise_type if noise_type else 'None'}")
    if input_path.isdigit():
        print(f"Source: Webcam ({input_path})")
    else:
        print(f"Source: File ({input_path})")
        print(f"Output: File ({output_path})")

    try:
        processor = VideoProcessor(
            input_path=input_path,
            output_path=f"result_{os.path.basename(input_path)}",
            num_workers=num_workers,
            noise_type=noise_type,
            noise_params=noise_params
        )

        processor.run()
    except (IOError, RuntimeError) as e:
        print(f"Error during initialization or run: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == '__main__':
    main()
