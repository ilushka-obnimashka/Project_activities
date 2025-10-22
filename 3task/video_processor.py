import multiprocessing as mp
import queue
import time
from typing import Union, Optional, Dict, Any
import cv2
import numpy as np

# Общая переменная памяти для размера ядра фильтра
FILTER_KERNEL_SIZE = mp.Value('i', 5)


def add_salt_and_pepper_noise(image: np.ndarray, salt_prob: float, pepper_prob: float) -> np.ndarray:
    """Добавляет шум типа соль-перец на изображение. (Корректная версия)

    Args:
        image (np.ndarray): Исходное изображение (предполагается 8-битное).
        salt_prob (float): Вероятность появления соли (белые пиксели).
        pepper_prob (float): Вероятность появления перца (черные пиксели).

    Returns:
        np.ndarray: Зашумленное изображение.
    """
    noisy = np.copy(image)
    height, width = image.shape[:2]
    
    num_salt = int(np.ceil(salt_prob * height * width))
    coords_y_salt = np.random.randint(0, height, num_salt)
    coords_x_salt = np.random.randint(0, width, num_salt)
    noisy[coords_y_salt, coords_x_salt] = 255 # 255 для белого

    num_pepper = int(np.ceil(pepper_prob * height * width))
    coords_y_pepper = np.random.randint(0, height, num_pepper)
    coords_x_pepper = np.random.randint(0, width, num_pepper)
    noisy[coords_y_pepper, coords_x_pepper] = 0 # 0 для черного

    return noisy


def add_gaussian_noise(image: np.ndarray, mean: float = 0, sigma: float = 25) -> np.ndarray:
    """Добавляет гауссовский шум на изображение. (Исправленная версия)

    Args:
        image (np.ndarray): Исходное изображение.
        mean (float): Среднее значение шума.
        sigma (float): Стандартное отклонение шума.

    Returns:
        np.ndarray: Зашумленное изображение.
    """
    noise = np.random.normal(mean, sigma, image.shape).astype('uint8')
    noisy_image = cv2.add(image, noise)
    return noisy_image


def worker(in_queue: mp.Queue, out_queue: mp.Queue, stop_event: mp.Event):
    """
    Рабочий процесс: берет зашумленные кадры из входной очереди, применяет медианное размытие
    (удаление шума) и помещает обработанный кадр в выходную очередь.
    """
    global FILTER_KERNEL_SIZE

    while not stop_event.is_set():
        try:
            frame_idx, frame = in_queue.get(timeout=0.01)
        except queue.Empty:
            continue
        except Exception:
            if stop_event.is_set():
                break
            continue

        with FILTER_KERNEL_SIZE.get_lock():
            current_kernel_size = FILTER_KERNEL_SIZE.value

        processed_frame = cv2.medianBlur(frame, current_kernel_size)
        out_queue.put((frame_idx, processed_frame))


def on_trackbar_change(value):
    """Функция обратного вызова для ползунка, обновляющая глобальный размер ядра."""
    global FILTER_KERNEL_SIZE
    new_value = max(3, value)
    if new_value % 2 == 0:
        new_value += 1
    with FILTER_KERNEL_SIZE.get_lock():
        FILTER_KERNEL_SIZE.value = new_value


def put_frames_to_queue(input_source: Union[str, int], 
                        queue: mp.Queue, 
                        is_camera: bool, 
                        stop_event: mp.Event, 
                        noise_type: Optional[str], 
                        noise_params: Dict[str, Any]):
    """
    Процесс-производитель: считывает кадры,
    ДОБАВЛЯЕТ ВЫБРАННЫЙ ШУМ и помещает их во входную очередь.
    """
    cap = None
    try:
        cap = cv2.VideoCapture(input_source)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open source: {input_source}")

        if is_camera:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        frame_idx = 0
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            if noise_type == 'salt_pepper':
                noisy_frame = add_salt_and_pepper_noise(
                    frame, 
                    noise_params.get('salt_prob', 0.01), 
                    noise_params.get('pepper_prob', 0.01)
                )
            elif noise_type == 'gaussian':
                noisy_frame = add_gaussian_noise(
                    frame, 
                    noise_params.get('mean', 0), 
                    noise_params.get('sigma', 25)
                )
            else:
                noisy_frame = frame
            
            queue.put((frame_idx, noisy_frame))
            frame_idx += 1

            if is_camera:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_event.set()
                    break

    except Exception as e:
        print(f"Producer error on source {input_source}: {e}")
        stop_event.set()
    finally:
        if cap is not None:
            cap.release()
        print("Producer process finished.")


class NoiseFilterProcessor:
    def __init__(self, input_path: str, 
                 output_path: str, 
                 noise_type: Optional[str] = None, 
                 noise_params: Optional[Dict[str, Any]] = None):
        
        self.input_path_str = input_path
        self.output_path = output_path
        self.noise_type = noise_type
        self.noise_params = noise_params if noise_params is not None else {}

        self.is_camera = input_path.isdigit()
        self.input_source = int(input_path) if self.is_camera else input_path

        self.cap = None
        self.out = None

        try:
            temp_cap = cv2.VideoCapture(self.input_source)
            if not temp_cap.isOpened():
                raise IOError(f"Failed to open source {input_path}")

            if not self.is_camera:
                width = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                source_fps = temp_cap.get(cv2.CAP_PROP_FPS)

                if (source_fps < 1.0):
                    source_fps = 30.0

                self.out = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*'mp4v'), source_fps,
                                           (width, height))
            
            temp_cap.release()

        except IOError as e:
            raise e
        except Exception as e:
            raise RuntimeError(f"Error setting up source {input_path}. Details: {e}")

    def __del__(self):
        if hasattr(self, 'out') and self.out is not None and self.out.isOpened():
            self.out.release()
        cv2.destroyAllWindows()

    def run(self):
        WINDOW_NAME = "Adaptive Noise Filtering (Press 'q' to exit)"
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.waitKey(100) 
        time.sleep(0.5)

        cv2.createTrackbar("Kernel Size (Odd)", WINDOW_NAME, 5, 21, on_trackbar_change)

        in_queue = mp.Queue(maxsize=self.num_workers * 4)
        out_queue = mp.Queue(maxsize=self.num_workers * 4)
        stop_event = mp.Event()

        worker_obj = mp.Process(target=worker, args=(in_queue, out_queue, stop_event))
        worker_obj.start()
          
        producer_obj = mp.Process(
            target=put_frames_to_queue,
            args=(self.input_source, in_queue, self.is_camera, stop_event, self.noise_type, self.noise_params)
        )
        producer_obj.start()

        frame_buffer = {}
        next_frame_idx = 0

        print("Starting video processing...")

        try:
            while (producer_obj.is_alive() or not out_queue.empty() or frame_buffer):
                try:
                    frame_idx, processed_frame = out_queue.get(timeout=0.01)
                    frame_buffer[frame_idx] = processed_frame
                except queue.Empty:
                    if not producer_obj.is_alive() and in_queue.empty() and out_queue.empty() and not frame_buffer:
                        break
                    continue
                except Exception:
                    if not producer_obj.is_alive():
                        break
                    continue

                while next_frame_idx in frame_buffer:
                    frame_to_display = frame_buffer.pop(next_frame_idx)  

                    if self.out:
                        self.out.write(frame_to_display)

                    cv2.imshow(WINDOW_NAME, frame_to_display)  

                    key = cv2.waitKey(1)
                    if key & 0xFF == ord('q'):
                        stop_event.set()
                        break

                    next_frame_idx += 1

                if stop_event.is_set():
                    break
            
            print("Processing finished.")

        except KeyboardInterrupt:
            print("Processing interrupted by user")
            stop_event.set()

        finally:
            print("Starting cleanup...")
            stop_event.set()
            
            producer_obj.join(timeout=2)
            if producer_obj.is_alive():
                producer_obj.terminate()

            worker_obj.join(timeout=2)
            if worker_obj.is_alive():
                worker_obj.terminate()

            while not in_queue.empty():
                try: in_queue.get_nowait()
                except queue.Empty: break
            while not out_queue.empty():
                try: out_queue.get_nowait()
                except queue.Empty: break
            
            if self.out:
                self.out.release()
                print(f"Video saved to {self.output_path}")

            cv2.destroyAllWindows()
             print("Cleanup finished.")


