import multiprocessing as mp
import queue
import time
from typing import Union, Optional, Dict, Any, Callable

import cv2
import numpy as np

# Глобальное значение для размера ядра, разделяемое между процессами
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
    noisy[coords_y_salt, coords_x_salt] = 255  # 255 для белого

    num_pepper = int(np.ceil(pepper_prob * height * width))
    coords_y_pepper = np.random.randint(0, height, num_pepper)
    coords_x_pepper = np.random.randint(0, width, num_pepper)
    noisy[coords_y_pepper, coords_x_pepper] = 0  # 0 для черного

    return noisy


def add_gaussian_noise(image: np.ndarray, mean: float = 0, sigma: float = 25) -> np.ndarray:
    """Добавляет гауссовский шум на изображение.

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


def blur_worker(in_queue: mp.Queue, out_queue: mp.Queue, stop_event: mp.Event):
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

        # Убедимся, что размер ядра нечетный и >= 3
        if current_kernel_size < 3:
            current_kernel_size = 3
        elif current_kernel_size % 2 == 0:
            current_kernel_size += 1

        processed_frame = cv2.medianBlur(frame, current_kernel_size)
        out_queue.put((frame_idx, processed_frame))


def on_trackbar_change(value):
    """Функция обратного вызова для ползунка, обновляющая глобальный размер ядра."""
    global FILTER_KERNEL_SIZE
    # Гарантируем, что значение нечетное и >= 3
    new_value = max(3, value)
    if new_value % 2 == 0:
        new_value += 1
    
    with FILTER_KERNEL_SIZE.get_lock():
        FILTER_KERNEL_SIZE.value = new_value
    
    # Обновляем позицию ползунка, если мы ее скорректировали
    cv2.setTrackbarPos("Kernel Size (Odd)", "Adaptive Noise Filtering (Press 'q' to exit)", new_value)


def put_frames_to_queue(input_source: Union[str, int],
                        queue: mp.Queue,
                        is_camera: bool,
                        stop_event: mp.Event,
                        noise_type: Optional[str],
                        noise_params: Dict[str, Any]):
    """
    Процесс-производитель по умолчанию: считывает кадры, добавляет выбранный шум
    и помещает их во входную очередь.
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
                print("Producer: End of source.")
                break # Конец файла или ошибка чтения

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

            try:
                # Помещаем в очередь с таймаутом, чтобы не блокироваться навечно, если очередь полна
                queue.put((frame_idx, noisy_frame), timeout=0.5)
                frame_idx += 1
            except queue.Full:
                if stop_event.is_set():
                    break
                continue

            if is_camera:
                # В режиме камеры задержка не нужна, т.к. cap.read() блокирует
                pass
            else:
                # Для видеофайлов можно добавить небольшую задержку,
                # чтобы не переполнять очередь слишком быстро
                time.sleep(0.001) 

    except Exception as e:
        print(f"Producer error on source {input_source}: {e}")
    finally:
        if cap is not None:
            cap.release()
        stop_event.set() # Сигнализируем всем, что продюсер закончил
        print("Producer process finished.")


class VideoProcessor:
    """
    Универсальный класс для обработки видеопотока с применением различных инструментов.
    Функции продюсера и воркера задаются при инициализации.
    Поддерживает многопроцессорность (multiprocessing).
    """

    def __init__(self, input_path: str,
                 output_path: str,
                 worker_function: Callable,
                 producer_function: Callable = put_frames_to_queue, # <-- ИЗМЕНЕНИЕ
                 num_workers: int = 1,
                 noise_type: Optional[str] = None,
                 noise_params: Optional[Dict[str, Any]] = None):

        self.input_path_str = input_path
        self.output_path = output_path
        self.noise_type = noise_type
        self.noise_params = noise_params if noise_params is not None else {}

        self.worker_func = worker_function
        self.producer_func = producer_function # <-- ИЗМЕНЕНИЕ
        self.num_workers = max(1, num_workers) # Гарантируем хотя бы 1 воркера

        self.is_camera = input_path.isdigit()
        self.input_source = int(input_path) if self.is_camera else input_path

        self.out = None
        self.workers_list = []

        try:
            temp_cap = cv2.VideoCapture(self.input_source)
            if not temp_cap.isOpened():
                raise IOError(f"Failed to open source {input_path}")

            if not self.is_camera:
                width = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                source_fps = temp_cap.get(cv2.CAP_PROP_FPS)

                if source_fps < 1.0 or source_fps > 200.0:
                    print(f"Warning: Invalid source FPS {source_fps}, defaulting to 30.0")
                    source_fps = 30.0

                self.out = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*'mp4v'), source_fps,
                                           (width, height))
                if not self.out.isOpened():
                    raise IOError(f"Failed to open output VideoWriter for {self.output_path}")

            temp_cap.release()

        except IOError as e:
            raise e
        except Exception as e:
            raise RuntimeError(f"Error setting up source {input_path}. Details: {e}")


    def run(self):
        WINDOW_NAME = "Adaptive Noise Filtering (Press 'q' to exit)"
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.waitKey(1)

        initial_kernel_size = 5
        with FILTER_KERNEL_SIZE.get_lock():
             FILTER_KERNEL_SIZE.value = initial_kernel_size
             
        cv2.createTrackbar("Kernel Size (Odd)", WINDOW_NAME, initial_kernel_size, 21, on_trackbar_change)
        cv2.waitKey(1)

        queue_size = max(10, self.num_workers * 4)
        in_queue = mp.Queue(maxsize=queue_size)
        out_queue = mp.Queue(maxsize=queue_size)
        stop_event = mp.Event()

        self.workers_list = []
        for _ in range(self.num_workers):
            worker_obj = mp.Process(
                target=self.worker_func,
                args=(in_queue, out_queue, stop_event),
                daemon=True 
            )
            worker_obj.start()
            self.workers_list.append(worker_obj)

        producer_obj = mp.Process(
            target=self.producer_func, 
            args=(self.input_source, in_queue, self.is_camera, stop_event, self.noise_type, self.noise_params),
            daemon=True # Делаем продюсера демоном
        )
        producer_obj.start()

        frame_buffer = {}
        next_frame_idx = 0
        last_frame_processed = False

        print(f"Starting video processing with {self.num_workers} worker(s)... Press 'q' to exit.")

        try:
            while not stop_event.is_set():
                
                # Проверяем, жив ли продюсер. Если нет, и очереди пусты, выходим.
                if not producer_obj.is_alive() and in_queue.empty() and out_queue.empty() and not frame_buffer:
                    if not last_frame_processed:
                        print("Producer finished and all queues are empty.")
                        last_frame_processed = True
                    # Даем небольшой таймаут, чтобы убедиться, что все воркеры точно закончили
                    time.sleep(0.1) 
                    if out_queue.empty() and not frame_buffer:
                         break # Выходим из основного цикла

                try:
                    # Пытаемся получить обработанный кадр из выходной очереди
                    frame_idx, processed_frame = out_queue.get(timeout=0.01)
                    frame_buffer[frame_idx] = processed_frame
                except queue.Empty:
                    # Если очередь пуста, просто продолжаем цикл
                    pass
                except Exception as e:
                    if stop_event.is_set():
                        break
                    print(f"Error getting from out_queue: {e}")
                    continue

                # Обрабатываем кадры из буфера в правильном порядке
                while next_frame_idx in frame_buffer:
                    frame_to_display = frame_buffer.pop(next_frame_idx)

                    if self.out:
                        self.out.write(frame_to_display)

                    cv2.imshow(WINDOW_NAME, frame_to_display)

                    key = cv2.waitKey(1)
                    if key & 0xFF == ord('q'):
                        print("'q' pressed, stopping...")
                        stop_event.set()
                        break

                    next_frame_idx += 1
                
                if last_frame_processed and not frame_buffer:
                    break # Все кадры показаны

            if not stop_event.is_set():
                 print("Processing finished naturally.")

        except KeyboardInterrupt:
            print("Processing interrupted by user (Ctrl+C)")
            stop_event.set()

        finally:
            print("Starting cleanup...")
            stop_event.set() # Устанавливаем событие остановки для всех процессов

            # Даем процессам время, чтобы завершиться штатно
            print("Joining producer process...")
            producer_obj.join(timeout=1.0) 
            if producer_obj.is_alive():
                print("Producer process still alive, terminating...")
                producer_obj.terminate() # Принудительно завершаем, если join не сработал
                producer_obj.join() # Ждем завершения
                print("Producer process terminated.")

            for idx, worker_obj in enumerate(self.workers_list):
                print(f"Joining worker process {idx}...")
                worker_obj.join(timeout=1.0)
                if worker_obj.is_alive():
                    print(f"Worker process {idx} still alive, terminating...")
                    worker_obj.terminate()
                    worker_obj.join()
                    print(f"Worker process {idx} terminated.")
            
            
            print("Starting to clear in_queue...")
            while not in_queue.empty():
                try:
                    in_queue.get_nowait()
                except queue.Empty:
                    break
            in_queue.close()
            in_queue.join_thread()
            print("In_queue cleared and closed.")

            print("Starting to clear out_queue...")
            while not out_queue.empty():
                try:
                    out_queue.get_nowait()
                except queue.Empty:
                    break
            out_queue.close()
            out_queue.join_thread()
            print("Out_queue cleared and closed.")

            if self.out:
                print("Releasing video output...")
                self.out.release()
                print(f"Video saved to {self.output_path}")

            cv2.destroyAllWindows()
            for _ in range(5):
                cv2.waitKey(1)
            print("Cleanup finished.")

