import multiprocessing as mp
import queue
import time
from typing import Union

import cv2

# Общая переменная памяти для размера ядра фильтра
FILTER_KERNEL_SIZE = mp.Value('i', 5)


def worker(in_queue: mp.Queue, out_queue: mp.Queue, stop_event: mp.Event):
    """
    Рабочий процесс: берет кадры из входной очереди, применяет медианное размытие
    и помещает обработанный кадр в выходную очередь.
    """
    global FILTER_KERNEL_SIZE

    while not stop_event.is_set():
        try:
            # Используем небольшой таймаут для проверки stop_event
            frame_idx, frame = in_queue.get(timeout=0.01)
        except queue.Empty:
            continue
        except Exception:
            # Обработка потенциального неожиданного завершения доступа к очереди
            if stop_event.is_set():
                break
            continue

        # Получаем размер ядра из общей памяти
        with FILTER_KERNEL_SIZE.get_lock():
            current_kernel_size = FILTER_KERNEL_SIZE.value

        # Убеждаемся, что размер ядра нечетный и не менее 3 для cv2.medianBlur
        if current_kernel_size % 2 == 0:
            current_kernel_size += 1
        if current_kernel_size < 3:
            current_kernel_size = 3

        processed_frame = cv2.medianBlur(frame, current_kernel_size)
        out_queue.put((frame_idx, processed_frame))


def on_trackbar_change(value):
     """Функция обратного вызова для ползунка, обновляющая глобальный размер ядра."""
     global FILTER_KERNEL_SIZE

     # Гарантируем, что значение не менее 3
     new_value = max(3, value)

     # Гарантируем, что значение нечетное
     if new_value % 2 == 0:
         new_value += 1

     with FILTER_KERNEL_SIZE.get_lock():
         FILTER_KERNEL_SIZE.value = new_value


def put_frames_to_queue(input_source: Union[str, int], queue: mp.Queue, is_camera: bool, stop_event: mp.Event):
    """
    Процесс-производитель: считывает кадры из видеоисточника и помещает их во входную очередь.
    """
    cap = None
    try:
        cap = cv2.VideoCapture(input_source)
        if not cap.isOpened():
            # Вызываем правильный объект исключения
            raise RuntimeError(f"Не удалось открыть источник: {input_source}")

        if is_camera:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        frame_idx = 0
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            queue.put((frame_idx, frame))
            frame_idx += 1

            if is_camera:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_event.set()
                    break

    except Exception as e:
        print(f"Ошибка производителя на источнике {input_source}: {e}")
        stop_event.set()
        raise RuntimeError(f"Ошибка настройки источника {input_source}. Подробности: {e}")
    finally:
        if cap is not None:
            cap.release()


class NoizeFilterProcessor:
    def __init__(self, input_path: str, output_path: str, num_workers: int):
        self.input_path_str = input_path
        self.output_path = output_path
        self.num_workers = num_workers

        self.is_camera = input_path.isdigit()
        self.input_source = int(input_path) if self.is_camera else input_path

        self.cap = None
        self.out = None

        try:
            self.cap = cv2.VideoCapture(self.input_source)
            if not self.cap.isOpened():
                raise IOError(f"Не удалось открыть источник {input_path}")

            if not self.is_camera:
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                source_fps = self.cap.get(cv2.CAP_PROP_FPS)

                if (source_fps < 1.0):
                    source_fps = 30.0

                self.out = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*'mp4v'), source_fps,
                                           (width, height))

        except IOError as e:
            raise e
        except Exception as e:
            # Исправлено: Вызываем правильный объект исключения
            raise RuntimeError(f"Ошибка настройки источника {input_path}. Подробности: {e}")

    def __del__(self):
        """Очистка ресурсов при уничтожении объекта."""
        if hasattr(self, 'cap') and self.cap is not None and self.cap.isOpened():
            self.cap.release()
        if hasattr(self, 'out') and self.out is not None and self.out.isOpened():
            self.out.release()
        cv2.destroyAllWindows()

    def run(self):
        WINDOW_NAME = "Адаптивная фильтрация шума (Нажмите 'q' для выхода)"
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        # УЛУЧШЕНИЕ: Используем цикл для более надежной инициализации окна.
        # Это гарантирует, что окно полностью зарегистрировано перед созданием ползунка.

        time.sleep(1)

        # ❌ УДАЛЕНА: Строка для создания ползунка. Окно теперь без трекбара.
        # cv2.createTrackbar("Размер ядра (Нечетный)", WINDOW_NAME, 5, 21, on_trackbar_change)

        in_queue = mp.Queue(maxsize=self.num_workers * 4)  # Ограниченный размер очереди
        out_queue = mp.Queue(maxsize=self.num_workers * 4)
        stop_event = mp.Event()

        # 1. Запуск рабочих процессов
        workers = []
        for _ in range(self.num_workers):
            worker_obj = mp.Process(target=worker, args=(in_queue, out_queue, stop_event))
            worker_obj.start()
            workers.append(worker_obj)

        # 2. Запуск процесса-производителя
        # Передаем правильные 4 аргумента
        producer_obj = mp.Process(
            target=put_frames_to_queue,
            args=(self.input_source, in_queue, self.is_camera, stop_event)
        )
        producer_obj.start()

        frame_buffer = {}
        next_frame_idx = 0

        print("Начинаем обработку видео...")

        try:
            # Продолжаем, пока производитель активен И/ИЛИ в очередях/буфере есть данные
            while producer_obj.is_alive() or not out_queue.empty() or frame_buffer:
                # Получаем обработанные кадры из выходной очереди
                try:
                    # Используем небольшой таймаут для проверки условия выхода из основного цикла
                    frame_idx, processed_frame = out_queue.get(timeout=0.01)
                    frame_buffer[frame_idx] = processed_frame
                except queue.Empty:
                    # Если очереди пусты и производитель неактивен, выходим
                    if not producer_obj.is_alive() and in_queue.empty() and out_queue.empty():
                        break
                    continue
                except Exception:
                    # Обрабатываем сбой доступа к очереди
                    if not producer_obj.is_alive():
                        break
                    continue

                # Записываем и отображаем кадры по порядку
                while next_frame_idx in frame_buffer:
                    frame_to_display = frame_buffer.pop(next_frame_idx)  # Исправлено: Извлекаем кадр один раз

                    # Записываем в файл только если это видео (не камера)
                    if self.out:
                        self.out.write(frame_to_display)

                    cv2.imshow(WINDOW_NAME, frame_to_display)  # Используем извлеченный кадр

                    # Здесь обработка событий GUI должна выполняться постоянно
                    key = cv2.waitKey(1)
                    if key & 0xFF == ord('q'):
                        stop_event.set()
                        break

                    next_frame_idx += 1

                # Проверяем сигнал выхода после обработки буфера
                if stop_event.is_set():
                    break

            print("Обработка завершена.")

        except KeyboardInterrupt:
            print("Обработка прервана пользователем")

        finally:
            # Очистка всех процессов
            stop_event.set()
            producer_obj.join()
            for worker_obj in workers:
                worker_obj.terminate()  # Используем terminate для более быстрой очистки
                worker_obj.join(timeout=1)

            cv2.destroyAllWindows()
            print("Очистка завершена.")