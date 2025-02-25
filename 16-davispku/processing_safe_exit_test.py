from multiprocessing import Process, Event, Queue
import time
import keyboard

def process_events(stop_event, queue, TIME_SLEEP):
    while not stop_event.is_set():
        print("Processing events...")
        time.sleep(TIME_SLEEP)

    print("Process exiting safely.")

if __name__ == "__main__":
    queue = Queue(maxsize=10)
    stop_event = Event()

    def on_press_callback(event):
        if event.name == 'a':
            print('You pressed the A key')
        if event.name == 'esc':
            stop_event.set()
            print('You pressed the Q key, exiting...')
    keyboard.on_press(on_press_callback)

    p = Process(target=process_events, args=(stop_event, queue, 0.1))
    p.start()

    p.join()  # 等待子进程结束