#%%
"""
Here's an example of how you can modify the worker function to log the prompt that caused the error:
"""
from multiprocessing import Process, Queue
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(15), wait=wait_exponential(multiplier=2, max=128))
def dummy_api_call():
    return 32

def worker():
    try:
        # Perform some computation
        result = dummy_api_call()  # Replace with your actual computation
        # Get the shared queue
        queue = shared_queue.get()
        # Put the result in the queue
        queue.put(result)
    except Exception as e:
        # file_with_args_caused_error.write(arg)  # write args to file that cause errors for later calling
        print('Some arg cased an error, fix implementation to print it')

if __name__ == '__main__':
    # Create a shared queue
    shared_queue = multiprocessing.Manager().Queue()

    # Create and start multiple worker processes
    processes = []
    for _ in range(4):  # Adjust the number of processes as needed
        p = Process(target=worker)
        processes.append(p)
        p.start()

    # Wait for all processes to finish
    for p in processes:
        p.join()

    # Retrieve and print all results from the queue
    while not shared_queue.empty():
        result = shared_queue.get()
        print(f'Result: {result}')

#%%
"""
Yes, there is a way to implement the .start() and .join() method with a Queue without passing the Queue object to each worker process. You can use a shared memory object provided by the multiprocessing module, which allows you to create a Queue that is accessible to all processes without explicitly passing it as an argument.
"""
from multiprocessing import Process, Queue

def worker():
    # Perform some computation
    result = 42  # Replace with your actual computation

    # Get the shared queue
    queue = shared_queue.get()

    # Put the result in the queue
    queue.put(result)

if __name__ == '__main__':
    # Create a shared queue
    shared_queue = multiprocessing.Manager().Queue()

    # Create and start multiple worker processes
    processes = []
    for _ in range(4):  # Adjust the number of processes as needed
        p = Process(target=worker)
        processes.append(p)
        p.start()

    # Wait for all processes to finish
    for p in processes:
        p.join()

    # Retrieve and print all results from the queue
    while not shared_queue.empty():
        result = shared_queue.get()
        print(f'Result: {result}')

#%%
"""
To ensure that all processes run in parallel with futures, you can use the concurrent.futures.as_completed function. This function returns an iterator that yields the Future objects as they complete, allowing you to process the results as they become available without blocking.

Here's an example of how to use as_completed to run processes in parallel with futures:
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

def worker(value):
    # Perform some computation
    result = value * 2  # Replace with your actual computation
    return result

if __name__ == '__main__':
    # Create a process pool
    with ProcessPoolExecutor(max_workers=4) as executor:
        # Submit tasks to the pool
        futures = [executor.submit(worker, value) for value in range(1, 5)]

        # Retrieve and print results as they become available
        for future in as_completed(futures):
            result = future.result()
            print(f'Result: {result}')
#%%
"""
Using starmap to run parallel procs.
"""
from tqdm import tqdm
num_workers = 4
args1, args2 = [1, 2], [-1, -2]
f = lambda a, b : a*b

pool = Pool(num_workers)
tasks = []
for task_idx, (arg1, arg2) in enumerate(zip(args1, args2)): 
    task_args = (arg1, arg2) 
    tasks.append(task_args)
print(f'Num Tasks: {len(tasks)=}')
with tqdm(total=len(tasks)) as progress_bar:
    results = pool.starmap(f, tqdm(tasks, total=len(tasks)))
pool.close()  # Prevent any new tasks from being submitted to the pool
pool.join()   # Wait for all the worker processes to finish their tasks and exit
