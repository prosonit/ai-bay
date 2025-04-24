import time
import ollama
import threading
import concurrent.futures

def benchmark_ollama(model_name, num_requests):
    """
    Benchmarks Ollama model with single, double, and triple parallel requests.

    Args:
        model_name (str): The name of the Ollama model to benchmark.
        num_requests (int): The number of requests to run in parallel.
    """

    print(f"Benchmarking model: {model_name} with {num_requests} parallel requests.")

    def run_request(prompt):
        """Runs a single Ollama request and measures its time."""
        start_time = time.time()
        try:
            response = ollama.generate(model_name, prompt)
            end_time = time.time()
            response_time = end_time - start_time
            return response_time, response_time
        except Exception as e:
            print(f"Error running request: {e}")
            return None, None

    # 1. Single Request
    print("\n--- Single Request ---")
    response_time_single, _ = run_request("Write a short poem about a cat.")
    if response_time_single is not None:
        tokens_per_second_single = 1 / response_time_single if response_time_single > 0 else 0
        print(f"Token per second: {tokens_per_second_single:.2f}")
    else:
        print("Failed to collect single request data.")

    # 2. Double Parallel Requests
    print("\n--- Double Parallel Requests ---")
    def run_double_parallel(prompts):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(run_request, prompt) for prompt in prompts]
            results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error in double parallel: {e}")

            return results

    prompts_double = ["Tell me a joke.", "Explain the concept of quantum entanglement."]
    results_double = run_double_parallel(prompts_double)
    
    tokens_per_second_double = 0
    if results_double:
        for result in results_double:
            if result[0] is not None:
                tokens_per_second_double += 1 / result[0]
    
    if tokens_per_second_double > 0:
        print(f"Token per second (Double): {tokens_per_second_double:.2f}")
    else:
        print("Failed to collect double request data.")
    

    # 3. Triple Parallel Requests
    print("\n--- Triple Parallel Requests ---")
    def run_triple_parallel(prompts):
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(run_request, prompt) for prompt in prompts]
            results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error in triple parallel: {e}")
            return results

    prompts_triple = ["Write a haiku.", "Translate 'Hello, world!' into Spanish.", "Summarize the plot of Hamlet."]
    results_triple = run_triple_parallel(prompts_triple)

    tokens_per_second_triple = 0
    if results_triple:
        for result in results_triple:
            if result[0] is not None:
                tokens_per_second_triple += 1 / result[0]

    if tokens_per_second_triple > 0:
        print(f"Token per second (Triple): {tokens_per_second_triple:.2f}")
    else:
        print("Failed to collect triple request data.")



if __name__ == "__main__":
    # Example Usage
    model_name = input("Enter Ollama model name: ")
    num_requests = 2  # You can change this to 1, 2, or 3 for different levels of parallelism

    benchmark_ollama(model_name, num_requests)