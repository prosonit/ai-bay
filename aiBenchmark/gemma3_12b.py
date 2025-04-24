import ollama
import time
import threading
import argparse
import statistics

def benchmark_ollama(model_name, num_requests, num_parallel):
    """
    Benchmarks the Ollama model with varying levels of concurrency.

    Args:
        model_name (str): The name of the Ollama model to benchmark.
        num_requests (int): The number of benchmark requests to run for each concurrency level.
        num_parallel (int): The number of parallel requests to run.
    """

    requests = [
        "What is the capital of France?",
        "Explain the theory of relativity in simple terms.",
        "Write a short poem about a cat."
    ]

    def run_single_request(request):
        start_time = time.time()
        try:
            response = ollama.generate(model=model_name, prompt=request)
            
            # If you are using the dictionary style (most common, especially with pip ollama>=0.1.5)
            if isinstance(response, dict) and "response" in response:
                text = response["response"]
            # Some older or alternate client versions may use attribute access:
            elif hasattr(response, "response"):
                text = response.response
            else:
                raise RuntimeError(f"Unexpected response type: {response}")

            end_time = time.time()
            return end_time - start_time, len(text.strip().split()) # number of tokens
        except Exception as e:
            print(f"Error generating response: {e}")
            return None, 0  # Return None for time, and 0 tokens on error

    def run_parallel_requests(request_list, parallel_count):
        start_time = time.time()
        threads = []
        results = []

        def worker(request):
            result = run_single_request(request)
            results.append(result)

        for request in request_list:
            thread = threading.Thread(target=worker, args=(request,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join() # Wait for all threads to finish.
        
        end_time = time.time()
        total_time = end_time - start_time
        return total_time, results



    print(f"Benchmarking {model_name} with {num_parallel} parallel requests, {num_requests} runs each...")

    # Single request benchmark
    single_times = []
    single_tokens = []
    for _ in range(num_requests):
        time_taken, tokens = run_single_request(requests[_ % len(requests)]) # Cycle through requests
        if time_taken is not None:
            single_times.append(time_taken)
            single_tokens.append(tokens)
        else:
            print("Skipping run due to error.")
        time.sleep(0.1)  # Add a small delay between runs

    if single_times:
        avg_single_time = statistics.mean(single_times)
        avg_single_tokens = statistics.mean(single_tokens)
        tokens_per_second_single = avg_single_tokens / avg_single_time
        print(f"  Single Request: Average Time: {avg_single_time:.4f}s, Tokens/s: {tokens_per_second_single:.2f}")
    else:
        print("  Single Request: No valid runs to calculate.")


    # Parallel requests benchmark
    parallel_times = []
    parallel_tokens = []
    for _ in range(num_requests):
      total_time, results = run_parallel_requests(requests[:num_parallel], num_parallel) # Use the first num_parallel requests
      if total_time is not None:
            parallel_times.append(total_time)
            total_tokens = sum([res[1] for res in results if res is not None]) # Sum all tokens generated
            parallel_tokens.append(total_tokens)
      else:
          print("Skipping run due to error.")
      time.sleep(0.1)

    if parallel_times:
      avg_parallel_time = statistics.mean(parallel_times)
      avg_parallel_tokens = statistics.mean(parallel_tokens)
      tokens_per_second_parallel = avg_parallel_tokens / avg_parallel_time
      print(f"  {num_parallel} Parallel Requests: Average Time: {avg_parallel_time:.4f}s, Total Tokens: {avg_parallel_tokens}, Tokens/s: {tokens_per_second_parallel:.2f}")
    else:
        print(f"  {num_parallel} Parallel Requests: No valid runs to calculate.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Ollama model.")
    parser.add_argument("model_name", type=str, help="The name of the Ollama model to benchmark.")
    parser.add_argument("--num_requests", type=int, default=5, help="The number of benchmark requests to run for each concurrency level.")
    parser.add_argument("--single_only", action="store_true", help="Run only single request benchmark.")

    args = parser.parse_args()
    
    if args.single_only:
        benchmark_ollama(args.model_name, args.num_requests, 1)
    else:
        benchmark_ollama(args.model_name, args.num_requests, 1)
        benchmark_ollama(args.model_name, args.num_requests, 2)
        benchmark_ollama(args.model_name, args.num_requests, 3)
        