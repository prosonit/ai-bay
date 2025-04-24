import concurrent.futures
import logging
from requests import Session

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_request(model_name, context):
    """Generates a single Ollama request for benchmarking."""
    # Construct the request URL based on model
    endpoint = f"{model_name}/v1/api/chat/completions"
    
    # Create sample prompts (can be modified)
    prompts = [
        "Write a poem about AI and its impact on humanity.",
        "Explain quantum computing to a 5-year-old.",
        "What are the benefits of using open-source software?"
    ]
    
    # Generate random context
    context = {"temperature": 0.7, "max_tokens": 1000}
    
    # Create requests for each prompt
    requests = []
    for prompt in prompts:
        requests.append({
            'model': model_name,
            'messages': [{'role': 'user', 'content': prompt}],
            'context': context
        })
    
    return requests

def time_request(request):
    """Time a single Ollama request and log results."""
    try:
        with timing() as timer:
            response = session.post(**request)
            
            # Extract token count from response (assuming Ollama returns JSON with token_count)
            try:
                token_count = response.json()['token_count']
            except:
                logger.error("Failed to extract token_count")
                return 0
            
            elapsed = timer.stop()
            if elapsed == 0:
                logger.warning("Request took zero time, possible error in timing.")
            else:
                logger.info(f"Successfully processed request in {elapsed:.2f}s with {token_count} tokens")
        return (elapsed, token_count)
    except Exception as e:
        logger.error(f"Request failed: {str(e)}")
        return (0, 0)

def run_benchmark(model_name, runs=5):
    """Run benchmark and collect performance metrics."""
    # Prepare output dictionary
    results = {'model': model_name}
    
    # Generate sample requests for timing
    request = generate_request(model_name, None)
    
    # Single request case
    single_times, single_tokens = [], []
    for _ in range(runs):
        elapsed, tokens = time_request(request[0])
        single_times.append(elapsed)
        single_tokens.append(tokens)
        logger.info(f"Single request: {elapsed:.2f}s with {tokens} tokens")
    
    # Calculate single request metrics
    single_avg_time = sum(single_times) / runs
    single_avg_tps = sum(single_tokens) / (single_avg_time * len(request))
    
    results['single'] = {
        'avg_time': single_avg_time,
        'avg_tps': single_avg_tps,
        'variance': max(single_times) - min(single_times)
    }
    
    # Two parallel requests
    if len(request) >= 2:
        parallel_times, parallel_tokens = [], []
        
        with concurrent.futures.as_completed(
            [time_request(req) for req in request[1:3]]
        ):
            for elapsed, tokens in concurrent.futures.resultiter():
                parallel_times.append(elapsed)
                parallel_tokens.append(tokens)
                
        # Calculate parallel metrics
        if len(parallel_times) > 0:
            parallel_avg_time = sum(parallel_times) / runs
            parallel_avg_tps = sum(parallel_tokens) / (parallel_avg_time * len(request))
            
            results['two_parallel'] = {
                'avg_time': parallel_avg_time,
                'avg_tps': parallel_avg_tps,
                'variance': max(parallel_times) - min(parallel_times)
            }
    
    # Print final results
    logger.info(f"Benchmark completed for model: {model_name}")
    if len(results) > 1:
        logger.info(results)

def main():
    """Main function to execute the benchmark."""
    # Process command line arguments
    parser = argparse.ArgumentParser(description='Benchmark Ollama model performance.')
    parser.add_argument('--model', type=str, required=True, help='Name of the model to benchmark')
    parser.add_argument('--runs', type=int, default=5,
                       help='Number of times to run the benchmark [default: 5]')
    
    args = parser.parse_args()
    
    # Run benchmark
    run_benchmark(args.model, runs=args.runs)
    
if __name__ == "__main__":
 #   from timing import timing
    import sys
    session = Session()
    
    main()
