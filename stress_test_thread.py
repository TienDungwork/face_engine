import concurrent.futures
import time
import requests
import statistics
from typing import Dict
import json

with open("base64.txt", "r") as f:
    base64_image = f.read()


def single_request() -> Dict:
    """Make a single face detection request"""
    url = "http://localhost:42075/api/v1/searchFace"
    payload = {
        "company_ids": [],
        "threshold": 0.7,
        "quality": 0.3,
        "num_result": 1,
        "img_base64": base64_image
    }

    start_time = time.time()
    try:
        response = requests.post(url, json=payload)
        response_json = response.json()
        elapsed_time = time.time() - start_time
        response_json['elapsed_time'] = elapsed_time
        response_json['status_code'] = response.status_code
        # if response_json.get('data', {}):
        #     print(response_json)
        return response_json
    except Exception as e:
        return {
            'status_code': 500,
            'error': str(e),
            'elapsed_time': time.time() - start_time
        }


def run_concurrent_requests(num_requests: int = 1000, max_workers: int = 50):
    """Run multiple requests concurrently using thread pool"""
    print(f"Starting stress test with {num_requests} requests...")
    start_time = time.time()

    responses = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all requests to the thread pool
        future_to_request = {executor.submit(
            single_request): i for i in range(num_requests)}

        # Process requests as they complete
        for future in concurrent.futures.as_completed(future_to_request):
            request_num = future_to_request[future]
            try:
                response = future.result()
                responses.append(response)
            except Exception as e:
                responses.append({
                    'status_code': 500,
                    'error': str(e),
                    'elapsed_time': 0
                })

    # Calculate statistics
    total_time = time.time() - start_time
    success_count = sum(1 for r in responses if r.get('status_code') == 200)
    error_count = num_requests - success_count

    # Calculate timing statistics
    request_times = [r['elapsed_time']
                     for r in responses if r.get('status_code') == 200]
    avg_time = statistics.mean(request_times) if request_times else 0
    max_time = max(request_times) if request_times else 0
    min_time = min(request_times) if request_times else 0

    # Log failed responses
    with open("error_log.txt", "w") as f:
        for i, response in enumerate(responses):
            if response.get('status_code') != 200:
                f.write(f"Request {i} failed:\n")
                f.write(f"{json.dumps(response, indent=2)}\n\n")

    print("\nTest Results:")
    print(f"Total requests: {num_requests}")
    print(f"Successful requests: {success_count}")
    print(f"Failed requests: {error_count}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Requests per second: {num_requests/total_time:.2f}")
    print(f"\nTiming Statistics:")
    print(f"Average request time: {avg_time:.3f} seconds")
    print(f"Maximum request time: {max_time:.3f} seconds")
    print(f"Minimum request time: {min_time:.3f} seconds")
    if error_count > 0:
        print(f"Failed responses have been logged to error_log.txt")


if __name__ == "__main__":
    run_concurrent_requests(num_requests=50)
