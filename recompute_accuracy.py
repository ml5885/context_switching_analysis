import json
import argparse
from collections import defaultdict

def recompute_from_debug(debug_file_path):
    """
    Loads a _debug.json file, recomputes the accuracy for each experimental
    condition, and prints the results.
    """
    try:
        with open(debug_file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {debug_file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {debug_file_path}")
        return

    print(f"--- Recomputing Accuracy for: {debug_file_path} ---")
    
    target_task = data.get("target_task", "N/A")
    print(f"Target Task: {target_task}\n")

    # Group predictions by configuration
    # Key: (history_len, history_content_task)
    # Value: {'correct': count, 'total': count}
    results = defaultdict(lambda: defaultdict(int))

    for ex in data.get("debug_examples", []):
        try:
            prediction = ex["model_prediction"]
            expected = ex["expected_answer"]
            config = ex["config"]
            
            h_len = config["history_len"]
            h_content = config["history_content_task"]
            
            key = (h_len, h_content)
            
            if prediction == expected:
                results[key]['correct'] += 1
            results[key]['total'] += 1
        except KeyError as e:
            print(f"Warning: Skipping an example due to missing key: {e}")
            continue
            
    if not results:
        print("No valid examples found to recompute.")
        return

    # Print the recomputed results
    print(f"{'History Len':<12} | {'History Content':<20} | {'Correct':<8} | {'Total':<8} | {'Recomputed Accuracy':<20}")
    print("-" * 80)

    # Sort keys for consistent output
    sorted_keys = sorted(results.keys())

    for key in sorted_keys:
        h_len, h_content = key
        stats = results[key]
        correct = stats['correct']
        total = stats['total']
        
        accuracy = (correct / total) if total > 0 else 0
        
        print(f"{h_len:<12} | {h_content:<20} | {correct:<8} | {total:<8} | {accuracy:<20.4f}")

    print("-" * 80)
    print("Verification complete.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Recompute accuracy from a debug JSON file."
    )
    parser.add_argument(
        "debug_file",
        type=str,
        help="Path to the _debug.json file to process."
    )
    args = parser.parse_args()
    
    recompute_from_debug(args.debug_file)

if __name__ == "__main__":
    main()
