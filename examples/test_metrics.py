#!/usr/bin/env python3
"""
Prometheus metrics example for auto_ai_router
Fetches and displays metrics from the /metrics endpoint
"""

import requests

def main():
    metrics_url = "http://localhost:8080/metrics"

    print("Fetching Prometheus metrics...")

    try:
        response = requests.get(metrics_url, timeout=5)

        if response.status_code == 200:
            print("\n=== Prometheus Metrics ===\n")

            # Filter and display only auto_ai_router metrics
            for line in response.text.split('\n'):
                if 'auto_ai_router' in line and not line.startswith('#'):
                    print(line)

            print("\n✓ Metrics retrieved successfully")
        else:
            print(f"✗ Error: Status code {response.status_code}")

    except requests.exceptions.ConnectionError:
        print("✗ Error: Could not connect to router. Is it running?")
        print("Make sure prometheus_enabled is set to true in config.yaml")
    except Exception as e:
        print(f"✗ Error: {e}")

if __name__ == "__main__":
    main()
