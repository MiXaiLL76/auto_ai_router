#!/usr/bin/env python3
"""
Health check example for auto_ai_router
Tests the /health endpoint
"""

import requests
import json

def main():
    health_url = "http://localhost:8080/health"

    print("Checking router health...")

    try:
        response = requests.get(health_url, timeout=5)

        print(f"\nStatus Code: {response.status_code}")
        print("\n=== Health Response ===")
        print(json.dumps(response.json(), indent=2))

        if response.status_code == 200:
            data = response.json()
            print(f"\n✓ Router is {data['status']}")
            print(f"  Available credentials: {data['credentials_available']}/{data['total_credentials']}")
            print(f"  Banned credentials: {data['credentials_banned']}")
        else:
            print("\n✗ Router is unhealthy")

    except requests.exceptions.ConnectionError:
        print("✗ Error: Could not connect to router. Is it running?")
    except Exception as e:
        print(f"✗ Error: {e}")

if __name__ == "__main__":
    main()
