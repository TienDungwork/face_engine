import requests
import json
import time


def test_face_recognition():
    """Test face recognition from base64.txt file"""

    # Read base64 image from file
    try:
        with open("base64.txt", "r") as f:
            base64_image = f.read().strip()
        print(f"✅ Successfully read base64 image from file")
        print(f"📏 Base64 length: {len(base64_image)} characters")
    except FileNotFoundError:
        print("❌ Error: base64.txt file not found")
        return
    except Exception as e:
        print(f"❌ Error reading base64.txt: {e}")
        return

    # API endpoint
    url = "http://192.168.1.191:54327/api/v1/searchFace"

    # Request payload
    payload = {
        "img_base64": base64_image,
        "comIds": [],  # Empty list means search all companies
        "threshold": 0.5,  # Similarity threshold
        "quality": 0.3  # Quality threshold
    }

    print(f"\n🚀 Making request to: {url}")
    print(f"📋 Request payload:")
    print(f"   - comIds: {payload['comIds']}")
    print(f"   - threshold: {payload['threshold']}")
    print(f"   - quality: {payload['quality']}")
    print(f"   - img_base64: {len(payload['img_base64'])} characters")

    # Make the request
    start_time = time.time()
    try:
        response = requests.post(url, json=payload, timeout=30)
        elapsed_time = time.time() - start_time

        print(f"\n⏱️  Request completed in {elapsed_time:.2f} seconds")
        print(f"📊 Response status: {response.status_code}")

        if response.status_code == 200:
            response_data = response.json()
            print(f"\n✅ Success! Response:")
            print(json.dumps(response_data, indent=2, ensure_ascii=False))

            # Extract and display key information
            if response_data.get('data'):
                data = response_data['data']

                # Input data
                if 'input_data' in data:
                    input_data = data['input_data']
                    print(f"\n📥 Input Data:")
                    print(
                        f"   - Face rectangle: {input_data.get('face_rectangle', 'N/A')}")
                    print(f"   - Quality: {input_data.get('quality', 'N/A')}")
                    print(
                        f"   - Wear mask: {input_data.get('wearmask', 'N/A')}")
                    print(f"   - Gender: {input_data.get('gender', 'N/A')}")
                    print(f"   - Age: {input_data.get('age', 'N/A')}")

                # Search results
                if 'result' in data:
                    results = data['result']
                    print(f"\n🔍 Search Results ({len(results)} matches):")
                    for i, person in enumerate(results, 1):
                        print(
                            f"   {i}. {person.get('fullname', 'Unknown')} (ID: {person.get('personId', 'N/A')})")
                        print(
                            f"      - Score: {person.get('scoreMatching', 'N/A')}")
                        print(
                            f"      - Code: {person.get('personCode', 'N/A')}")

                # Most similar person
                if 'similar' in data and data['similar']:
                    similar = data['similar']
                    print(f"\n🎯 Most Similar Person:")
                    print(f"   - Name: {similar.get('fullname', 'Unknown')}")
                    print(f"   - ID: {similar.get('personId', 'N/A')}")
                    print(f"   - Score: {similar.get('scoreMatching', 'N/A')}")
                else:
                    print(f"\n❌ No similar person found above threshold")

        else:
            print(f"\n❌ Error response:")
            try:
                error_data = response.json()
                print(json.dumps(error_data, indent=2))
            except:
                print(f"Response text: {response.text}")

    except requests.exceptions.ConnectionError:
        print(f"\n❌ Connection Error: Could not connect to the API server")
        print(f"   Make sure the server is running on port 42070")
    except requests.exceptions.Timeout:
        print(f"\n❌ Timeout Error: Request took too long")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")


if __name__ == "__main__":
    print("🧪 Face Recognition Test")
    print("=" * 50)
    test_face_recognition()
