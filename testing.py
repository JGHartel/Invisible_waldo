import sys

#if sys.platform.startswith('win'):
#    import asyncio
#    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import logging

from pupil_labs.realtime_api.simple import Device

ADDRESS = "192.168.137.234"  # Replace with your device's IP address
PORT = "8080"                # Replace with your device's port if different

def test_device_connection(address=ADDRESS, port=PORT):
    device = Device(address=address, port=port)
    try:
        i=0
        while True:
            if i>10:
                break

            # Attempt to create a device instance
            
            print("Successfully connected to the device.")

            # Retrieve a single frame and gaze data
            frame, gaze_data = device.receive_matched_scene_video_frame_and_gaze()
            print("Successfully received frame and gaze data.")

            # Access the image data
            frame_pixels = frame.bgr_pixels  # Use .rgb_pixels for RGB format

            # Print basic information about the frame and gaze data
            #print(f"Frame resolution: {frame.width}x{frame.height}")
            print(f"Frame data type: {type(frame_pixels)}")
            print(f"Frame shape: {frame_pixels.shape}")
            print(f"Gaze data: {gaze_data}")

            i=i+1

        # Close the device connection
        device.close()
        print("Device connection closed.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_device_connection()
