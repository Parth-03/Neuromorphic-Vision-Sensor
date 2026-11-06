import numpy as np
import matplotlib.pyplot as plt
import cv2
import csv
from PyLTSpice import SimCommander, RawRead # Requires python >= 3.9
import os
from pathlib import Path
import shutil

##### This script will not run unless it is ran locally on a machine with LTSpice installed #####
##### If an error is thrown that LTSpice the app cannot be found see line ~150 for the fix #####

# ON = 1 will execute video creation, SPEED = 0 will execute dyanmic thresholds
ON = 0
SPEED = 0
# I placed these to speedup my test runs - Jacob

# Change to absolute path of desired video
video_path = "/Users/jacoby/Downloads/NeuromorphTest2.mov"

# Defines box side length in pixels to simulate through LTSpice (Choose an even value) 
SPICE_GRID_SIZE = 2        # DO NOT CHANGE UNLESS YOU KNOW WHAT YOU'RE DOING
# Total pixels ran through LTSpice per frame will be (SPICE_GRID_SIZE^2)

# Store Temporary Files From LTSpice simulations
LTSPICE_CACHE = Path("LTSpice_cache")
LTSPICE_CACHE.mkdir(exist_ok=True)

# Function to emulate dvs analysis
# Passes in a fixed-frame video, displays intensity change at pixel and time
def dvs_emulate(video, timestamps, C_on, C_off):

    # Video parameter definition: number of frames, vertical and horizontal pixels
    T, H, W = video.shape
    eps = 1e-3  # to avoid log(0), shifts all intensities slightly
    clip_min = 1e-2 # To reduce the log(close to 0) max/min delta increase
    logIntensity = np.log(np.clip(video, clip_min, 1.0) + eps) # Takes log of offset and clipped video intensities

    last_event_time = np.full((H, W), -np.inf) # Take time of event
 
    prev_logIntensity = logIntensity[0].copy() # Set the previous intensity
    events = [] # Initialize events array
    ltspice_events = [] # Initialize LTSpice events array

    # Validation tracking
    correct = 0
    incorrect = 0

    for i in range(1, T):
        
        half_grid = SPICE_GRID_SIZE // 2
        # DO NOT CHANGE UNTIL YOU KNOW WHAT YOU ARE DOING THIS WILL MASSIVELY INFLUENCE RUNTIME #
        ystart, yend = H // 2 - half_grid, H // 2 + half_grid       # Written as such to capture near the center of the scene
        xstart, xend = W // 2 - half_grid, W // 2 + half_grid       # These can be manually changed to pixels you want to
        frame_start, frame_end = 60, 65                             # specifically target - Jacob
        #########################################################################################

        if frame_start <= i < frame_end:
            # Because we start midway through video we can't initialize from zero so we do it as such
            if i == frame_start:
                ref_initial = np.zeros((yend-ystart, xend-xstart))
                ref_initial[:, :] = np.log(np.clip(video[i-1, ystart:yend, xstart:xend], clip_min, 1.0) + eps)
                print("Status Update: LTSpice Simulations Starting (This may be time consuming)")

            # Compute the emulated DVS deltas within the ROI for the frame
            delta_region = logIntensity[i, ystart:yend, xstart:xend] - logIntensity[i-1, ystart:yend, xstart:xend]


            for y in range(ystart, yend):
                for x in range(xstart, xend):
                    apparenty = y-ystart
                    apparentx = x-xstart

                    current_raw = np.clip(video[i, y, x], clip_min, 1.0)
                    prev_intensity = ref_initial[apparenty, apparentx]
                    spice_event, vref_at_sim_end = LTSpiceSim(current_raw, prev_intensity, C_on, abs(C_off))

                    # Determine what the emulator logic predicts for this pixel
                    if delta_region[apparenty, apparentx] >= C_on:
                        emulated_event = +1
                    elif delta_region[apparenty, apparentx] <= C_off:
                        emulated_event = -1
                    else:
                        emulated_event = 0

                    # Compare LTSpice event with emulated event
                    if spice_event == emulated_event:
                        correct += 1
                    else:
                        incorrect += 1

                    # Update memory for NEXT frame, Note: This replaces a reset switch in the circuit model
                    ref_initial[apparenty, apparentx] = vref_at_sim_end

                    if spice_event != 0:
                        ltspice_events.append([x, y, timestamps[i], spice_event])
                        print(f"Frame: {i}, Location: x={x} y={y}, Event: {spice_event}")


        if i == frame_end-1:            
            print("Status Update: All LTSpice Simulations Complete")
            print("\nValidation Statistics")
            print(f"Total Pixels Simulated: {(yend-ystart)*(xend-xstart)*(frame_end-frame_start)}")
            print(f"Correct Predictions: {correct}")
            print(f"Incorrect Predictions: {incorrect}")
            print(f"Validation Accuracy: {100*correct/(correct+incorrect)}%")

        delta = logIntensity[i] - prev_logIntensity # Compute log intensity difference

        # Positive (brighter)
        on_mask = delta >= C_on             
        # Negative (darker)
        off_mask = delta <= C_off       

        # Extract coordinates for both types
        on_y, on_x = np.nonzero(on_mask)
        off_y, off_x = np.nonzero(off_mask)

        # Append ON events
        for y, x in zip(on_y, on_x):
            events.append([x, y, timestamps[i], +1])
            last_event_time[y, x] = timestamps[i]

        # Append OFF events
        for y, x in zip(off_y, off_x):
            events.append([x, y, timestamps[i], -1])
            last_event_time[y, x] = timestamps[i]
            
        # Reset reference intensity where events occurred
        prev_logIntensity = logIntensity[i]
        
    return np.array(events), np.array(ltspice_events)

# Handles all pipelining between python and LTSpice
def LTSpiceSim(current_raw, prev_intensity, C_on, C_off):

    # Gather netlist path and top level working directory
    netlist_path = Path("DVS_Pixel.net").resolve()
    original_cwd = os.getcwd()
    
    try:
        os.chdir(LTSPICE_CACHE) # Enter cache folder so top level directory doesn't get polluted
        
        # Copy netlist to cache directory
        cache_netlist = Path("DVS_Pixel.net")
        if not cache_netlist.exists() or cache_netlist.stat().st_mtime < netlist_path.stat().st_mtime:
            shutil.copy(netlist_path, cache_netlist)
        
        ############ FOR NO LTSPICE APP FOUND ################
        # If LTSpice is not found automatically, specify the path in SimCommander: SimCommander("DVS_Pixel.net")
        # CHANGE TO SimCommander("DVS_Pixel.net", simulator="/absolute/application/file/path")
        # For more details see https://pypi.org/project/PyLTSpice/ and https://github.com/nunobrum/spicelib/blob/main/spicelib/simulators/ltspice_simulator.py

        # Now run sim and output files within cache folder
        sim = SimCommander("DVS_Pixel.net") 
        sim.set_parameters(
            VLight=current_raw,
            C_on=C_on,
            C_off=C_off,
            ref_ic=prev_intensity)
        
        sim.run()
        sim.wait_completion()
        
        # Look for .raw files in cache
        raw_files = sorted(Path(".").glob("*.raw"), key=os.path.getmtime)
        
        # Use the most recent .raw file (Was expecting file explosion but that didn't maintain this structure for security)
        latest_raw = raw_files[-1]
        raw = RawRead(str(latest_raw))
        
        # Gather relevant simulation data (Sim is 30us with 100ns steps)
        von = max(raw.get_trace("V(on)").data[:]) # Max of full sim data 
        voff = min(raw.get_trace("V(off)").data[:]) # Min of full sim data
        vref_at_sim_end = raw.get_trace("V(ref)").data[-1] # Gather value at last data point
        
        if von > 0.5:
            event = +1
        elif voff < -0.5:
            event = -1
        else:
            event = 0
        
        return event, vref_at_sim_end
    
    finally:
        os.chdir(original_cwd)

# Capture video data
cap = cv2.VideoCapture(video_path)

# Error message if video fails to open
if not cap.isOpened():
    raise ValueError("Cannot open video")

# Initialize frame array
frames = []

# Iterate through video frames and check successful capture
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Normalize to 0-1
    gray = gray.astype(np.float32) / 255.0

    # Downsample
    scale = 1
    gray_downsampled = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    frames.append(gray_downsampled)
    # Show video
    
# Release captured video frames
cap.release()
video = np.stack(frames)  # create 'video' out of list of frames with shape = (T, H, W)

print("Video shape:", video.shape)

if SPEED == 0:
    # Compute max and min deltas before function execution to dynamically define thresholds and print
    logIntensity = np.log(np.clip(video, 1e-2, 1.0) + 1e-3)
    delta = logIntensity[1:] - logIntensity[:-1]
    maxthresh = delta.max() - (0.9*delta.max())
    minthresh = delta.min() + (0.9*-delta.min())
else:
    maxthresh = .4003024
    minthresh = -.3729496

print("maxthresh: ", maxthresh)
print("minthresh: ", minthresh)

# Turns frame array into a series of times from frame count 'timestamps'
timestamps = np.arange(video.shape[0]) / 30

# Create events array using the function with + & - thresholds
events, ltspice_events = dvs_emulate(video, timestamps, maxthresh, minthresh)

# Cleanup LTSpice cache folder
if LTSPICE_CACHE.exists():
    shutil.rmtree(LTSPICE_CACHE)

print("Total events:", len(events))
print(f"{len(ltspice_events)} events detected by the LTSpice circuit")

if ON == 1:
    # Create Video
    H, W = video.shape[1], video.shape[2]
    output_path = "dvs_events_output.mp4"
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H), isColor=False)

    # Convert events list to numpy array
    events = np.array(events)

    # Frames are defined by timestamps[i] = i/30
    frame_events = {i: [] for i in range(len(video))}
    for x, y, t, p in events:
        frame_index = int(t * fps)  # convert time to frame index
        if frame_index < len(video):
            frame_events[frame_index].append((int(x), int(y), int(p)))

    # Create video visualization
    for i in range(len(video)):
        # Start with gray background
        event_frame = np.full((H, W), 128, dtype=np.uint8)

        for (x, y, polarity) in frame_events[i]:
            if polarity == +1:      # ON = bright
                event_frame[y, x] = 255
            else:                   # OFF = dark
                event_frame[y, x] = 0

        out.write(event_frame)

    out.release()
    print("Event video saved as:", output_path)

    csv_path = "dvs_events_output.csv"

    # Write header + event rows
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "timestamp", "polarity"])
        for x, y, t, p in events:
            writer.writerow([int(x), int(y), float(t), int(p)])

    print("CSV saved as:", csv_path)