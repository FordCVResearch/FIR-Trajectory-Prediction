import cv2
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt


FRAME_RATE = 20
SMOOTHING_RADIUS = 20

def movingAverage(curve, radius):
    window_size = 2 * radius + 1
    # Define the filter
    f = np.ones(window_size)/window_size
    # Add padding to the boundaries
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
    # Apply convolution
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    # Remove padding
    curve_smoothed = curve_smoothed[radius:-radius]
    # return smoothed curve
    return curve_smoothed

def smooth(trajectory):
    smoothed_trajectory = np.copy(trajectory)
    # Filter the x, y and angle curves
    for i in range(3):
        smoothed_trajectory[:,i] = movingAverage(trajectory[:,i], radius=SMOOTHING_RADIUS)

    return smoothed_trajectory

def fixBorder(frame):
    s = frame.shape
    # Scale the image 4% without moving the center
    T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.04)
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame

VID_PATH = './videos/seq_1_vid.avi'
cap = cv2.VideoCapture(VID_PATH)

n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
FRAME_SIZE_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
FRAME_SIZE_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'XVID') 
out = cv2.VideoWriter('video_out.avi', fourcc, FRAME_RATE, (FRAME_SIZE_W, FRAME_SIZE_H))

_, prev = cap.read()
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

transforms = np.zeros((n_frames-1, 3), np.float32)

for i in range(n_frames-2):
    prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                        maxCorners=200,
                                        qualityLevel=0.01,
                                        minDistance=30,
                                        blockSize=3)
    
    success, curr = cap.read()
    if not success:
        break
    
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
    
    assert prev_pts.shape == curr_pts.shape
    idx = np.where(status==1)[0]
    prev_pts = prev_pts[idx]
    curr_pts = curr_pts[idx]
    
    #Find transformation matrix
    m = cv2.estimateRigidTransform(prev_pts, curr_pts, fullAffine=False) #will only work with OpenCV-3 or less
    dx = m[0,2]
    dy = m[1,2]
    
    da = np.arctan2(m[1,0], m[0,0])
    transforms[i] = [dx,dy,da]
    
    prev_gray = curr_gray
    print("Frame: " + str(i) +  "/" + str(n_frames) + " -  Tracked points : " + str(len(prev_pts)))
    
trajectory = np.cumsum(transforms, axis=0)
smoothed_trajectory = smooth(trajectory)

# Calculate difference in smoothed_trajectory and trajectory
difference = smoothed_trajectory - trajectory
# Calculate newer transformation array
transforms_smooth = transforms + difference

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot3D(smoothed_trajectory[:,0], smoothed_trajectory[:,1], smoothed_trajectory[:,2], 'gray')
# plt.show()

# Reset stream to first frame
cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 

# Write n_frames-1 transformed frames
for i in range(n_frames-2):
    # Read next frame
    success, frame = cap.read()
    if not success:
     break

    # Extract transformations from the new transformation array
    dx = transforms_smooth[i,0]
    dy = transforms_smooth[i,1]
    da = transforms_smooth[i,2]

    # Reconstruct transformation matrix accordingly to new values
    m = np.zeros((2,3), np.float32)
    m[0,0] = np.cos(da)
    m[0,1] = -np.sin(da)
    m[1,0] = np.sin(da)
    m[1,1] = np.cos(da)
    m[0,2] = dx
    m[1,2] = dy

    frame_stabilized = cv2.warpAffine(frame, m, (FRAME_SIZE_W, FRAME_SIZE_H))
    frame_stabilized = fixBorder(frame_stabilized) 
    frame_out = cv2.hconcat([frame, frame_stabilized])

    if(frame_out.shape[1] > 1920):
        frame_out = cv2.resize(frame_out, (frame_out.shape[1]/2, frame_out.shape[0]/2));

    cv2.imshow("Before and After", frame_out)
    cv2.waitKey(10)
    out.write(frame_stabilized)

print("Done")