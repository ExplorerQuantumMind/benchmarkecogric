wrist_marker_index = 2  # <--- update this after your plot, e.g. 2 or 3
lwri_xyz = MotionData[wrist_marker_index, 0].astype(float)
speed = np.linalg.norm(np.diff(lwri_xyz, axis=0), axis=1) * 120
speed = np.concatenate(([0], speed))
# ...continue as before
