import os
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from scipy.spatial import ConvexHull, distance

class PerspectiveCorrection:
    def __init__(self, src_img):
        self.src_img = src_img
        self.corrected_img = None

    def process(self, src_img = None, characteristic_length = 1000):
        if (src_img is None):
            src_img = self.src_img
        anchor_pts = get_anchor_points(src_img) # In clockwise order -- TL, TR, BR, BL
        anchor_pts, transformed_pts, newH, newW = retain_aspect_ratio(src_img, anchor_pts)
        corrected_img = correct_perspective(src_img, anchor_pts, transformed_pts, newH, newW)
        self.corrected_img = corrected_img

    def save_output(self, output_filename, tgt_image_path):
        if (self.corrected_img is None):
            raise AttributeError("No non-null attribute 'corrected_img', call process() first?")

        corrected_img = self.corrected_img 
        isfolder = os.path.exists(tgt_image_path)
        if not isfolder:
            os.makedirs(tgt_image_path)
        
        if (corrected_img.shape[2] != 3):
            corrected_img = cv2.cvtColor(corrected_img, cv2.COLOR_GRAY2BGR)
        
        output_fp = os.path.join(tgt_image_path, f"{output_filename}_corrected.jpg")
        
        
        cv2.imwrite(output_fp, corrected_img)

def tellme(s, verbose = False):
    if (verbose):
        print(s)
    plt.title(s, fontsize=16)
    plt.draw()

def get_anchor_points(src_img):
    fig, ax = plt.subplots(1, 1, figsize = (20, 20))
    ax.imshow(src_img)
    ax.axis('off')
    tellme('Select 4 points defining the region of interest. Click to begin.')   
    plt.waitforbuttonpress()

    while True:
        pts = []
        while len(pts) < 3:
            pts = np.asarray(plt.ginput(4, timeout=-1))
            if len(pts) < 3:
                tellme('Too few points, starting over')
                time.sleep(1)  # Wait a second
        
        hull = ConvexHull(pts)
        pts = pts[hull.vertices]

        ph = plt.fill(pts[:, 0], pts[:, 1], 'limegreen', lw=2, alpha = 0.3)

        tellme('Happy? Key click for yes, mouse click for no.')

        if plt.waitforbuttonpress():
            break

        # Get rid of fill
        for p in ph:
            p.remove()
    
    min_distance_ind = np.argmin(np.linalg.norm(pts, axis = 1))
    pts = np.roll(pts, -min_distance_ind, axis = 0)
    pts = pts.astype(int)

    return pts

def retain_aspect_ratio(img, p, characteristic_length = 1000):
    # Expects in zig-zag TL, TR, BL, BR
    p[[2,3]] = p[[3, 2]]
    
    (rows,cols,_) = img.shape
    #image center
    u0 = (cols)/2.0
    v0 = (rows)/2.0

    #widths and heights of the projected image
    w1 = distance.euclidean(p[0],p[1])
    w2 = distance.euclidean(p[2],p[3])

    h1 = distance.euclidean(p[0],p[2])
    h2 = distance.euclidean(p[1],p[3])

    w = max(w1,w2)
    h = max(h1,h2)

    #visible aspect ratio
    ar_vis = float(w)/float(h)

    #make numpy arrays and append 1 for linear algebra
    m1 = np.array((p[0][0],p[0][1],1)).astype('float32')
    m2 = np.array((p[1][0],p[1][1],1)).astype('float32')
    m3 = np.array((p[2][0],p[2][1],1)).astype('float32')
    m4 = np.array((p[3][0],p[3][1],1)).astype('float32')

    #calculate the focal distance
    k2 = np.dot(np.cross(m1,m4),m3) / np.dot(np.cross(m2,m4),m3)
    k3 = np.dot(np.cross(m1,m4),m2) / np.dot(np.cross(m3,m4),m2)

    n2 = k2 * m2 - m1
    n3 = k3 * m3 - m1

    n21 = n2[0]
    n22 = n2[1]
    n23 = n2[2]

    n31 = n3[0]
    n32 = n3[1]
    n33 = n3[2]

    f = np.sqrt(np.abs((1.0/(n23*n33)) * ((n21*n31 - (n21*n33 + n23*n31)*u0 + n23*n33*u0*u0) + (n22*n32 - (n22*n33+n23*n32)*v0 + n23*n33*v0*v0))))

    A = np.array([[f,0,u0],[0,f,v0],[0,0,1]]).astype('float32')

    At = np.transpose(A)
    Ati = np.linalg.inv(At)
    Ai = np.linalg.inv(A)

    #calculate the real aspect ratio
    ar_real = np.sqrt(np.dot(np.dot(np.dot(n2,Ati),Ai),n2)/np.dot(np.dot(np.dot(n3,Ati),Ai),n3))

    if ar_real < ar_vis:
        W = int(w)
        H = int(W / ar_real)
    else:
        H = int(h)
        W = int(ar_real * H)

    if (W > H):
        newW = int(characteristic_length)
        newH = int((H / W) * characteristic_length)
    else:
        newH = int(characteristic_length)
        newW = int((W / H) * characteristic_length)

    transformed_pts = np.array([[0, 0], [0, newH - 1], [newW-1, newH-1], [newW-1, 0]])
    
    # currently TL, TR, BL, BR
    anchor_pts = np.array([p[0], p[2], p[3], p[1]])

    return anchor_pts, transformed_pts, newH, newW

def compute_homography(p1, p2):	
    v1, v2, v3, v4 = tuple(p1)
    u1, u2, u3, u4 = tuple(p2)

    A = np.array([[u1[0], u1[1], 1, 0, 0, 0, -u1[0]*v1[0], -u1[1]*v1[0], -v1[0]],
                 [0, 0, 0, u1[0], u1[1], 1, -u1[0]*v1[1], -u1[1]*v1[1], -v1[1]],
                 
                 [u2[0], u2[1], 1, 0, 0, 0, -u2[0]*v2[0], -u2[1]*v2[0], -v2[0]],
                 [0, 0, 0, u2[0], u2[1], 1, -u2[0]*v2[1], -u2[1]*v2[1], -v2[1]],
                 
                 [u3[0], u3[1], 1, 0, 0, 0, -u3[0]*v3[0], -u3[1]*v3[0], -v3[0]],
                 [0, 0, 0, u3[0], u3[1], 1, -u3[0]*v3[1], -u3[1]*v3[1], -v3[1]],
                 
                 [u4[0], u4[1], 1, 0, 0, 0, -u4[0]*v4[0], -u4[1]*v4[0], -v4[0]],
                 [0, 0, 0, u4[0], u4[1], 1, -u4[0]*v4[1], -u4[1]*v4[1], -v4[1]]])
  
    U, S, VT = np.linalg.svd(A)

    h = VT[-1]

    h /= h[-1]
    
    H = np.reshape(h, (3, 3))

    return H

def correct_perspective(img, source_pts, target_pts, height, width):
    if (img.shape[2] == 3):
        canvas = np.zeros((height, width, 3))
    else:
        canvas = np.zeros((height, width))
    H = compute_homography(source_pts, target_pts)
    
    grid = np.meshgrid(np.arange(width), np.arange(height))
    positions = np.vstack(list(map(np.ravel, grid))).T
    positions = np.hstack((positions, np.ones((positions.shape[0],1)))).T

    p1_warped_coords = H @ positions
    p1_warped_coords = p1_warped_coords / p1_warped_coords[-1, :]
    p1_warped_coords = p1_warped_coords[:2]
    p1_warped_coords = (p1_warped_coords).astype(int)
    
    if (img.shape[2] == 3):
        canvas += np.reshape(img[p1_warped_coords[1], p1_warped_coords[0]], (height, width, 3))
    else:
        canvas += np.reshape(img[p1_warped_coords[1], p1_warped_coords[0]], (height, width))

    return canvas


