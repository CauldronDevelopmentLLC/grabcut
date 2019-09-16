#!/usr/bin/env python
'''
===============================================================================
# grabcut

A simple program for interactively removing the background from an image using
the grab cut algorithm and OpenCV.

This code was derived from the Grab Cut example from the OpenCV project.

## Usage
    grabcut.py <input> [output]

## Operation

At startup, two windows will appear, one for input and one for output.

To start, in input window, draw a rectangle around the object using mouse right
button.  For finer touch-ups, press any of the keys below and draw circles on
the areas you want.  Finally, press 's' to save the result.

## Keys
  * 0 - Select areas of sure background
  * 1 - Select areas of sure foreground
  * 2 - Select areas of probable background
  * 3 - Select areas of probable foreground
  * n - Update the segmentation
  * r - Reset the setup
  * s - Save the result
  * q - Quit
===============================================================================
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import sys


class App():
    BLUE  = [255, 0, 0]       # rectangle color
    RED   = [0, 0, 255]       # PR BG
    GREEN = [0, 255, 0]       # PR FG
    BLACK = [0, 0, 0]         # sure BG
    WHITE = [255, 255, 255]   # sure FG

    DRAW_BG    = {'color' : BLACK, 'val' : 0}
    DRAW_FG    = {'color' : WHITE, 'val' : 1}
    DRAW_PR_FG = {'color' : GREEN, 'val' : 3}
    DRAW_PR_BG = {'color' : RED,   'val' : 2}

    thickness  = 3


    def onmouse(self, event, x, y, flags, param):
        # Draw rectangle
        if event == cv.EVENT_RBUTTONDOWN:
            self.rectangle = True
            self.ix, self.iy = x,y

        elif event == cv.EVENT_MOUSEMOVE:
            if self.rectangle == True:
                self.img = self.copy.copy()
                cv.rectangle(self.img, (self.ix, self.iy), (x, y), self.BLUE, 2)
                self.rect = (min(self.ix, x), min(self.iy, y), abs(self.ix - x),
                             abs(self.iy - y))
                self.rect_or_mask = 0

        elif event == cv.EVENT_RBUTTONUP:
            self.rectangle = False
            self.rect_over = True
            cv.rectangle(self.img, (self.ix, self.iy), (x, y), self.BLUE, 2)
            self.rect = (min(self.ix, x), min(self.iy, y), abs(self.ix - x),
                         abs(self.iy - y))
            self.rect_or_mask = 0
            self.segment()

        # Draw touchup curves
        if event == cv.EVENT_LBUTTONDOWN:
            if not self.rect_over: print('First draw a rectangle')

            else:
                self.drawing = True
                cv.circle(self.img, (x,y), self.thickness,
                          self.value['color'], -1)
                cv.circle(self.mask, (x,y), self.thickness,
                          self.value['val'], -1)

        elif event == cv.EVENT_MOUSEMOVE:
            if self.drawing == True:
                cv.circle(self.img, (x, y), self.thickness,
                          self.value['color'], -1)
                cv.circle(self.mask, (x, y), self.thickness,
                          self.value['val'], -1)

        elif event == cv.EVENT_LBUTTONUP:
            if self.drawing == True:
                self.drawing = False
                cv.circle(self.img, (x, y), self.thickness,
                          self.value['color'], -1)
                cv.circle(self.mask, (x, y), self.thickness,
                          self.value['val'], -1)
                self.segment()


    def reset(self):
        print('Resetting')

        self.rect = (0, 0, 1, 1)
        self.drawing = False
        self.rectangle = False
        self.rect_or_mask = 100
        self.rect_over = False
        self.value = self.DRAW_FG

        self.img = self.copy.copy()
        self.mask = np.zeros(self.img.shape[:2], dtype = np.uint8)
        self.output = np.zeros(self.img.shape, np.uint8)


    def crop_to_alpha(self, img):
        x, y = self.alpha.nonzero()
        if len(x) == 0 or len(y) == 0: return img
        return img[np.min(x) : np.max(x), np.min(y) : np.max(y)]


    def save(self):
        # Apply alpha
        b, g, r, = cv.split(self.copy)
        img = cv.merge((b, g, r, self.alpha))

        cv.imwrite(self.outfile, self.crop_to_alpha(img))
        print('Saved')


    def segment(self):
        try:
            if self.rect_or_mask == 0:
                mask_type = cv.GC_INIT_WITH_RECT
                self.rect_or_mask = 1

            elif self.rect_or_mask == 1:
                mask_type = cv.GC_INIT_WITH_MASK

            bgdmodel = np.zeros((1, 65), np.float64)
            fgdmodel = np.zeros((1, 65), np.float64)
            cv.grabCut(self.copy, self.mask, self.rect, bgdmodel, fgdmodel, 1,
                       mask_type)

        except:
            import traceback
            traceback.print_exc()


    def load(self):
        self.outfile = 'grabcut.png'
        if len(sys.argv) == 2: filename = sys.argv[1]
        elif len(sys.argv) == 3: filename, self.outfile = sys.argv[1:3]
        else: raise Exception('Usage: grabcut.py <input> [output]')

        self.img    = cv.imread(filename)
        self.copy   = self.img.copy()             # a copy of original image
        self.mask   = np.zeros(self.img.shape[:2], dtype = np.uint8)
        self.output = np.zeros(self.img.shape, np.uint8)
        self.alpha  = np.zeros(self.img.shape[:2], dtype = np.uint8)


    def run(self):
        self.load()
        self.reset()

        # Input and output windows
        cv.namedWindow('output')
        cv.namedWindow('input')
        cv.setMouseCallback('input', self.onmouse)
        cv.moveWindow('input', self.img.shape[1] + 10, 90)

        print('Draw a rectangle around the object using right mouse button')

        while True:
            cv.imshow('output', self.output)
            cv.imshow('input', self.img)
            k = cv.waitKey(1)

            # key bindings
            if k == 27 or k == ord('q'): break # exit
            elif k == ord('0'): self.value = self.DRAW_BG
            elif k == ord('1'): self.value = self.DRAW_FG
            elif k == ord('2'): self.value = self.DRAW_PR_BG
            elif k == ord('3'): self.value = self.DRAW_PR_FG
            elif k == ord('s'): self.save()
            elif k == ord('r'): self.reset()
            elif k == ord('n'): self.segment()

            self.alpha = np.where((self.mask == 1) + (self.mask == 3), 255,
                                  0).astype('uint8')
            img = cv.bitwise_and(self.copy, self.copy, mask = self.alpha)
            self.output = self.crop_to_alpha(img)


if __name__ == '__main__':
    print(__doc__)
    App().run()
    cv.destroyAllWindows()
