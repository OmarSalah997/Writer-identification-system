# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 22:30:13 2021

@author: somar
"""
import numpy as np
import cv2 as cv
class preprocessing:
    def __init__(self,img):
        # Store references to the page images.
        self.gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        th,self.binary = cv.threshold(self.gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        self.gray,self.binary=self.extract_paragraph(self.gray,self.binary)
        self.gray_paragraph=self.gray
        self.binary_paragraph=self.binary
        # Get horizontal histogram.
        self.hor_hist = np.sum(self.binary, axis=1, dtype=int) // 255
        # Get line density thresholds.
        self.threshold_high = int(np.max(self.hor_hist) // 3)
        self.threshold_low = 25
        # Initialize empty lists.
        self.peak_rows = []
        self.valley_rows = []
        self.lines_boundaries = []

        # Calculate peaks and valleys of the page.
        self.detect_peaks()
        self.avg_peaks_dist = int((self.peak_rows[-1] - self.peak_rows[0]) // len(self.peak_rows))
        self.detect_valleys()

        # Detect missing peaks and valleys in a second iteration.
        #self.detect_missing_peaks_valleys()

        # Detect line boundaries.
        self.detect_line_boundaries()
        self.lines_list,self.lines_bin_list=self.segmentLines()
        self.words=self.segmentWords()
    def get_lines(self):
        return self.lines_list
    def get_paragraph(self):
        return cv.cvtColor(self.gray_paragraph, cv2.COLOR_BGR2GRAY)
    def get_words(self):
        return self.words  
    def extract_paragraph(self,gray,binary):
        height, width = gray.shape
        _,contours, hierarchy = cv.findContours(binary, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        # Minimum line width
        min_line_width = 1500
        line_skew_offset = 10
        # cropping borders
        upper_row, down_row, left, right = 0, height - 1, 100, width - 40
        # get the 3 seprating lines
        for cnt in contours:
            x, y, w, h = cv.boundingRect(cnt)
            if x<10 or y<10 or w < min_line_width:
                continue
            if y < height // 2:
                upper_row = max(upper_row, y + h + line_skew_offset)
            else:
                down_row = min(down_row, y - line_skew_offset)       
        # erosion to remove noise and dots.
        kernel = np.ones((3, 3), np.uint8)
        eroded_img = cv.erode(binary, kernel, iterations=2)
        # Get horizontal and vertical histograms.
        horizontal_hist = np.sum(eroded_img, axis=1) / 255
        vertical_hist = np.sum(eroded_img, axis=0) / 255
        while left < right and vertical_hist[left] == 0:
            left += 1
        while right > left and vertical_hist[right] ==0:
            right -= 1
        while upper_row < down_row and horizontal_hist[upper_row] == 0:
            upper_row += 1
        while down_row > upper_row and horizontal_hist[down_row] == 0:
            down_row -= 1 
        # Crop hand written paragraph only
        gray_paragraph = gray[upper_row:down_row + 1, left:right + 1]
        binary_paragraph = binary[upper_row:down_row+ 1, left:right + 1]
        return gray_paragraph,binary_paragraph
    def segmentWords(self):
        words=[]
        kernel = np.ones((9, 9), np.uint8)
        #print("no. of lines= ",len(self.lines_bin_list))
        for i,j in zip(self.lines_bin_list, self.lines_list):
            binary_line = cv.dilate(i, kernel, iterations=2)
            ver_hist = np.sum(binary_line, axis=0, dtype=int) // 255
            k=0
            th=round(np.average(ver_hist))
            th/=1.5
            while k in range(len(ver_hist)):
                if(ver_hist[k]>=th):
                    for end in range(k,len(ver_hist)):
                        if(ver_hist[end]<1 and ver_hist[end+1]<1  and ver_hist[end+2]<1):
                            if end-k > 50:
                                words.append(j[:,k:end])
                            k=end
                            break
                k+=1           
        return words  
           
    def segmentLines(self):
        # Initialize lines lists.
        gray_lines, bin_lines = [], []
        # Loop on every line boundary.
        for l, u, r, d in self.lines_boundaries:
            # Crop gray line.
            g_line = self.gray[u:d + 1, l:r + 1]
            gray_lines.append(g_line)
            # Crop binary line.
            b_line = self.binary[u:d + 1, l:r + 1]
            bin_lines.append(b_line)

        # Return list of separated lines.
        return gray_lines, bin_lines
    def detect_peaks(self):
        self.peak_rows = []

        i = 0
        while i < len(self.hor_hist):
            # If the black pixels density of the row is below than threshold
            # then continue to the next row.
            if self.hor_hist[i] < self.threshold_high:
                i += 1
                continue

            # Get the row with the maximum density from the following
            # probable row lines.
            peak_idx = i
            while i < len(self.hor_hist) and self.is_peak(i):
                if self.hor_hist[i] > self.hor_hist[peak_idx]:
                    peak_idx = i
                i += 1

            # Add peak row index to the list.
            self.peak_rows.append(peak_idx)

    def detect_valleys(self):

        self.valley_rows = [0]

        i = 1
        while i < len(self.peak_rows):
            u = self.peak_rows[i - 1]
            d = self.peak_rows[i]
            i += 1

            expected_valley = d - self.avg_peaks_dist // 2
            valley_idx = u

            while u < d:
                dist1 = np.abs(u - expected_valley)
                dist2 = np.abs(valley_idx - expected_valley)

                cond1 = self.hor_hist[u] < self.hor_hist[valley_idx]
                cond2 = self.hor_hist[u] == self.hor_hist[valley_idx] and dist1 < dist2

                if cond1 or cond2:
                    valley_idx = u

                u += 1

            self.valley_rows.append(valley_idx)

        self.valley_rows.append(len(self.hor_hist) - 1)
    def detect_line_boundaries(self):

        # Get image dimensions.
        height, width = self.binary.shape

        self.lines_boundaries = []

        i = 1
        while i < len(self.valley_rows):
            u = self.valley_rows[i - 1]
            d = self.valley_rows[i]
            l = 0
            r = width - 1
            i += 1
            while u < d and self.hor_hist[u] == 0:
                u += 1
            while d > u and self.hor_hist[d] < 20:
                d -= 1

            ver_hist = np.sum(self.binary[u:d + 1, :], axis=0) // 255

            while l < r and ver_hist[l] == 0:
                l += 1
            while r > l and ver_hist[r] == 0:
                r -= 1
            
            self.lines_boundaries.append((l, u, r, d))    

    def calc_average_line_slope(self) -> int:
        avg_slope = 0

        i = 1
        while i < len(self.valley_rows):
            u = self.valley_rows[i - 1]
            d = self.valley_rows[i]
            avg_slope += self.calc_range_slope(u, d)
            i += 1

        return int(avg_slope // (len(self.valley_rows) - 1))

    def calc_range_slope(self, upper_row: int, down_row: int) -> int:
        max_der, min_der = -1e9, 1e9

        while upper_row < down_row:
            upper_row += 1
            val = self.hor_hist[upper_row] - self.hor_hist[upper_row - 1]
            max_der = max(max_der, val)
            min_der = min(min_der, val)

        return max_der - min_der

    def get_peak_in_range(self, upper_row: int, down_row: int) -> int:
        peak_idx = upper_row

        while upper_row < down_row:
            if self.hor_hist[upper_row] > self.hor_hist[peak_idx]:
                peak_idx = upper_row
            upper_row += 1

        return peak_idx
    def is_peak(self, row):
        width = 15
        for i in range(-width, width):
            if row + i < 0 or row + i >= len(self.hor_hist):
                continue
            if self.hor_hist[row + i] >= self.threshold_high:
                return True

        return False
    def is_valley(self, row: int) -> bool:
        width = 30
        count = 0
        for i in range(-width, width):
            if row + i < 0 or row + i >= len(self.hor_hist):
                return True
            if self.hor_hist[row + i] <= self.threshold_low:
                count += 1
        if count * 2 >= width:
            return True
        return False