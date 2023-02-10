from imutils import contours
import numpy as np
import imutils
import cv2

FIRST_NUMBER = {
	"3": "American Express",
	"4": "Visa",
	"5": "MasterCard",
	"6": "Discover Card"
}

class CardReader:
    def __init__(self, file):
        self.ref = cv2.imread('images/ocr_a_reference.png')
        self.ref = cv2.cvtColor(self.ref, cv2.COLOR_BGR2GRAY)
        self.ref = cv2.threshold(self.ref, 10, 255, cv2.THRESH_BINARY_INV)[1]
        self.image = cv2.imread(file)
        self.image = imutils.resize(self.image, width=300)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
    
    def find_contours(self):
        self.refCnts = cv2.findContours(self.ref.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        self.refCnts = imutils.grab_contours(self.refCnts)
        self.refCnts = contours.sort_contours(self.refCnts, method="left-to-right")[0]
        self.digits = {}
    
    def sort_contours(self):
        for (i, c) in enumerate(self.refCnts):
            # compute the bounding box for the digit, extract it, and resize
            # it to a fixed size
            (x, y, w, h) = cv2.boundingRect(c)
            self.roi = self.ref[y:y + h, x:x + w]
            self.roi = cv2.resize(self.roi, (57, 88))
            # update the self.digits dictionary, mapping the digit name to the ROI
            self.digits[i] = self.roi
    
    def isolate_digit(self):
        self.rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
        self.sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    

    def apply_morph(self):
        self.tophat = cv2.morphologyEx(self.gray, cv2.MORPH_TOPHAT, self.rectKernel)

    def compute_gradient(self):
        self.gradX = cv2.Sobel(self.tophat, ddepth=cv2.CV_32F, dx=1, dy=0,
            ksize=-1)
        self.gradX = np.absolute(self.gradX)
        (minVal, maxVal) = (np.min(self.gradX), np.max(self.gradX))
        self.gradX = (255 * ((self.gradX - minVal) / (maxVal - minVal)))
        self.gradX = self.gradX.astype("uint8")
    
    def find_digit(self):
        self.gradX = cv2.morphologyEx(self.gradX, cv2.MORPH_CLOSE, self.rectKernel)
        self.thresh = cv2.threshold(self.gradX, 0, 255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # apply a second closing operation to the binary self.image, again
        # to help close gaps between credit card number regions
        self.thresh = cv2.morphologyEx(self.thresh, cv2.MORPH_CLOSE, self.sqKernel)
    
    def find_card_contours(self):
        self.cnts = cv2.findContours(self.thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        self.cnts = imutils.grab_contours(self.cnts)
        self.locs = []

    def filter_contour(self):
        # loop over the contours
        for (i, c) in enumerate(self.cnts):
            # compute the bounding box of the contour, then use the
            # bounding box coordinates to derive the aspect ratio
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)
            # since credit cards used a fixed size fonts with 4 groups
            # of 4 self.digits, we can prune potential contours based on the
            # aspect ratio
            if ar > 2.5 and ar < 4.0:
                # contours can further be pruned on minimum/maximum width
                # and height
                if (w > 40 and w < 55) and (h > 10 and h < 20):
                    # append the bounding box region of the self.digits group
                    # to our locations list
                    self.locs.append((x, y, w, h))
    
    def sort_grouping(self):
        # sort the digit locations from left-to-right, then initialize the
        # list of classified self.digits
        self.locs = sorted(self.locs, key=lambda x:x[0])
        self.output = []

    def find_digit_in_grouping(self):
        # loop over the 4 groupings of 4 self.digits
        for (i, (gX, gY, gW, gH)) in enumerate(self.locs):
            # initialize the list of group self.digits
            groupOutput = []
            # extract the group ROI of 4 self.digits from the grayscale self.image,
            # then apply thresholding to segment the self.digits from the
            # background of the credit card
            group = self.gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
            group = cv2.threshold(group, 0, 255,
                cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            # detect the contours of each individual digit in the group,
            # then sort the digit contours from left to right
            digitCnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
            digitCnts = imutils.grab_contours(digitCnts)
            digitCnts = contours.sort_contours(digitCnts,
                method="left-to-right")[0]
        
            # loop over the digit contours
            for c in digitCnts:
                # compute the bounding box of the individual digit, extract
                # the digit, and resize it to have the same fixed size as
                # the reference OCR-A images
                (x, y, w, h) = cv2.boundingRect(c)
                self.roi = group[y:y + h, x:x + w]
                self.roi = cv2.resize(self.roi, (57, 88))
                # initialize a list of template matching scores	
                scores = []
                # loop over the reference digit name and digit ROI
                for (digit, digitROI) in self.digits.items():
                    # apply correlation-based template matching, take the
                    # score, and update the scores list
                    result = cv2.matchTemplate(self.roi, digitROI,
                        cv2.TM_CCOEFF)
                    (_, score, _, _) = cv2.minMaxLoc(result)
                    scores.append(score)
                # the classification for the digit ROI will be the reference
                # digit name with the *largest* template matching score
                groupOutput.append(str(np.argmax(scores)))
            
            # draw the digit classifications around the group
            cv2.rectangle(self.image, (gX - 5, gY - 5),
                (gX + gW + 5, gY + gH + 5), (0, 0, 255), 2)
            cv2.putText(self.image, "".join(groupOutput), (gX, gY - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            # update the self.output self.digits list
            self.output.extend(groupOutput)

        return self.output

    def execute(self):
        self.find_contours()
        self.sort_contours()
        self.isolate_digit()
        self.apply_morph()
        self.compute_gradient()
        self.find_digit()
        self.find_card_contours()
        self.filter_contour()
        self.sort_grouping()
        return self.find_digit_in_grouping()
        