import cv2 as cv
import numpy as np


class BoundingBox:

    def __init__(self, image, out_path):
        self.image = image
        self.out_path = out_path

    def show_image(self, img, name):
        cv.imshow(name, img)
        cv.waitKey(0)

    def sorted_contours(self, cnts):
        bounding_box = [cv.boundingRect(cnt) for cnt in cnts]
        (cnts, bounding_box) = zip(*sorted(zip(cnts, bounding_box),
                                           key=lambda bb: bb[1][0], reverse=False))
        return cnts, bounding_box

    def scale_image(self):
        # rgb = cv.pyrDown(self.image)
        rgb = self.image
        rgb = cv.resize(rgb, (600, 800), interpolation=cv.INTER_AREA)
        gray_image = cv.cvtColor(rgb, cv.COLOR_BGR2GRAY)
        self.show_image(gray_image, "gray_image")
        return rgb, gray_image

    def threshold_image(self, grad_image):
        _, bo_image = cv.threshold(grad_image, 0.0, 255.0, cv.THRESH_BINARY
                                   | cv.THRESH_OTSU)
        return bo_image

    def process_image(self, gray_image):
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 7))
        grad_image = cv.morphologyEx(gray_image, cv.MORPH_GRADIENT, kernel)
        thresh_image = self.threshold_image(grad_image)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 4))
        connected_image = cv.morphologyEx(thresh_image, cv.MORPH_CLOSE, kernel)
        return thresh_image, connected_image

    def find_contours(self):
        rgb, gray_image = self.scale_image()
        thresh_img, conn_image = self.process_image(gray_image)
        conn, hrchy = cv.findContours(conn_image.copy(), cv.RETR_EXTERNAL,
                                      cv.CHAIN_APPROX_NONE)
        return conn, hrchy, thresh_img, rgb

    def write_image(self, roi, count):
        output_path = self.out_path + "file {}.png"
        cv.imwrite(output_path.format(count), roi)

    def bounding_box_detector(self):
        count = 0
        contours, hierarchy, thresh_img, rgb_image = self.find_contours()
        mask = np.zeros(thresh_img.shape, dtype=np.uint8)
        sorted_ctrs, bounding_box = self.sorted_contours(contours)
        for index in range(len(sorted_ctrs)):
            x, y, w, h = cv.boundingRect(sorted_ctrs[index])
            mask[y:y + h, x:x + w] = 0
            cv.drawContours(mask, sorted_ctrs, index, (255, 255, 255), -1)
            coverage = float(cv.countNonZero(mask[y:y + h, x:x + w])) / (w * h)
            if coverage > 0.35 and w > 5 and h > 5:
                count += 1
                cv.rectangle(rgb_image, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 1)
                roi = rgb_image[y:y + h, x:x + w]
                self.write_image(roi, count)
        self.show_image(rgb_image, "rects")
        self.write_image(rgb_image, count)


if __name__ == '__main__':
    imgs = cv.imread('images/test5.png')
    output_paths = 'debug/'
    bounding_boxes = BoundingBox(imgs, output_paths)
    bounding_boxes.bounding_box_detector()
