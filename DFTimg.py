import cv2
from matplotlib import pyplot as plt
import numpy as np

class DFT():
    def __init__(self, img):
        self.grayImg = cv2.imread(img,0)
        self.img = cv2.imread(img)
        self.img = cv2.cvtColor(self.img,cv2.COLOR_BGR2RGB)

    def HPF(self,img,r):
        rows, cols = img.shape
        crow, ccol = int(rows / 2), int(cols / 2)

        mask = np.ones((rows, cols, 2), np.uint8)
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
        mask[mask_area] = 0


        # Apply mask and inverse DFT
        fshift = np.fft.fftshift(cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT))* mask
        fshift_mask_mag = 2000 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))

        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        return img_back
    
    def LPF(self,img,r):
        rows, cols = img.shape
        crow, ccol = int(rows / 2), int(cols / 2)
        mask = np.zeros((rows, cols, 2), np.uint8)
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
        mask[mask_area] = 1
        # Band Pass Filter - Concentric circle mask, only the points living in concentric circle are ones
        rows, cols = img.shape
        crow, ccol = int(rows / 2), int(cols / 2)
        mask = np.zeros((rows, cols, 2), np.uint8)
        r_out = 80
        r_in = 10
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = np.logical_and(((x - center[0]) ** 2 + (y - center[1]) ** 2 >= r_in ** 2),
                                ((x - center[0]) ** 2 + (y - center[1]) ** 2 <= r_out ** 2))
        mask[mask_area] = 1


        # apply mask and inverse DFT
        fshift = np.fft.fftshift(cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)) * mask

        fshift_mask_mag = 2000 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))

        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        return img_back
    def colorHPF(self,r):
        (R,G,B) = cv2.split(self.img)
        R = self.HPF(R,r)
        G = self.HPF(G,r)
        B = self.HPF(B,r)
        return cv2.merge([R/500000000,G/500000000,B/500000000])

    def colorLPF(self,r):
        (R,G,B) = cv2.split(self.img)
        R = self.LPF(R,r)
        G = self.LPF(G,r)
        B = self.LPF(B,r)
        return cv2.merge([R/500000000,G/500000000,B/500000000])

    def showHPFTransform(self,r):
        plt.imshow(self.HPF(self.grayImg,r))
        plt.title("Filtered above: "+str(np.abs(cv2.dft(np.float32(self.grayImg), flags=cv2.DFT_COMPLEX_OUTPUT)[r][r]))+"Hz")
        plt.show()
    def showLPFTransform(self,r):
        plt.imshow(self.LPF(self.grayImg,r))
        plt.title("Filtered below: "+str(np.abs(cv2.dft(np.float32(self.grayImg), flags=cv2.DFT_COMPLEX_OUTPUT)[r][r]))+"Hz")
        plt.show()
    def showColorHPFTransform(self,lowR,highR):
        f = plt.figure(figsize=(12, 12))
        inc = (highR-lowR)/10
        for i in range(0,10):
            f.add_subplot(2,5,i+1)
            plt.imshow(self.colorHPF(inc*i))
            plt.xticks(ticks=[])
            plt.yticks(ticks=[])
            plt.title("Filtered above: "+str(np.abs(cv2.dft(np.float32(self.grayImg), flags=cv2.DFT_COMPLEX_OUTPUT)[int(inc*i)][int(inc*i)]))+"Hz")
        plt.show()

    def showColorLPFTransform(self,lowR,highR):
        f = plt.figure(figsize=(12, 24))
        inc = (highR-lowR)/10
        for i in range(0,10):
            f.add_subplot(2,5,i+1)
            plt.imshow(self.colorLPF(inc*i))
            plt.xticks(ticks=[])
            plt.yticks(ticks=[])
            plt.title("Filtered below: "+str(np.abs(cv2.dft(np.float32(self.grayImg), flags=cv2.DFT_COMPLEX_OUTPUT)[int(inc*i)][int(inc*i)]))+"Hz")
        plt.show()

    def showImg(self):
        plt.imshow(self.img)
        plt.show()

    def compress_image(self, img, compression_factor):
        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        threshold = np.percentile(np.abs(dft_shift), compression_factor)
        dft_shift_compressed = dft_shift * (np.abs(dft_shift) > threshold)

        f_ishift = np.fft.ifftshift(dft_shift_compressed)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

        return img_back, dft_shift, dft_shift_compressed

    def compression_ratio(self, dft_shift, dft_shift_compressed):
        non_zero_before = np.count_nonzero(dft_shift)
        non_zero_after = np.count_nonzero(dft_shift_compressed)

        return non_zero_before / non_zero_after

    def show_compressed_image(self, compression_factor):
        (R, G, B) = cv2.split(self.img)
        compressed_R, dft_shift_R, dft_shift_compressed_R = self.compress_image(R, compression_factor)
        compressed_G, dft_shift_G, dft_shift_compressed_G = self.compress_image(G, compression_factor)
        compressed_B, dft_shift_B, dft_shift_compressed_B = self.compress_image(B, compression_factor)

        compressed_img = cv2.merge([compressed_R, compressed_G, compressed_B])

        # Normalize the pixel values to the range [0, 1]
        compressed_img_normalized = cv2.normalize(compressed_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                                  dtype=cv2.CV_32F)

        compression_ratio_R = self.compression_ratio(dft_shift_R, dft_shift_compressed_R)
        compression_ratio_G = self.compression_ratio(dft_shift_G, dft_shift_compressed_G)
        compression_ratio_B = self.compression_ratio(dft_shift_B, dft_shift_compressed_B)

        average_compression_ratio = (compression_ratio_R + compression_ratio_G + compression_ratio_B) / 3

        plt.imshow(compressed_img_normalized)
        plt.title(
            f"Compression Factor: {compression_factor}%\nAverage Compression Ratio: {average_compression_ratio:.2f}")
        plt.show()

Bonk = DFT('Bonk.jpeg')

#Bonk.showImg()
#Bonk.showHPFTransform(1)
#Bonk.showLPFTransform(1)
Bonk.showColorHPFTransform(0,10)

compression_factor = 99
Bonk.show_compressed_image(compression_factor)
