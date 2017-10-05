"""Binarize (make it black and white) an image with Pyhton."""

from PIL import Image
import pytesseract

def ocr(img):
    txt = pytesseract.image_to_string(img)#,lang="eng")

    return txt

def Resize(img,fac=1.2):
    ''' Resize PIL Image

        img: Image
            Pil Image to resize
        fac: float or int, default 1.2
            Value to resize image in times
        Output:
            Pil Image resized
    '''
    w,h=img.size
    h = int(round(h*fac))
    w = int(round(w*fac))

    img = img.resize((w, h), Image.ANTIALIAS)

    return img

def BinarizeImage(fname,threshold=220):
    """ Binarize an image.
        fname: str
            Image filename
        threshold: int, default 200
            Threshold value pixel to converts to black

        output:
            PIL image with black and white values
    """
    col = Image.open(fname)
    gray = col.convert('L')
    bw = gray.point(lambda x: 0 if x<threshold else 255, '1')

    return bw

if __name__ == "__main__":

    fname= 'memb_inv1.png'
    bw = BinarizeImage(fname)
    bw = Resize(bw,fac=2)
    #bw.save(fname.replace('.png','_new2.png'), quality=95)
    txt = ocr(bw)
    print(txt)
