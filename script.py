from random import randint,random
import cv2 as cv
import numpy as np
from pathlib import Path

test_dir = Path('test')

# img = cv.imread(str(test_dir / 'Bedroom' / '5.jpg'))
# rows,cols,color = img.shape
# Affine Transformation
# pts1 = np.float32([[50,50],[200,50],[50,200]])
# pts2 = np.float32([[50 * random(),100 * random()],[200 * random(),50 * random()],[100 * random(),250 * random()]])
# M = cv.getAffineTransform(pts1,pts2)
# dst = cv.warpAffine(img,M,(cols,rows))
# cv.imshow('perspective_dst', dst)
# cv.waitKey(0)



for cls in test_dir.iterdir():
    for img_path in cls.iterdir():
        img = cv.imread(str(img_path))
        rows,cols,color = img.shape

        # 对图片做旋转变换
        # cols-1 and rows-1 are the coordinate limits.
        rotate_Ms = [cv.getRotationMatrix2D(((cols-1)/(random() * 10),(rows-1)/2.0),randint(0,180),randint(1,4)) for _ in range(1)]
        rotate_dst = [cv.warpAffine(img, M, (cols, rows)) for M in rotate_Ms]

        # 对图片做透视变换
        # pts1 = np.float32([[0,0],[cols-1,0],[0,rows-1],[cols-1,rows-1]])
        # pts2 = np.float32([[cols*random(),rows*random()],[cols*random(),rows*random()],[cols*random(),rows*random()],[cols*random(),rows*random()]])
        # perspective_M = cv.getPerspectiveTransform(pts1,pts2)
        # perspective_dst = cv.warpPerspective(img,perspective_M,(cols,rows))

        i = 0
        for dst in rotate_dst:
            cv.imwrite(str(cls / f'{img_path.name}_rotate{i}.jpg'), dst)
            i += 1
