'''
This class can be used to feed input from an image, webcam, or video to your model.
Sample usage:
    feed=InputFeeder(input_type='video', input_file='video.mp4')
    feed.load_data()
    for batch in feed.next_batch():
        do_something(batch)
    feed.close()
'''
import cv2


class InputFeeder:
    def __init__(self, input_type, input_file = None):
        '''
        input_type: str, The type of input. Can be 'video' for video file, 'image' for image file,
                    or 'cam' to use webcam feed.
        input_file: str, The file that contains the input image or video file. Leave empty for cam input_type.
        '''        
        self.input_type = input_type
        if input_type == 'video' or input_type == 'image': #if any of these?
            self.input_file = input_file #then a file is required
    
    def load_data(self):
        if self.input_type == 'video': #use an mp4 file
            self.cap = cv2.VideoCapture(self.input_file)
        elif self.input_type == 'cam': #use a camera stream
            self.cap = cv2.VideoCapture(0)
        else:
            self.cap = cv2.imread(self.input_file) #use an image



    def next_batch(self):
        '''
        Returns the next image from either a video file or webcam.
        If input_type is 'image', then it returns the same image.
        '''
        while True:
            flag = False #flag in case an image is used
            for _ in range(1):
                flag, frame = self.cap.read()
            
            if not flag:
                break
            yield frame



    def close(self):
        if not self.input_type == 'image':
            self.cap.release()

