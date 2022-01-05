############### Import Libraries ###############
from pynput.keyboard import Key, Listener    # keyboard handler for key press
from threading import Thread
from datetime import datetime
import time
import warnings
warnings.simplefilter("ignore", UserWarning)

############### Import Modules ###############
from src import file_read_write
from src import constant as c

################# Class ################
class keyboard_read:
    """
    constructor: initialize related variables
    """
    def __init__(self):
        self.thread = None
        self.label = ''
        self.writeLock = False      # True when csv file is writing and lock any keyboard input
        self.liveRecordOn = False   # True when recording live input
        self.predictionOn = False

    """
    keybaord_start_thread:
    start a new thread for listening to keyboard input
    """
    def keybaord_start_thread(self):
        if self.thread == None:
            self.thread = Thread(target=self.background_thread)
            self.thread.start()

    """
    Keyboard listener 
    """
    def background_thread(self):
        ############ Key Release ############
        def on_release(key):
            if self.writeLock == False:
                self.label = ''     # initialize gesture label
                # str(key) return a string '1' with length of 3 characters
                # str(key)[1] return the second character and remove the pair of bucket "'"
                input = str(key)[1]
                if input in c.INPUT_KEYS:
                    idx = c.INPUT_KEYS.index(input)
                    self.label = c.LABELS[idx]
                    time.sleep(c.DELAY_TIME_BEFORE_RECORDING)
                    self.trigger_record()
                elif (key == Key.space or key== Key.enter) and c.IS_AUTO_PREDICT == False:
                    self.trigger_predict()
                else:
                    print("Invalid key prssed: ", key)
        ############# Key Press #############
        #def on_press(key):
            #do something
        ######### Initiated Listener ########
        with Listener(
                #on_press=on_press,
                on_release=on_release) as listener:
            listener.join()

    """
    trigger_record:
    run once an input key is pressed for recording acoustic data (for WAIT_TIME second)
    """
    def trigger_record(self):
        print('Record start time" ', datetime.now(), '   Label: ', self.label)  # print timestamp
        self.liveRecordOn = True
        self.writeLock = True   #lock keyboard from input until recording is finished
        file_read_write.recorded_chunk_num = 0  # reset before recording

    """
    trigger_predict:
    run once an input key (spacebar) is pressed for recording prediction data (for WAIT_TIME second)
    """
    def trigger_predict(self):
        self.predictionOn = True
        file_read_write.recorded_chunk_num = 0  # reset before recording


    """
    close:
    terminate the thread when the program end
    """
    def close(self):
        self.thread.join()  # close the thread for keyboard reading