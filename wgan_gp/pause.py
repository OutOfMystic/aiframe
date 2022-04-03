import os
import sys
import time

s_path = os.path.dirname(os.getcwd())
sys.path.insert(0, s_path)
import manage.pause as pause

if __name__ == '__main__':
    pause.pause(65400)
    print('PAUSED, braza')
    for i in range(5, 0, -1):
        print('Closting in...', i)
        time.sleep(1)