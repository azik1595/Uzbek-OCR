from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog,messagebox
from ttk import Frame, Label
from fuzzy import *
from wordSegmentation import wordSegmentation, segmentation
import configparser
import cv2
import tempfile
import numpy as np

classification = np.loadtxt("classifications.txt", np.float32)
flattened_image_features = np.loadtxt("flattenedImageFeatures.txt", np.float32)

# Reshape
classification = classification.reshape((classification.size, 1))

# KNN Training
kNearest = cv2.ml.KNearest_create()
print(cv2.ml.ROW_SAMPLE)
kNearest.train(flattened_image_features, cv2.ml.ROW_SAMPLE, classification)
def recognition(charCandidate, charThreshold):
    # Find Contours
    cnts = cv2.findContours(charCandidate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[1]

    boxes = []
    # Get all boxes
    for c in cnts:
        (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)
        boxes.append(((boxX, boxY, boxX + boxW, boxY + boxH)))

    # Order boxes from left to right
    boxes = sorted(boxes, key=lambda b: b[0])

    # Initialize Thresholds
    resize_w = 20
    resize_h = 30

    plate_number = ""

    for (startX, startY, endX, endY) in boxes:
        current_char = charThreshold[startY:endY, startX:endX]
        cv2.imshow("character", current_char)
        cv2.waitKey(0)

        cur_char_resize = cv2.resize(current_char, (resize_w, resize_h))
        cur_char_reshape = cur_char_resize.reshape((1, resize_w * resize_h))
        cur_feature = np.float32(cur_char_reshape)

        _, result, _, _ = kNearest.findNearest(cur_feature, k=1)

        plate_char = str(chr(int(result[0][0])))
        plate_number += plate_char

    print("Detection Completed:")
    print(plate_number)

    return plate_number
IMAGE_SIZE = 1800
BINARY_THREHOLD = 180

config = configparser.ConfigParser()
config.read('conf.ini')


def process_image_for_ocr(file_path):
    #temp_filename = set_image_dpi(file_path)
    im_new = remove_noise_and_smooth(file_path)
    return im_new


def set_image_dpi(file_path):
    im = Image.open(file_path)
    length_x, width_y = im.size
    factor = max(1, int(IMAGE_SIZE / length_x))
    size = factor * length_x, factor * width_y
    # size = (1800, 1800)
    im_resized = im.resize(size, Image.ANTIALIAS)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_filename = temp_file.name
    im_resized.save(temp_filename, dpi=(300, 300))
    return temp_filename


def image_smoothening(img):
    ret1, th1 = cv2.threshold(img, BINARY_THREHOLD, 255, cv2.THRESH_BINARY)
    ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(th2, (1, 1), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3


def remove_noise_and_smooth(file_name):
    img = cv2.imread(file_name, 0)
    filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 3)
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    img = image_smoothening(img)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    or_image = cv2.bitwise_or(img, closing)
    return or_image

def rasim(*w,**kw):
    root = Tk()
    root.title('Rasim')

    class Fram(Frame):
        def __init__(self, parent):
            Frame.__init__(self, parent)
            self.parent = parent
            self.initUI()

        def initUI(self):
            self.parent.title("Uzbek yozuvi tanish")
            self.pack(fill=BOTH, expand=True)
            self.columnconfigure(1, weight=1)
            self.columnconfigure(3, pad=7)
            self.rowconfigure(5, weight=1)
            self.rowconfigure(5, pad=7)
            e1_v = StringVar()
            e2_v = StringVar()

            def save(e1 ,e2 ):
                config.set('rasim', 'h', e1.get())
                config.set('rasim', 'w', e2.get())
                with open('conf.ini', 'w') as configfile:
                    config.write(configfile)
            Label(self, text="Uzinligi").grid(row=1)
            Label(self, text="Eni").grid(row=2)
            e1 = Entry(self,textvariable=e1_v)
            e2 = Entry(self,textvariable=e2_v)
            e1.insert(END,config['rasim']['h'])
            e2.insert(END, config['rasim']['w'])
            # e1.set( "значение по умолчанию" )
            # e1_text = e1.get()
            # e2.set("значение по умолчанию")
            # e2_text = e1.get()
            e1.grid(row=1, column=1)
            e2.grid(row=2, column=1)
            Button(self, text='Yopish', command=butCallback).grid(row=3, column=0, sticky=W, pady=4)
            Button(self, text='Saqlash',command=lambda : save(e1,e2)).grid(row=3, column=2, sticky=W, pady=4)

    def butCallback():
        root.destroy()
    app = Fram(root)
    root.mainloop()
def segment(*w,**kw):
    root = Tk()
    root.title('Segmentatsiya')

    class Fram(Frame):
        def __init__(self, parent):
            Frame.__init__(self, parent)
            self.parent = parent
            self.initUI()

        def initUI(self):
            self.parent.title("Uzbek yozuvi tanish")
            self.pack(fill=BOTH, expand=True)
            self.columnconfigure(1, weight=1)
            self.columnconfigure(3, pad=7)
            self.rowconfigure(5, weight=1)
            self.rowconfigure(5, pad=7)
            e1_v = StringVar()
            e2_v = StringVar()
            e3_v = StringVar()
            e4_v = StringVar()
            def save(e1 ,e2 ,e3,e4):
                config.set('segment', 'kernelSize', e1.get())
                config.set('segment', 'sigma', e2.get())
                config.set('segment', 'teta', e3.get())
                config.set('segment', 'minArea', e4.get())
                with open('conf.ini', 'w') as configfile:
                    config.write(configfile)

            Label(self,text='kernelSize: filtri yadrosining kattaligi yagona obyekniki bo\'lishi kerak.\n'+
                            'sigma: filtr yadrosi uchun ishlatiladigan Gauss funktsiyasining standart sozlamasi.\n'+
                            'teta: so\'zlarning taxminan kengligi / balandligi nisbati, filtr vazifasi bu omilga ko\'ra buziladi.\n'+
                            'minArea: nomdagi nomzodlarni belgilangan maydondan kichikroq deb hisoblang.').grid(row=0)
            Label(self, text="kernelSize").grid(row=2)
            Label(self, text="sigma").grid(row=3)
            Label(self, text="teta").grid(row=4)
            Label(self, text="minArea").grid(row=5)
            e1 = Entry(self,textvariable=e1_v)
            e2 = Entry(self,textvariable=e2_v)
            e3 = Entry(self, textvariable=e3_v)
            e4 = Entry(self, textvariable=e4_v)
            e1.insert(END,config['segment']['kernelSize'])
            e2.insert(END, config['segment']['sigma'])
            e3.insert(END, config['segment']['teta'])
            e4.insert(END, config['segment']['minArea'])
            e1.grid(row=2, column=1)
            e2.grid(row=3, column=1)
            e3.grid(row=4, column=1)
            e4.grid(row=5, column=1)
            Button(self, text='Yopish', command=butCallback).grid(row=6, column=0, sticky=W, pady=4)
            Button(self, text='Saqlash',command=lambda : save(e1,e2,e3,e4)).grid(row=6, column=2, sticky=W, pady=4)

    def butCallback():
        root.destroy()
    app = Fram(root)
    root.mainloop()


def fuzzy_conf(*w,**kw):
    root = Tk()
    root.title('Fuzzy')
    class Fram(Frame):
        def __init__(self, parent):
            Frame.__init__(self, parent)
            self.parent = parent
            self.initUI()

        def initUI(self):
            self.parent.title("Uzbek yozuvi tanish")
            self.pack(fill=BOTH, expand=True)
            self.columnconfigure(1, weight=1)
            self.columnconfigure(3, pad=7)
            self.rowconfigure(5, weight=1)
            self.rowconfigure(5, pad=7)
            e1_v = StringVar()
            e2_v = StringVar()
            e3_v = StringVar()
            e4_v = StringVar()
            e5_v = StringVar()
            e6_v = StringVar()
            e7_v = StringVar()


            def save(e1 ,e2,e3,e4,e5,e6,e7 ):
                config.set('fuzzy',"n_clusters", e1.get())
                config.set('fuzzy',"m", e2.get())
                config.set('fuzzy',"kernel_size", e3.get())
                config.set('fuzzy',"kernel_shape", e4.get())
                config.set('fuzzy',"lam", e5.get())
                config.set('fuzzy',"epsilon", e6.get())
                config.set('fuzzy',"max_iter", e7.get())
                with open('conf.ini', 'w') as configfile:
                    config.write(configfile)

            Label(self, text='<n_clusters>: int, yaratish uchun guruhlar / segmentlar soni.\n' +
                             '<m>: float> 1,noaniqlik parametri. Odatda 2 ga sozlangan.\n' +
                             '<kernel_size>: int> = 1, element hajmi.\n' +
                             '<kernel_shape>: str, iform: teng ravishda ogirlikdagi yadro funktsiyasi yordamida og\'irliklarni yig\'ish.' +
                             '\"Gauss\": yonidagilari yig\'ish uchun Gaussning og\'irliklari.\n' +
                             '<lam>: float> 0, intuitiv noaniq parametri\n' +
                             '<epsilon>:<float> 0, yaqinlikni tekshiruvi uchun chegara.\n' +
                             '<max_iter>: int, maksimal izdoshlar soni.').grid(row=1)


            Label(self, text="n_clusters").grid(row=2, column= 0)
            Label(self, text="m").grid(row=3,column= 0 )
            Label(self, text="kernel_size").grid(row=4, )
            Label(self, text="kernel_shape").grid(row=5, )
            Label(self, text="lam").grid(row=6, )
            Label(self, text="epsilon").grid(row=7, )
            Label(self, text="max_iter").grid(row=8, )
            e1 = Entry(self,textvariable=e1_v)
            e2 = Entry(self,textvariable=e2_v)
            e3 = Entry(self, textvariable=e3_v)
            e4 = Entry(self, textvariable=e4_v)
            e5 = Entry(self, textvariable=e5_v)
            e6 = Entry(self, textvariable=e6_v)
            e7 = Entry(self, textvariable=e7_v)

            e1.insert(END,config['fuzzy']['n_clusters'])
            e2.insert(END, config['fuzzy']['m'])
            e3.insert(END, config['fuzzy']['kernel_size'])
            e4.insert(END, config['fuzzy']['kernel_shape'])
            e5.insert(END, config['fuzzy']['lam'])
            e6.insert(END, config['fuzzy']['epsilon'])
            e7.insert(END, config['fuzzy']['max_iter'])
            # e1.set( "значение по умолчанию" )
            # e1_text = e1.get()
            # e2.set("значение по умолчанию")
            # e2_text = e1.get()
            e1.grid(row=2, column=1)
            e2.grid(row=3, column=1)
            e3.grid(row=4, column=1)
            e4.grid(row=5, column=1)
            e5.grid(row=6, column=1)
            e6.grid(row=7, column=1)
            e7.grid(row=8, column=1)
            Button(self, text='Yopish', command=butCallback).grid(row=10, column=0, sticky=W, pady=4)
            Button(self, text='Saqlash',command=lambda : save(e1,e2,e3,e4,e5,e6,e7)).grid(row=10, column=3, sticky=W, pady=4)

    def butCallback():
        root.destroy()

    # but1 = Button(root,
    #                     text ="-> window 1",
    #                     command = butCallback )
    # but1.pack()
    # text1 = Text(root)
    # text1.pack()
    app = Fram(root)
    root.mainloop()
def change_model():
        root = Tk()
        root.title('Model')
        class Fram(Frame):
            def __init__(self, parent):
                Frame.__init__(self, parent)
                self.parent = parent
                self.initUI()

            def initUI(self):
                self.parent.title("Uzbek yozuvi tanish")
                self.pack(fill=BOTH, expand=True)
                self.columnconfigure(1, weight=1)
                self.columnconfigure(3, pad=7)
                self.rowconfigure(5, weight=1)
                self.rowconfigure(5, pad=7)
                e1_v = StringVar()

                def openfn(e1):
                    fn = filedialog.askopenfilename(title='open')
                    e1.delete(0, 'end')
                    e1.insert(END, fn)
                def save(e1):
                    config.set('model', 'path', e1.get())
                    with open('conf.ini', 'w') as configfile:
                        config.write(configfile)

                Label(self, text="Model turgan joy").grid(row=1)
                e1 = Entry(self, textvariable=e1_v)
                e1.insert(END, config['model']['path'])
                e1.grid(row=1, column=1)
                Button(self, text='FAYL', command=lambda: openfn(e1)).grid(row=1, column=2, sticky=W, pady=4)
                Button(self, text='Yopish', command=butCallback).grid(row=3, column=0, sticky=W, pady=4)
                Button(self, text='Saqlash', command=lambda: save(e1)).grid(row=3, column=2, sticky=W, pady=4)



        def butCallback():
            root.destroy()

        app = Fram(root)
        root.mainloop()

root = Tk()
menu = Menu(root)
root.config(menu=menu)
filemenu = Menu(menu)
root.geometry("1100x600+300+150")
root.resizable(width=True, height=True)
menu.add_cascade(label='Konfiguratsiya', menu=filemenu)
filemenu.add_command(label='Rasim',command=rasim)
filemenu.add_command(label='Fuzzy',command= fuzzy_conf)
filemenu.add_command(label='Segmentatsiya',command=segment)
filemenu.add_command(label='Modelni uzgartirish',command=change_model)
filemenu.add_separator()
filemenu.add_command(label='Chiqish', command=root.quit)
helpmenu = Menu(menu)
menu.add_cascade(label='Yordam', menu=helpmenu)
helpmenu.add_command(label='Programma xaqida')
filename=None
words=[]
result_string = ''
def openfn():
    global filename
    filename= filedialog.askopenfilename(title='open')
    return filename

def open_img(label1,lbl):
    x = openfn()
    if x:
     img = Image.open(x)
     img = img.resize((int(config['rasim']['h']), int(config['rasim']['w'])), Image.ANTIALIAS)
     img = ImageTk.PhotoImage(img)
     label1.configure(image=img)
     label1.image = img
     lbl.configure(text="Fayl:"+x)

def fuzzy(label1,lbl):
    fuzzy_filter(filename,config)

    image = process_image_for_ocr('tmp\prep.jpg')

    cv2.imwrite('tmp\prep2.jpg',image)
    img =  Image.fromarray(image)
    img = img.resize((int(config['rasim']['h']), int(config['rasim']['w'])), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    label1.configure(image=img)
    label1.image = img
    lbl.configure(text="Siz fuzzy algoritmini bajrdingiz")

def segmentatsiya(label1,lbl):
        img = cv2.imread('tmp/prep2.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #prepareImg(cv2.imread('tmp/prep2.jpg'), 500)
       # cv2.imwrite('tmp/seg')
        res = wordSegmentation(img,
                               kernelSize=int(config['segment']['kernelSize']),
                               sigma =int(config['segment']['sigma']),
                               theta=int(config['segment']['teta']),
                               minArea=int(config['segment']['minArea']))
        print('Segmented into %d words' % len(res))
        for (j, w) in enumerate(res):
            (wordBox, wordImg) = w
            (x, y, w, h) = wordBox
            cv2.imwrite('tmp/%d.png' % (j), wordImg)
            words.append({'image':'tmp/%d.png'% (j),'x':x,'y':y})
            cv2.rectangle(img, (x, y), (x + w, y + h), (0,0,0))
        cv2.imwrite('tmp/segmented.jpg',img)
        img = Image.fromarray(img)
        img = img.resize((int(config['rasim']['h']), int(config['rasim']['w'])), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        label1.configure(image=img)
        label1.image = img
        lbl.configure(text="Siz rasimdagi suzlarni segmentatsiya qildingiz")
def tanish(label,lbl):
    pass
     # img = cv2.imread('tmp/tmp.png')
     # tests_set =[]
     # for w in words:
     #   tests_set.append(w['image'])
     # output = estimator.predict(input_fn(tests_set, mode=tf.estimator.ModeKeys.PREDICT),
     #                       checkpoint_path=LOG_DIR + 'model.'+config['model']['path'].split('.')[1])
     # global result_string
     # result_string = ''
     # for i,out in enumerate(output):
     #    if float(out['probabilities'])>0.20:
     #     text  =out['prediction']
     #     word = words[i]
     #     result_string=result_string+' '+words_dict[str(text)]
     #     cv2.putText(img,words_dict[str(text)], (int(word['x']), int(word['y'])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
     #                 cv2.LINE_AA)
     #
     # cv2.imwrite('tmp/seg.jpg', img)
     # img = Image.fromarray(img)
     # img = ImageTk.PhotoImage(img)
     # label.configure(image=img)
     # label.image = img
     # lbl.configure(text="Siz rasimdagi suzlarni aniqladingiz")

     # root = Toplevel()
     # text1 = Text(root, height=20, width=30)
     # image = Image.open("tmp/seg.jpg")
     # image = image.resize((300, 200), Image.ANTIALIAS)
     # photo = ImageTk.PhotoImage(image,size=(20,30))
     # text1.insert(END, '\n')
     # text1.image_create(END, image=photo)
     #
     # text1.pack(side=LEFT)
     #
     # text2 = Text(root, height=20, width=50)
     # scroll = Scrollbar(root, command=text2.yview)
     # text2.configure(yscrollcommand=scroll.set)
     # text2.tag_configure('bold_italics', font=('Arial', 12, 'bold', 'italic'))
     # text2.tag_configure('big', font=('Verdana', 20, 'bold'))
     # text2.tag_configure('color', foreground='#476042',
     #                     font=('Tempus Sans ITC', 12, 'bold'))
     # text2.insert(END, result_string, 'color')
     # text2.pack(side=LEFT)
     # scroll.pack(side=RIGHT, fill=Y)
     # root.mainloop()



class Main(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.parent = parent
        self.initUI()
    def initUI(self):
        self.parent.title("Uzbek yozuvi tanish")
        self.pack(fill=BOTH, expand=True)

        self.columnconfigure(1, weight=1)
        self.columnconfigure(3, pad=7)
        self.rowconfigure(5, weight=1)
        self.rowconfigure(5, pad=7)

        lbl = Label(self, text="Rasim")
        lbl.grid(sticky=W, pady=4, padx=5)


        label1 = Label(self)
        label1.grid(row=1, column=0, columnspan=2, rowspan=4,
                   padx=5, sticky=E + W + S + N, )
        # area = Text(self)
        # area.grid(row=1, column=0, columnspan=2, rowspan=4,
        #           padx=5, sticky=E + W + S + N)

        abtn = Button(self, text="1.Rasim ochish" ,width =15, command=lambda : open_img(label1,lbl))
        abtn.grid(row=1,  column=3,sticky=E + W + S + N,  padx=2, pady=2,)

        cbtn = Button(self, text="2.Fuzzy algoritmi",width =15, command= lambda : fuzzy(label1,lbl))
        cbtn.grid(row=2, column=3, sticky=E + W + S + N, padx=2, pady=2, )

        cbtn2 = Button(self, text="3.Segmentatsiya",width =15, command=lambda: segmentatsiya(label1,lbl))
        cbtn2.grid(row=3, column=3, sticky=E + W + S + N, padx=2, pady=2, )

        cbtn3 = Button(self, text="4.Tanish",width =15, command=lambda: tanish(label1,lbl))
        cbtn3.grid(row=4, column=3,  padx=2, pady=2 ,sticky=E + W + S + N, )


def callback():
    if messagebox.askokcancel("Chiqish", "Programmani yopasizmi?"):
        root.destroy()
root.protocol("WM_DELETE_WINDOW", callback)
root.title("Click the Button")
app = Main(root)
root.mainloop()
