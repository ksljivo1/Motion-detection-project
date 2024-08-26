import cv2
import numpy as np

#putanje do video fajlova
file1 = 'C:\\Users\\sljivo\\Desktop\\input\\video_1.mp4'
file2 = 'C:\\Users\\sljivo\\Desktop\\input\\video_2.mp4'
file3 = 'C:\\Users\\sljivo\\Desktop\\input\\video_3.mp4'
cam = 0

#funkcija koja vraca pozadinski model videa
def get_background(file_path):
    #ucitavamo video
    v = cv2.VideoCapture(file_path);
    if not v.isOpened():
        print('Neuspjesno ucitavanje videa')
        exit()
    #uniform vraca 50 nasumicnih vrijednosti u intervalu [0, 1),
    #pa ih mnozimo sa ukupnim brojem frejmova da dobijemo 50 nasumicnih indeksa frejmova
    rand = (v.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size = 50)).astype(int)
    frames = []
    #unosimo nasumicno odabrane frejmove u listu frames
    for i in rand:
        v.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = v.read()
        frames.append(frame)
    #vracamo medijanu nasumicno odabranih frejmova
    return np.median(frames, axis = 0).astype(np.uint8)

file = file1

#konvertujemo pozadinski model u grayscale format
background = cv2.cvtColor(get_background(file), cv2.COLOR_BGR2GRAY)

v = cv2.VideoCapture(file);

frame_count = 0
consecutive_frame = 5
frame_diff_list = []

while v.isOpened():
    ret, original_frame = v.read()
    if not ret:
        break
    frame_count += 1
    frame = original_frame.copy()
    #konvertujemo ucitani frejm u grayscale format
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #nakon 5 ucitanih frejmova praznimo listu frame_diff_list
    if frame_count % consecutive_frame == 0:
        frame_diff_list = []
    #racunamo apsolutnu vrijednost razlike tekuceg frejma i pozadinskog modela
    frame_diff = cv2.absdiff(gray, background)
    #potom je konvertujemo u binarni format
    ret, thresh = cv2.threshold(frame_diff, 50, 255, cv2.THRESH_BINARY)
    #koristimo morfolosku operaciju dilate da spojimo odvojene dijelove istog objekta u pokretu
    dilate_frame = cv2.dilate(thresh, None, iterations = 2)
    #i na kraju dodajemo u listu frame_diff_list
    frame_diff_list.append(dilate_frame)
    if len(frame_diff_list) == consecutive_frame:
        #sumiramo 5 uzastopnih frejmova 
        """
        moze i ovo umjesto sum:
        sum_of_frames = frame_diff_list[0]
        for fr in frame_diff_list:
            sum_of_frames = cv2.bitwise_or(sum_of_frames, fr)
        """
        sum_of_frames = sum(frame_diff_list)
        #nalazimo konture detektovanih objekata
        contours, hierarchy = cv2.findContours(sum_of_frames, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            #zanemarujemo konture koje zauzimaju povrsinu manju od 500, ovaj korak filtrira šum
            if cv2.contourArea(contour) < 500:
                continue
            #ova metoda vraca koordinate pravougaonika koji ogranicava pronadjenu konturu
            x, y, w, h = cv2.boundingRect(contour)
            #zatim taj pravougaonik crtamo na izvorni frejm
            cv2.rectangle(original_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        #prikazujemo frejm sa iscrtanim pravougaonicima
        cv2.imshow('Detected Objects', original_frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
v.release()
cv2.destroyAllWindows()
