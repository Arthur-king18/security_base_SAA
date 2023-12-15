import os

import face_recognition
import cv2
import numpy
import numpy as np
from datetime import datetime

import pygame


artur_image = face_recognition.load_image_file("ImagesAttendance/artur.jpg")
artur_face_encoding = face_recognition.face_encodings(artur_image)[0]

known_face_encodings = [
    artur_face_encoding

]
known_face_names = [
    "Artur"
]


video_capture = cv2.VideoCapture(0)

face_locations = []
face_encodings = []
face_names = []


def main():
    process_this_frame = True
    while True:
        ret, frame = video_capture.read()

        # Обрабатывает только каждый второй кадр видео, чтобы сэкономить время
        if process_this_frame:
            # Изменяет размер кадра видео до 1/4 для более быстрой обработки распознавания лиц.
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Преобразование изображения из цвета BGR (который использует OpenCV) в цвет RGB (который использует face_recognition)
            rgb_small_frame = rgb_small_frame = numpy.ascontiguousarray(small_frame[:, :, ::-1])

            # Находит все лица и кодировки лиц в текущем кадре видео.
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # Проверяет, соответствует ли лицо известному лицу (лицам)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # Или вместо этого использует известное лицо с наименьшим расстоянием до нового лица.
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)
                if name != 'Unknown':
                    print(f'Обнаружен(а) {name} ')
                    markAttendance(name=name)

                else:
                    markAttendance(name=name)
                    cv2.imwrite(f"{datetime.utcnow()}.jpg", frame)
                    print('Неизвестный обЪект')
                    pygame.mixer.init()
                    pygame.mixer.music.load("c7b68d9df6de3c5.mp3")
                    pygame.mixer.music.play()

        process_this_frame = not process_this_frame

        # Отображение результатов - только для показа. На микрокомпьютере не запускаю функци для экономии ресурсов
        # for (top, right, bottom, left), name in zip(face_locations, face_names):
        #     # Масштабируем обратное расположение лиц, поскольку обнаруженный нами кадр был уменьшен до 1/4 размера.
        #     top *= 4
        #     right *= 4
        #     bottom *= 4
        #     left *= 4
        #
        #     # Нарисуем рамку вокруг лица
        #     cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        #
        #     # Нарисуем метку с именем под лицом
        #     cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        #     font = cv2.FONT_HERSHEY_DUPLEX
        #     cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        #
        # # Отобрази полученное изображение
        # cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(name)
        if name not in nameList:
            now = datetime.utcnow()
            f.writelines(f'\n{name} - {now}')


if __name__ == '__main__':
    main()

