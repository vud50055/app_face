import cv2
import os
from flask import Flask,request,render_template
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import shutil


app = Flask(__name__)


datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")



face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
try:
    cap = cv2.VideoCapture(1)
except:
    cap = cv2.VideoCapture(0)


if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv','w') as f:
        f.write('Name,ID,Check_in_time,Check_out_time,Total_time')
if not os.path.isdir(f'Attendance/Attendance_faces-{datetoday}'):
    os.makedirs(f'Attendance/Attendance_faces-{datetoday}')



def totalreg():
    return len(os.listdir('static/faces'))


def getusers():
    nameUsers = []
    idUsers = []
    l = len(os.listdir('static/faces'))
    for user in os.listdir('static/faces'):
        nameUsers.append(user.split('_')[0])
        idUsers.append(user.split('_')[1])
    return nameUsers, idUsers, l


def delUser(userid, username):
    for user in os.listdir('static/faces'):
        if user.split('_')[1] == userid:
            shutil.rmtree(f'static/faces/{username}_{userid}', ignore_errors=True)


def extract_faces(img):
    if img!=[]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.3, 5)
        return face_points
    else:
        return []


def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)



def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces,labels)
    joblib.dump(knn,'static/face_recognition_model.pkl')



def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['ID']
    inTimes = df['Check_in_time']
    outTimes = df['Check_out_time']
    totalTimes = df['Total_time']
    l = len(df)
    return names,rolls,inTimes,outTimes,totalTimes,l


def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')

    if int(userid) not in list(df['ID']):
        with open(f'Attendance/Attendance-{datetoday}.csv','a') as f:
            f.write(f'\n{username},{userid},{current_time},'',''')
    else:
        row_index = 0

        for i in range(0, df['ID'].count()):
            if str(df['ID'][i]) == userid:
                row_index = i
                break

        if str(df['Check_out_time'][row_index]) == 'nan':
            df.loc[row_index, 'Check_out_time'] = current_time

            inTime = (datetime.strptime(df['Check_in_time'][row_index], '%H:%M:%S'))
            outTime = (datetime.strptime(df['Check_out_time'][row_index], '%H:%M:%S'))

            totalTime = outTime - inTime

            df.loc[row_index, 'Total_time'] = totalTime

            df.to_csv(f'Attendance/Attendance-{datetoday}.csv', index=False)

   
def getUserTime(userid):
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    row_index = 0

    for i in range(0, df['ID'].count()):
        if str(df['ID'][i]) == userid:
            row_index = i
            break
            
    return str(df['Check_in_time'][row_index]), str(df['Check_out_time'][row_index])

def checkUserID(newuserid):
    listID = os.listdir('static/faces')
    for i in range(0, len(listID)):
        if listID[i].split('_')[1] == newuserid:
            return True
    return False



@app.route('/')
def home():
    names,rolls,inTimes,outTimes,totalTimes,l = extract_attendance()    
    return render_template('home.html',names=names,rolls=rolls,inTimes=inTimes,outTimes=outTimes,totalTimes=totalTimes,l=l,totalreg=totalreg(),datetoday2=datetoday2) 

@app.route('/listUsers')
def users():
    names, rolls, l = getusers()
    return render_template('ListUser.html', names= names, rolls=rolls, l=l)

@app.route('/deletetUser', methods=['POST'])
def deleteUser():
    userid = request.form['userid']
    username = request.form['username']
    delUser(userid, username)
    train_model()
    names, rolls, l = getusers()
    return render_template('ListUser.html', names= names, rolls=rolls, l=l)


@app.route('/start',methods=['GET'])
def start():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html',totalreg=totalreg(),datetoday2=datetoday2,mess='There is no trained model in the static folder. Please add a new face to continue.') 

    cap = cv2.VideoCapture(0)
    ret = True
    while ret:
        ret,frame = cap.read()
        if extract_faces(frame)!=():
            (x,y,w,h) = extract_faces(frame)[0]
            cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 20), 2)
            face = cv2.resize(frame[y:y+h,x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1,-1))[0]
            cv2.putText(frame,f'{identified_person}',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
        cv2.imshow('Attendance',frame)
        if cv2.waitKey(1)==27:
            break
    cap.release()
    cv2.destroyAllWindows()
    add_attendance(identified_person)

    print(identified_person)
    
    names,rolls,inTimes,outTimes,totalTimes,l = extract_attendance()    

    
    username = identified_person.split('_')[0]
    userid = identified_person.split('_')[1]
    userimagefolder = f'Attendance/Attendance_faces-{datetoday}/'+username+'_'+str(userid)+'_'+datetoday2
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    inTime, outTime = getUserTime(userid)

    print(inTime, outTime)
    if inTime != 'nan':
        name = username+'_'+userid+'_'+'checkin'+'.jpg'
        if name not in os.listdir(userimagefolder):
            cv2.imwrite(userimagefolder+'/'+name,frame[y:y+h,x:x+w])
    if outTime != 'nan':
        name = username+'_'+userid+'_'+'checkout'+'.jpg'
        if name not in os.listdir(userimagefolder):
            cv2.imwrite(userimagefolder+'/'+name,frame[y:y+h,x:x+w])

    return render_template('home.html',names=names,rolls=rolls,inTimes=inTimes,outTimes=outTimes,totalTimes=totalTimes,l=l,totalreg=totalreg(),datetoday2=datetoday2) 



@app.route('/add',methods=['GET','POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    if checkUserID(newuserid) == False:
        userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
        if not os.path.isdir(userimagefolder):
            os.makedirs(userimagefolder)
        cap = cv2.VideoCapture(0)
        i,j = 0,0
        while 1:
            _,frame = cap.read()
            faces = extract_faces(frame)
            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 20), 2)
                cv2.putText(frame,f'Images Captured: {i}/100',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
                if j%10==0:
                    name = newusername+'_'+str(i)+'.jpg'
                    cv2.imwrite(userimagefolder+'/'+name,frame[y:y+h,x:x+w])
                    i+=1
                j+=1
            if j==1000:
                break
            cv2.imshow('Adding new User',frame)
            if cv2.waitKey(1)==27:
                break
        cap.release()
        cv2.destroyAllWindows()
        print('Training Model')
        train_model()
        names,rolls,inTimes,outTimes,totalTimes,l = extract_attendance()    
        return render_template('home.html',names=names,rolls=rolls,inTimes=inTimes,outTimes=outTimes,totalTimes=totalTimes,l=l,totalreg=totalreg(),datetoday2=datetoday2) 
    else:
        names,rolls,inTimes,outTimes,totalTimes,l = extract_attendance()    
        return render_template('home.html',names=names,rolls=rolls,inTimes=inTimes,outTimes=outTimes,totalTimes=totalTimes,l=l,totalreg=totalreg(),datetoday2=datetoday2,mess='User ID has existed. Please type other ID.') 




if __name__ == '__main__':
    app.run(debug=True)
