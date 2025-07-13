import os
import cv2
import matplotlib.image as matimg
from app.face_recognition import faceRecognitionPipeline
from flask import render_template,request
UPLOAD_FOLDER='static/upload'


def index():
    return render_template('index.html')

def app():
    return render_template('app.html')
def genderapp():
    # Ensure upload and predict folders exist
    os.makedirs('static/upload', exist_ok=True)
    os.makedirs('static/predict', exist_ok=True)
    if request.method == 'POST':
        if 'image_name' not in request.files:
            return render_template('gender.html', error='No file part')
        f = request.files['image_name']
        if f.filename == '':
            return render_template('gender.html', error='No selected file')
        filename = f.filename
        path = os.path.join(UPLOAD_FOLDER, filename)
        f.save(path)
        try:
            pred_image, predictions = faceRecognitionPipeline(path)
            pred_filename = 'prediction_image.jpg'
            cv2.imwrite(f'./static/predict/{pred_filename}', pred_image) 
            #print(predictions)
            #generate report
            report=[]
            for i,obj in enumerate(predictions):
                gray_image=obj['roi'] #grayscale image
                eigen_image=obj['eig_img'].reshape(100,100) #eigen image
                gender_name=obj['prediction_name'] #name
                score=round(obj['prob_score']*100,2) #probability score
                #save gray scale and eigen in preict folder
                gray_image_name=f'roi_{i}.jpg'
                eig_image_name=f'eigen_{i}.jpg'
                matimg.imsave(f'./static/predict/{gray_image_name}',gray_image,cmap='gray')
                matimg.imsave(f'./static/predict/{eig_image_name}',eigen_image,cmap='gray')
                #save report
                report.append([gray_image_name,
                eig_image_name,
                gender_name,
                score])
                    
                
            return render_template('gender.html', fileupload=True,report=report) #POSTREQUEST
        except Exception as e:
            print(f'Error in prediction: {e}')
            return render_template('gender.html', error=str(e))
    return render_template('gender.html',fileupload=False)   #GETREQUEST