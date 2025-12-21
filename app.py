
import os
from forms import   contactform,  transform
import tensorflow_hub as hub
from pprint import pprint
from flask import Flask, request, render_template, redirect, flash
from email_function import gmail_email
import io
from tensorflow.keras.models import load_model
application = Flask(__name__)
application.config['SECRET_KEY'] = 'iuhguivfduivdifvjdfhv'
@application.route('/', methods=['GET','POST'])
def predict_func():
    form = transform()
 
    if form.is_submitted():
        result = request.form
        print(form.image.data)
        print(result)
  

        file = request.files.get("image")
        
       
        fi = file.filename
        l = fi.split('.')
        print(l)
        model=load_model('96_ac_0.1_Loss_covidmodel.h5', custom_objects={'KerasLayer': hub.KerasLayer})
        import numpy as np
      
       
        from keras.preprocessing import image
        file_bytes = io.BytesIO(file.read())
        xtest_image = image.image_utils.load_img(file_bytes, target_size = (299, 299))
        xtest_image = image.image_utils.img_to_array(xtest_image)
        xtest_image = np.expand_dims(xtest_image, axis = 0)
        results = model.predict(xtest_image)
        
        print(results)
        # cv2_imshow(imggg)
        


        {'COVID19': 0, 'MARILA': 1, 'NORMAL': 2, 'PNEUMONIA': 3, 'TURBERCULOSIS': 4}
        
        if l[3] == 'png':
            if results[0][0]==0:
                results='We are sorry u have covid-19'
                return render_template('result.html',res=results)
            
                
            if results[0][0]==1:
                results='Horay u are normal'
                return render_template('result.html',res=results)
          
            
        if l[3] == 'jpg':
            
            
           if results[0][0]==0:
                results='We are sorry u have covid-19'
                return render_template('result.html',res=results)
            
                
           if results[0][0]==1:
                results='Horay u are normal'
                return render_template('result.html',res=results)
        else:
            flash("jpg and png file only supported","info")
        
        

            
            
        
    return render_template('index.html',form=form)
@application.route('/feedback', methods=['GET','POST'])
def feedback_func():
    form = contactform()

    if form.is_submitted():
        result = request.form
        
        # with open('database.txt', mode='a') as database:
        #     email = result['email']
        #     name = result['name']
        #     body = result['body']
        #     file = database.write(f'\n{email},{name},{body} ')
       
        gmail_email('thayyilsuhaan@gmail.com',f"feedback from translater from: {result['name']},{result['email']},",result['body'])
    
        return render_template('rf.html')
    return render_template('feedback.html',form=form)
@application.route('/logs')
def logs_func():
    return render_template('logs.html')
@application.route('/ca')
def Ca_func():
    return render_template('ca.html')
@application.route('/about')
def Aboutpage():
    return render_template('about.html')
@application.route('/pp')
def pp_page():
    return render_template('pp.html')
@application.route('/robots.txt')
def r():
    return render_template('robots.html')
if __name__ == '__main__':
    application.run(debug=True)
