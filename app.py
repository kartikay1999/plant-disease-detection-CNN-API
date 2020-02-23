from flask import Flask, request, jsonify, render_template
from keras.models import load_model
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from werkzeug import secure_filename

model=load_model('model.h5')
app = Flask(__name__)

graph = tf.get_default_graph()
@app.route('/')
def home():
    return render_template('index.html')
  

@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))
        test_image=image.load_img(f.filename,target_size=(64,64))
        test_image=image.img_to_array(test_image)
        test_image=np.expand_dims(test_image,axis=0)
        with graph.as_default():
            y = model.predict_classes(test_image)
        d={0:'Applescab',1:'Blackrot',2:'Cedarapplerust',3:'healthy',4:'healthy',5:'Powderymildew',6:'healthy',7:'CercosporaleafspotGrayleafspot',8:'Commonrust',9:'NorthernLeafBlight',10:'healthy',11:'Blackrot',12:'Esca(BlackMeasles)',13:'Leafblight(IsariopsisLeafSpot)',14:'healthy',15:'Haunglongbing(Citrusgreening)',16:'Bacterialspot',17:'healthy',18:'Bacterialspot',19:'healthy',20:'Earlyblight',21:'Lateblight',22:'healthy',23:'healthy',24:'healthy',25:'Powderymildew',26:'Leafscorch',27:'healthy',28:'Bacterialspot',29:'Earlyblight',30:'Lateblight',31:'LeafMold',32:'Septorialeafspot',33:'SpidermitesTwo-spottedspidermite',34:'TargetSpot',35:'TomatoYellowLeafCurlVirus',36:'Tomatomosaicvirus',37:'healthy'}    
        dc={0: 'Apple',1: 'Apple',2: 'Apple',3: 'Apple',4: 'Blueberry',5: 'Cherry (including sour)',6: 'Cherry (including sour)',7: 'Corn (maize)', 8: 'Corn (maize)', 9: 'Corn (maize)',10: 'Corn (maize)',11: 'Grape',12: 'Grape', 13: 'Grape', 14: 'Grape', 15: 'Orange', 16: 'Peach', 17: 'Peach', 18: 'Pepper, bell',19: 'Pepper, bell',20: 'Potato',21: 'Potato',22: 'Potato', 23: 'Raspberry', 24: 'Soybean',25: 'Squash',26: 'Strawberry', 27: 'Strawberry', 28: 'Tomato',29: 'Tomato',30: 'Tomato',31: 'Tomato',32: 'Tomato',33: 'Tomato',34: 'Tomato',35: 'Tomato',36: 'Tomato',37: 'Tomato'}
        crop=dc[y[0]]
        dis=d[y[0]]
    return render_template("prediction.html",result = {'Crop':crop,'Disease':dis})
if __name__ == "__main__":
    app.run(debug=False)



