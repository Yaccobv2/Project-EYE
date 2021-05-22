# Project-EYE
Face mask detection system using deep learning and RaspberryPi.

# Purpose of the project
The project is intended to detect people who are not wearing facemask protection.

# How does it work?
When system detects a person without mask takes and saves a picture in database directory, it also creates record in MySQL database in format: id, date, directory link. There are also developer features like clearing whole database and directory after clicking the key "c", printing in console every database record after clicking the key "s", quiting program after clicking the key "q". Those features and saving logic take place in separated threads every time they are called.

MySQL record example:
(1, datetime.datetime(2021, 5, 21, 12, 54, 48), 'E:/GitHub/Project-EYE/database/detection_time_2021-05-21 12_54_47.jpg')



# Used tools
* darknet framework
* yolov4
* python
* threading
* opencv
* mysql

# Detection examples

Person detected without mask:

<p float="left">
<img src="https://user-images.githubusercontent.com/39679208/119143311-c37b1580-ba47-11eb-8fba-d4ab30fb07f4.jpg"  width="49%" height="50%"/> <img src="https://user-images.githubusercontent.com/39679208/119143408-d988d600-ba47-11eb-9cac-a4cd1592898d.jpg"  width="49%" height="50%"/>
</p>

<p float="left">
<img src="https://user-images.githubusercontent.com/39679208/119143465-eb6a7900-ba47-11eb-9ac9-5ecf4a4c8da1.jpg"  width="49%" height="50%"/> <img src="https://user-images.githubusercontent.com/39679208/119238225-872be000-bb41-11eb-8767-142ce1245252.jpg"  width="49%" height="50%"/>
</p>

Person detected with mask:

<p float="left">
<img src="https://user-images.githubusercontent.com/39679208/119144382-e528cc80-ba48-11eb-8522-d1428292cb06.jpg"  width="49%" height="50%"/> <img src="https://user-images.githubusercontent.com/39679208/119144429-efe36180-ba48-11eb-848a-7ff4d287f9f3.jpg"  width="49%" height="50%"/>
</p>

<p float="left">
<img src="https://user-images.githubusercontent.com/39679208/119144464-f96cc980-ba48-11eb-9a5b-cd0630bac627.jpg"  width="49%" height="50%"/> <img src="https://user-images.githubusercontent.com/39679208/119144504-038ec800-ba49-11eb-8164-3740f937a7e3.jpg"  width="49%" height="50%"/>
</p>

# Video presentation
Even with poor camera qualty and lighting results are satisfying.
https://youtu.be/kEa3-nvqm-4

