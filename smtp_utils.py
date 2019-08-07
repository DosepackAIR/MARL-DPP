import os
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart


class MailMan:
    """
    Class is used to mail some information to list of mail ids.
    """

    def __init__(self, user_name, password):
        self.smtp_server = 'smtp.gmail.com'
        self.smtp_host = 587
        self.user_name = user_name
        self.password = password
        self.smtp = smtplib.SMTP(self.smtp_server, self.smtp_host)

    def login_to_mail(self):
        self.smtp.ehlo()
        self.smtp.starttls()
        self.smtp.ehlo()
        self.smtp.login(self.user_name, self.password)

    def send_text(self, subject, text_content, receiver_list):
        msg = MIMEMultipart()
        msg['Subject'] = subject
        msg['From'] = self.user_name
        text = MIMEText(text_content)
        msg.attach(text)
        for receiver in receiver_list:
            msg['To'] = receiver
            self.smtp.sendmail(self.user_name, receiver, msg.as_string())
        pass

    def send_image_with_text(self, subject, text_content, img_file_name, receiver_list):
        img_data = open(img_file_name, 'rb').read()
        msg = MIMEMultipart()
        msg['Subject'] = subject
        msg['From'] = self.user_name
        text = MIMEText(text_content)
        msg.attach(text)
        image = MIMEImage(img_data, name=os.path.basename(img_file_name))
        msg.attach(image)
        for receiver in receiver_list:
            msg['To'] = receiver
            self.smtp.sendmail(self.user_name, receiver, msg.as_string())
        pass
    
    def send_text_file_with_text(self, subject, text_content, text_file_name, receiver_list):
        msg = MIMEMultipart()
        msg['Subject'] = subject
        msg['From'] = self.user_name
        text = MIMEText(text_content)
        text_file_data = MIMEText(open(text_file_name, 'r').read().splitlines()[-1])
        msg.attach(text); msg.attach(text_file_data)
        # image = MIMEImage(img_data, name=os.path.basename(img_file_name))
        # msg.attach(image)
        for receiver in receiver_list:
            msg['To'] = receiver
            self.smtp.sendmail(self.user_name, receiver, msg.as_string())
        pass

    def quit_mail_man(self):
        self.smtp.quit()


# """
# Example
# """
# # user_name = 'floorlayoutobserver@gmail.com'
# # password = 'yash@123'
# # img_file_name = 'design_patterns.png'
# # mm = MailMan(user_name=user_name, password=password)
# # mm.login_to_mail()
# # mm.send_text(subject='Class_Test', text_content='Checking class is working or not',
# #              receiver_list=['yashk@dosepack.com', 'yashhkhandhediya@gmail.com'])
# # mm.quit_mail_man()
