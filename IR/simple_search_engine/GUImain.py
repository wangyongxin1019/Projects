'''
开发时间：2021/12/12 19:21

python 3.8.5

开发人；Lasseford Wang

'''

import kivy
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import ObjectProperty
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.properties import StringProperty
from database import DataBase
import Search

from kivy.core.text import LabelBase
LabelBase.register(name='Font_Hanzi',fn_regular='D:/py文件/intermediate/venv/Lib/site-packages/kivy/data/fonts/wryh.ttf')

currentemail=''
global searchRes
searchRes=''

global histt
histt=''

db = DataBase("users.txt","history.txt")

class CreateAccountWindow(Screen):
    namee = ObjectProperty(None)
    email = ObjectProperty(None)
    password = ObjectProperty(None)

    def submit(self):
        if self.namee.text != "" and self.email.text != "" and self.email.text.count("@") == 1 and self.email.text.count(".") > 0:
            if self.password != "":
                db.add_user(self.email.text, self.password.text, self.namee.text)

                self.reset()

                sm.current = "login"
            else:
                invalidForm()
        else:
            invalidForm()

    def login(self):
        self.reset()
        sm.current = "login"

    def reset(self):
        self.email.text = ""
        self.password.text = ""
        self.namee.text = ""


class LoginWindow(Screen):
    email = ObjectProperty(None)
    password = ObjectProperty(None)

    def loginBtn(self):
        if db.validate(self.email.text, self.password.text):
            MainWindow.current = self.email.text
            self.reset()
            sm.current = "main"
        else:
            invalidLogin()

    def createBtn(self):
        self.reset()
        sm.current = "create"

    def reset(self):
        self.email.text = ""
        self.password.text = ""


class MainWindow(Screen):
    n = ObjectProperty(None)
    created = ObjectProperty(None)
    email = ObjectProperty(None)
    current = ""

    def logOut(self):
        sm.current = "login"

    def on_enter(self, *args):
        password, name, created = db.get_user(self.current)
        self.n.text = "Account Name: " + name
        self.email.text = "Email: " + self.current
        self.created.text = "Created On: " + created


class SearchWindow(Screen):

    pass



class DocSearchWindow(Screen):
    a=ObjectProperty(None)

    def search(self):
        if(MainWindow.current!='' and self.a.text != ''):
            db.save_history(MainWindow.current,self.a.text)
            searchRes = Search.printres(Search.simpleSearch(self.a.text))

    def reset(self):
        pass


class OnsiteSearchWindow(Screen):
    url1=ObjectProperty(None)
    a=ObjectProperty(None)

    def search(self):
        if (MainWindow.current != '' and self.a.text != ''):
            db.save_history(MainWindow.current, self.a.text)
            searchRes = Search.printres(Search.OnsiteSearch(self.url1.text,self.a.text))


    def reset(self):
        pass


class DateSearchWindow(Screen):
    time1 = ObjectProperty(None)
    time2 = ObjectProperty(None)

    def search(self):
        if (MainWindow.current != '' and self.time1.text != '' and self.time2.text != ''):
            db.save_history(MainWindow.current, self.time1.text+self.time2.text)
            searchRes = Search.printres(Search.dateSearch(self.time1.text, self.time2.text))
        print(searchRes)


    def reset(self):
        pass

class PhraseSearchWindow(Screen):
    a = ObjectProperty(None)

    def search(self):
        if (MainWindow.current != '' and self.a.text != ''):
            db.save_history(MainWindow.current, self.a.text)
            searchRes = Search.printres(Search.phraseSearch(self.a.text))

    def reset(self):
        pass

class WildCardSearchWindow(Screen):
    a = ObjectProperty(None)

    def search(self):
        if (MainWindow.current != '' and self.a.text != ''):
            db.save_history(MainWindow.current, self.a.text)
            searchRes = Search.printres(Search.wildcardSearch(self.a.text))
        pass

    def reset(self):
        pass


class ResWindow(Screen):

    def show(self):
        print(searchRes)
        self.ids.res_label.text = searchRes
        return "res"

class SearchHistoryWindow(Screen):

    def show(self):
        if(len(db.load_history(MainWindow.current))!=0):
            s=''
            for a in db.load_history(MainWindow.current):
                s=s+str(a)+'\n'
            #histt=s
            self.ids.his_label.text = s
            return s
        else:
            s="search history is empty!!"
            self.ids.his_label.text = s
            return "search history is empty!!"

class WindowManager(ScreenManager):
    pass


def invalidLogin():
    pop = Popup(title='Invalid Login',
                  content=Label(text='Invalid username or password.'),
                  size_hint=(None, None), size=(400, 400))
    pop.open()


def invalidForm():
    pop = Popup(title='Invalid Form',
                  content=Label(text='Please fill in all inputs with valid information.'),
                  size_hint=(None, None), size=(400, 400))

    pop.open()


kv = Builder.load_file("my.kv")

sm = WindowManager()


screens = [LoginWindow(name="login"), CreateAccountWindow(name="create"),MainWindow(name="main"),
           SearchWindow(name="search"),DocSearchWindow(name="doc"),OnsiteSearchWindow(name="onsite"),
           DateSearchWindow(name="date"),PhraseSearchWindow(name="phrase"),WildCardSearchWindow(name="wildcard"),
           SearchHistoryWindow(name="history"),ResWindow(name="res")]
for screen in screens:
    sm.add_widget(screen)

sm.current = "login"



class MyMainApp(App):
    result=StringProperty("")

    def build(self):
        self.result=searchRes
        return sm


if __name__ == "__main__":
    MyMainApp().run()


