'''
开发时间：2021/12/12 19:22

python 3.8.5

开发人；Lasseford Wang

'''


import datetime


class DataBase:
    def __init__(self, filename1,filename2):
        self.filename1 = filename1
        self.filename2 = filename2
        self.users = None
        self.file = None
        self.load()

    def load(self):
        self.file = open(self.filename1, "r")
        self.users = {}

        for line in self.file:
            email, password, name, created = line.strip().split(";")
            self.users[email] = (password, name, created)

        self.file.close()

    def get_user(self, email):
        if email in self.users:
            return self.users[email]
        else:
            return -1

    def add_user(self, email, password, name):
        if email.strip() not in self.users:
            self.users[email.strip()] = (password.strip(), name.strip(), DataBase.get_date())
            self.save()
            return 1
        else:
            print("Email exists already")
            return -1

    def validate(self, email, password):
        if self.get_user(email) != -1:
            return self.users[email][0] == password
        else:
            return False

    def save(self):
        with open(self.filename1, "w") as f:
            for user in self.users:
                f.write(user + ";" + self.users[user][0] + ";" + self.users[user][1] + ";" + self.users[user][2] + "\n")

    @staticmethod
    def get_date():
        return str(datetime.datetime.now()).split(" ")[0]


    def save_history(self,ee,history):
        with open(self.filename2,"a",encoding="utf-8") as f:
            f.write(ee+";"+history+";"+self.get_date()+"\n")


    def load_history(self,email):
        self.file = open(self.filename2,"r",encoding="utf-8")
        historyl=[]

        for line in self.file:
            e,history,searchtime=line.strip().split(";")
            if(e==email):
                historyl.append((history,searchtime))

        return historyl