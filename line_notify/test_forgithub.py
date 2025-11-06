# -*- coding: utf-8 -*-

import telegram
import os
import sys
import re
import time
import json
import paramiko
from scp import SCPClient
import requests

op_th = 500
udn_th = 500
edn_th = 500

def read_list(fn):
    out = []
    with open(fn, 'r') as fp:
        for line in fp:
            line1 = re.split('\n', line)
            if (line1[0] != ''):
                out.append(line1[0][2:-4])
    return out

def ask(url, ty, fn_):
    uu = url + ty
    for i in range(len(fn_)):
        uu2 = uu + fn_[i]
        REx = requests.get(uu2)
        if (REx.status_code != 200):
            print('%s is not existed.   %s'%(fn_[i], ty))

def createSSHClient(server, port, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client

def lineNotifyMessage(token, msg):

    headers = {
        "Authorization": "Bearer " + token, 
        "Content-Type" : "application/x-www-form-urlencoded"
    }

    payload = {'message': msg }
    r = requests.post("https://notify-api.line.me/api/notify", headers = headers, params = payload)
    return r.status_code

argv = sys.argv

# authentication ###
fndata = './data/authentication.json'
with open(fndata, 'r') as finjson:
    json_data = json.load(finjson) 
server = json_data['server']
port = json_data['port']
user = json_data['user']
password = json_data['password']
token = json_data['token']
chat_id = json_data['chat_id']

ssh = createSSHClient(server, port, user, password)
scp = SCPClient(ssh.get_transport())

line_token = 'your token'

# scp log from v01
mtable = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
msize = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
TT = time.ctime()
yy = TT[-4:]
mm = TT[4:7]
hm = TT[11:16]
for i in range(len(mtable)):
    if (mm == mtable[i]):
        MM = str(i+1).zfill(2)
        break
ddd = str(int(TT[8:10])).zfill(2)
fnlog = os.path.join('TTS_count-' + yy + MM + ddd + '.log')
print(fnlog)
remote_path = os.path.join('/fsdata/logs/' + fnlog)
local_path = os.path.join('./data/' + fnlog)
scp.get(remote_path, local_path=local_path)

# read log #########
with open(local_path, 'r') as fp:
    for line in fp:
        pass

# alarm condition ##
line1 = re.split(' ', line)
logdate = line1[0]
logtime = line1[1]
op = int(re.split(':', line1[3])[1])
udn = int(re.split(':', line1[4])[1])
edn = int(re.split(':', line1[5])[1])
smy = int(re.split(':', line1[6])[1])
print(logdate, end=' ')
print(logtime)
print("op: %d"%op, end=' ')
print("udn: %d"%udn, end=' ')
print("edn: %d"%edn, end=' ')
print("smy: %d"%smy)

ans = logdate + ' ' + logtime + '\n' + "op: " + str(op) + " udn: " + str(udn) + " edn: " + str(edn)

# call bot #########
bot = telegram.Bot(token=token)
#print(bot.get_me())
updates = bot.get_updates()
#print(updates[0])
#bot.send_message(text=line, chat_id=chat_id)

if (op >= op_th):
    alarm_msg = "Onepiece service is stuffed!!!\n" + ans
    bot.send_message(text=alarm_msg, chat_id=chat_id)
    lineNotifyMessage(line_token, alarm_msg)
if (udn >= udn_th):
    alarm_msg = "Udn service is stuffed!!!\n" + ans
    bot.send_message(text=alarm_msg, chat_id=chat_id)
    lineNotifyMessage(line_token, alarm_msg)
if (edn >= edn_th):
    alarm_msg = "Edn service is stuffed!!!\n" + ans
    bot.send_message(text=alarm_msg, chat_id=chat_id)
    lineNotifyMessage(line_token, alarm_msg)

if (len(argv) > 1):
    alarm_msg = argv[1]
    bot.send_message(text=alarm_msg, chat_id=chat_id)
    lineNotifyMessage(line_token, alarm_msg)
elif (hm == '12:00'):
    alarm_msg = "Lunch time!\n" + ans
    bot.send_message(text=alarm_msg, chat_id=chat_id)
    lineNotifyMessage(line_token, alarm_msg)




