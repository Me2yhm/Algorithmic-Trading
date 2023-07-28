import requests
import json
import os

json_url = "http://48.push2.eastmoney.com/api/qt/clist/get?cb=jQuery112402508937289440778_1658838703304&pn={page}&pz=20&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&wbp2u=|0|0|0|web&fid=f3&fs=m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048&fields=f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f22,f11,f62,f128,f136,f115,f152&_=1658838703305"

filename = "stock_data.csv"
if not os.path.exists(filename):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("股票代码,股票名称,最新价,涨跌幅,涨跌额,成交量（手）,成交额,振幅,换手率,市盈率,量比,最高,最低,今开,昨收,市净率\n")

for i in range(1, 300):
    print("第%s页" % str(i))
    res = requests.get(json_url.format(page=str(i)))
    result = res.text.split("jQuery112402508937289440778_1658838703304")[1].split("(")[1].split(");")[0]
    result_json = json.loads(result)
    try:
        stock_data = result_json['data']['diff']
    except:
        print('over')
        break
    with open(filename, "a", encoding="utf-8") as f:
        for j in stock_data:
            Code = j["f12"]
            Name = j["f14"]
            Close = j['f2']
            ChangePercent = j["f3"]
            Change = j['f4']
            Volume = j['f5']
            Amount = j['f6']
            Amplitude = j['f7']
            TurnoverRate = j['f8']
            PERation = j['f9']
            VolumeRate = j['f10']
            Hign = j['f15']
            Low = j['f16']
            Open = j['f17']
            PreviousClose = j['f18']
            PB = j['f22']
            row = '{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(
                Code, Name, Close, ChangePercent, Change, Volume, Amount, Amplitude,
                TurnoverRate, PERation, VolumeRate, Hign, Low, Open, PreviousClose, PB)
            f.write(row)
j