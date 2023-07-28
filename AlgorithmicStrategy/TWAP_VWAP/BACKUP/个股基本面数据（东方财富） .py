from tkinter import Tk, Label, Entry, Button, messagebox
from urllib.parse import urlencode
import pandas as pd
import requests
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import sqlite3
'''
前面是数据爬取，后面有一个简单的量价关系的机器学习，可以直接删了相关的控件
'''
def gen_secid(rawcode: str) -> str:
    # 沪市指数
    if rawcode[:3] == '000':
        return f'1.{rawcode}'
    # 深证指数
    if rawcode[:3] == '399':
        return f'0.{rawcode}'
    # 沪市股票
    if rawcode[0] != '6':
        return f'0.{rawcode}'
    # 深市股票
    return f'1.{rawcode}'


def get_k_history(code: str, beg: str, end: str, klt: int = 101, fqt: int = 1) -> pd.DataFrame:
    '''
    功能获取k线数据
    -

    '''
    EastmoneyKlines = {
        'f51': '日期',
        'f52': '开盘',
        'f53': '收盘',
        'f54': '最高',
        'f55': '最低',
        'f56': '成交量',
        'f57': '成交额',
        'f58': '振幅',
        'f59': '涨跌幅',
        'f60': '涨跌额',
        'f61': '换手率',
    }
    EastmoneyHeaders = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; WOW64; Trident/7.0; Touch; rv:11.0) like Gecko',
        'Accept': '*/*',
        'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
        'Referer': 'http://quote.eastmoney.com/center/gridlist.html',
    }#伪装头部
    #构建json格式的url
    fields = list(EastmoneyKlines.keys())
    columns = list(EastmoneyKlines.values())
    fields2 = ",".join(fields)
    secid = gen_secid(code)
    params = (
        ('fields1', 'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13'),
        ('fields2', fields2),
        ('beg', beg),
        ('end', end),
        ('rtntype', '6'),
        ('secid', secid),
        ('klt', f'{klt}'),
        ('fqt', f'{fqt}'),
    )
    params = dict(params)
    base_url = 'https://push2his.eastmoney.com/api/qt/stock/kline/get'
    url = base_url + '?' + urlencode(params)
    json_response: dict = requests.get(url, headers=EastmoneyHeaders).json()

    data = json_response.get('data')#数据获取
    if data is None:
        if secid[0] == '0':
            secid = f'1.{code}'
        else:
            secid = f'0.{code}'
        params['secid'] = secid
        url = base_url + '?' + urlencode(params)
        json_response: dict = requests.get(url, headers=EastmoneyHeaders).json()
        data = json_response.get('data')
    if data is None:
        messagebox.showerror('错误', f'股票代码: {code} 可能有误')
        return pd.DataFrame(columns=columns)

    klines = data['klines']

    rows = []
    for _kline in klines:#循环将数据存入
        kline = _kline.split(',')
        rows.append(kline)

    df = pd.DataFrame(rows, columns=columns)

    # 清除空数据
    df = df.dropna()

    # 清除收盘价为"-"的数据
    df = df[df['收盘'] != '-']
    #连接数据库
    conn = sqlite3.connect('database.db')
    df.to_sql('your_table_name', conn, if_exists='replace', index=False)#将DATA 转存到数据库
    df1 = pd.read_sql('SELECT * FROM your_table_name', conn)#将数据提取出数据库
    conn.close()
    print(df1)#打印数据


    return df

#获取数据函数
def fetch_stock_data():
    code = code_entry.get()
    if not code:
        messagebox.showerror('错误', '请输入股票代码')
        return

#获取开始日期和截止日期
    start_date = start_date_entry.get()
    end_date = end_date_entry.get()

    print(f'正在获取 {code} 从 {start_date} 到 {end_date} 的 k线数据......')
    df = get_k_history(code, start_date, end_date)






#通过机器学习预测股价
def predict_next_day_price():
    code = code_entry.get()
    if not code:
        messagebox.showerror('错误', '请输入股票代码')
        return
    start_date = start_date_entry.get()
    end_date = end_date_entry.get()

    print(f'正在获取 {code} 从 {start_date} 到 {end_date} 的 k线数据......')
    df = get_k_history(code, start_date, end_date)

    if df.empty:
        return

    # 添加前一日至前三日的开盘价和收盘价作为目标变量
    df['前一日开盘'] = df['开盘'].shift(1)
    df['前一日收盘'] = df['收盘'].shift(1)
    df['前二日开盘'] = df['开盘'].shift(2)
    df['前二日收盘'] = df['收盘'].shift(2)
    df['前三日开盘'] = df['开盘'].shift(3)
    df['前三日收盘'] = df['收盘'].shift(3)

    # 去除前三日缺失的数据
    df = df.dropna()

    # 提取输入特征和目标变量
    features = df[['成交额', '换手率', '振幅', '前一日开盘', '前一日收盘', '前二日开盘', '前二日收盘', '前三日开盘', '前三日收盘']]
    target_open = df['开盘']
    target_close = df['收盘']

    # 划分训练集和测试集
    X_train, X_test, y_train_open, y_test_open, y_train_close, y_test_close = train_test_split(
        features, target_open, target_close, test_size=0.2, random_state=42
    )

    # 创建线性回归模型
    model_open = LinearRegression()
    model_close = LinearRegression()

    # 训练模型
    model_open.fit(X_train, y_train_open)
    model_close.fit(X_train, y_train_close)

    # 预测测试集
    y_pred_open = model_open.predict(X_test)
    y_pred_close = model_close.predict(X_test)

    # 评估模型性能（均方误差）
    mse_open = mean_squared_error(y_test_open, y_pred_open)
    mse_close = mean_squared_error(y_test_close, y_pred_close)
    print('开盘价的均方误差:', mse_open)
    print('收盘价的均方误差:', mse_close)

    # 使用模型进行预测
    last_row = df.iloc[-1]

    # 提取最后一行的成交额、换手率和振幅数据以及前三日的开盘价和收盘价
    your_new_volume = last_row['成交额']
    your_new_turnover = last_row['换手率']
    your_new_amplitude = last_row['振幅']
    your_new_previous_open = last_row['开盘']
    your_new_previous_close = last_row['收盘']
    your_new_second_open = last_row['前一日开盘']
    your_new_second_close = last_row['前一日收盘']
    your_new_third_open = last_row['前二日开盘']
    your_new_third_close = last_row['前二日收盘']

    # 创建包含最后一行数据的new_data数据框
    new_data = pd.DataFrame({
        '成交额': [your_new_volume],
        '换手率': [your_new_turnover],
        '振幅': [your_new_amplitude],
        '前一日开盘': [your_new_previous_open],
        '前一日收盘': [your_new_previous_close],
        '前二日开盘': [your_new_second_open],
        '前二日收盘': [your_new_second_close],
        '前三日开盘': [your_new_third_open],
        '前三日收盘': [your_new_third_close]
    })

    predicted_open = model_open.predict(new_data)
    predicted_close = model_close.predict(new_data)

    print('下一日开盘价的预测值:', predicted_open[0])
    print('下一日收盘价的预测值:', predicted_close[0])

    messagebox.showinfo('预测结果', f'下一日开盘价的预测值: {predicted_open[0]}\n下一日收盘价的预测值: {predicted_close[0]}')




# 创建主窗口
root = Tk()
root.title('股票预测')
root.geometry('400x300')
# # 爬取大盘数据
# fetch_stock_all_data_button = Button(root, text='获取大盘数据', command=fetch_stock_all_data)
# fetch_stock_all_data_button.pack()
# # 数据清洗
# clean_stock_data_button = Button(root, text='数据清洗', command= clean_stock_data)
# clean_stock_data.pack()
# # 输出评价排名
# evaluate_stocks_button= Button(root, text='选股排名', command= evaluate_stocks)
# evaluate_stocks.pack()
# 股票代码标签和输入框
code_label = Label(root, text='股票代码:')
code_label.pack()
code_entry = Entry(root)
code_entry.pack()

# 开始日期标签和输入框
start_date_label = Label(root, text='开始日期:')
start_date_label.pack()
start_date_entry = Entry(root)
start_date_entry.pack()

# 结束日期标签和输入框
end_date_label = Label(root, text='结束日期:')
end_date_label.pack()
end_date_entry = Entry(root)
end_date_entry.pack()

# 获取数据按钮
fetch_data_button = Button(root, text='获取K线数据', command=fetch_stock_data)
fetch_data_button.pack()

# 预测按钮
predict_button = Button(root, text='预测', command=predict_next_day_price)
predict_button.pack()

# 运行主窗口的消息循环
root.mainloop()
