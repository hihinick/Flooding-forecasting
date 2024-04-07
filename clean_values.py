import pandas as pd
import numpy as np
import datetime
from datetime import datetime
from datetime import timedelta
from sklearn.metrics import mean_absolute_error
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error
class clean_data_values:

    def clean(fillna_data,cnames):
        for m in cnames:
            df1=fillna_data[m]
            df1=np.array(df1)
            k=len(df1)
            for i in range(0,k):
                try:
                    df1[i]=float(df1[i])
                except:
                    df1[i]=str(df1[i])
                    #print(type(df1[i]))
            count=0
            for i in range(0,k):
                now=df1[i]
                if type(now)==str:
                    #print(now,'str')
                    count+=1
                else:
                    if count != 0 and count <24:
                        try:
                            add = (now - df1[i-(count+1)]) / (count + 1)
                            #print(count, "count")
                            #print(df1[i-(count+1)], "pre")
                            #print(now, 'now')
                            #print(i, "i")
                            for j in range(count, 0, -1):
                                df1[i - j] = df1[i - j - 1] + add
                                #print(i - j, "i-j")
                                #print(df1[i - j])
                            count = 0
                        except:
                            add = now
                            #print(count, "count")
                            #print(pre,"pre")
                            #print(now, 'now')
                            #print(i, "i")
                            for j in range(count, 0, -1):
                                df1[i-j] = add
                                #print(i-j, "i-j")
                                #print(df1[i-j])
                            count = 0
                        #print('--------------')
                    else:
                        count = 0
            fillna_data[m]=df1
        for m in cnames:
            df1=fillna_data[m]
            df1=np.array(df1)
            k=len(df1)
            for i in range(0,k):
                try:
                    df1[i]=float(df1[i])
                except:
                    df1[i]=str(df1[i])
                    #print(type(df1[i]))
            count=0
            for i in range(1, len(df1)):
                now = df1[i]
                if type(now) == str:
                    #print(now, 'str')
                    count += 1
                else:
                    if count != 0:
                        #try:
                        #print(count, "count")
                        #print(df1[i])
                        prehour = i - count
                        nowhour = i
                        #print(prehour,'day')
                        #print(nowhour, 'day')
                        #print(len(df1))
                        #print(df1[prehour-24:prehour])
                        add = 0
                        if len(df1[nowhour:nowhour + 24]) != 0 and len(df1[prehour-24:prehour]) != 0:
                            #print('我拉')
                            try:
                                add = ((np.mean(df1[prehour - 24:prehour])) - (np.mean(df1[nowhour:nowhour + 24])))/ (count+1)
                            except:
                                string=0
                                for counts in range(24):
                                    if df1[nowhour+counts]==str:
                                        string=int(i)
                                        break
                                    else:
                                        pass
                                try:
                                    add = ((np.mean(df1[prehour - 24:prehour])) - (np.mean(df1[nowhour:nowhour + string])))/ (count+1)
                                except:
                                    add = ((np.mean(df1[prehour - 24:prehour])) - 0)/ (count+1)
                            try:     
                                for j in range(1, count+1, 1):
                                    df1[i - j] = np.mean(df1[nowhour:nowhour + 24]) + (add*j)
                                    #print(i - j, "i-j")
                                    #print(df1[i - j])
                                count = 0
                            except:
                                for j in range(1, count+1, 1):
                                    try:
                                        df1[i - j] = np.mean(df1[nowhour:nowhour + int(string)]) + (add*j)
                                    except:
                                        df1[i - j] = 0 + (add*j)
                                        #print(i - j, "i-j")
                                        #print(df1[i - j])
                                count = 0
                        else:
                            if prehour < 24:#如果剛開始不夠24個數值的話處理
                                #print('在這拉')
                                add = ((np.mean(df1[0:prehour])) - (np.mean(df1[nowhour:nowhour + prehour]))) / (count +1)
                                for j in range(1, count+1, 1):
                                    df1[i - j] =np.mean(df1[nowhour:nowhour + prehour])+ (add*j)
                                    #print(i - j, "i-j")
                                    #print(df1[i - j])
                                count = 0
                            elif nowhour + 23 > len(df1):
                                #print('不對 是在這')
                                #print("in")
                                #print(df1[nowhour:len(df1)])
                                #print(df1[prehour - (len(df1) - nowhour):prehour])
                                add = ((np.mean(df1[prehour - ((len(df1)) - nowhour):prehour])) - (
                                    np.mean(df1[nowhour:len(df1)]))) / (count + 1)
                                for j in range(1, count+1, 1):
                                    df1[i - j] =np.mean(df1[nowhour:len(df1)])+ (add*j)
                                    #print(i - j, "i-j")
                                    #print(df1[i - j])
                                count = 0
            fillna_data[m]=df1
        return fillna_data

    def score(ans,fillna_data,days,test_count):
        # 做答案
        from sklearn.metrics import mean_absolute_error
        line_1 = np.zeros(shape=(len(ans), 24))
        for i in range(len(ans)):
            line_1[i] = ans[i][0:24]
        line_1 = pd.DataFrame(line_1.flatten(), columns=['line_1'])

        line_2 = np.zeros(shape=(len(ans), 24))
        for i in range(len(ans)):
            line_2[i] = ans[i][24:48]
        line_2 = pd.DataFrame(line_2.flatten(), columns=['line_2'])

        line_3 = np.zeros(shape=(len(ans), 24))
        for i in range(len(ans)):
            line_3[i] = ans[i][48:72]
        line_3 = pd.DataFrame(line_3.flatten(), columns=['line_3'])

        # pred date
        test_start_day = '2019/09/01 00:00'
        test_start_day = datetime.strptime(test_start_day, '%Y/%m/%d %H:%M') + timedelta(hours=days*24)
        test_end_day = test_start_day + timedelta(hours=(test_count+2)*24-1)
        test_start_day = datetime.strftime(test_start_day, '%Y/%m/%d %H:%M')
        test_end_day = datetime.strftime(test_end_day, '%Y/%m/%d %H:%M')

        line_1_end_day = datetime.strptime(test_end_day, '%Y/%m/%d %H:%M') - timedelta(days=2)
        line_1_end_day = datetime.strftime(line_1_end_day, '%Y/%m/%d %H:%M')
        line_2_end_day = datetime.strptime(test_end_day, '%Y/%m/%d %H:%M') - timedelta(days=1)
        line_2_end_day = datetime.strftime(line_2_end_day, '%Y/%m/%d %H:%M')
        line_2_start_day = datetime.strptime(test_start_day, '%Y/%m/%d %H:%M') + timedelta(days=1)
        line_2_start_day = datetime.strftime(line_2_start_day, '%Y/%m/%d %H:%M')
        line_3_start_day = datetime.strptime(test_start_day, '%Y/%m/%d %H:%M') + timedelta(days=2)
        line_3_start_day = datetime.strftime(line_3_start_day, '%Y/%m/%d %H:%M')

        line_1_date = pd.date_range(test_start_day, line_1_end_day, freq='H')
        line_1['date'] = line_1_date
        line_2_date = pd.date_range(line_2_start_day, line_2_end_day, freq='H')
        line_2['date'] = line_2_date
        line_3_date = pd.date_range(line_3_start_day, test_end_day, freq='H')
        line_3['date'] = line_3_date
        fillna_data = fillna_data.set_index('時間')
        test_true_line1 = fillna_data[test_start_day:line_1_end_day]
        test_true_line2 = fillna_data[line_2_start_day:line_2_end_day]
        test_true_line3 = fillna_data[line_3_start_day:test_end_day]

        # rmase
        print('RMSE：', np.sqrt(mean_squared_error(line_1['line_1'], test_true_line1['西屯'])))
        print('RMSE：', np.sqrt(mean_squared_error(line_2['line_2'], test_true_line2['西屯'])))
        print('RMSE：', np.sqrt(mean_squared_error(line_3['line_3'], test_true_line3['西屯'])))
        print('mae:',float(mean_absolute_error(np.array(line_1['line_1']).flatten(), np.array(test_true_line1['西屯']).flatten())))

        # true ans
        test_start_day = '2019/09/01 00:00'
        test_start_day = datetime.strptime(test_start_day, '%Y/%m/%d %H:%M') + timedelta(hours=days*24)
        test_end_day = test_start_day + timedelta(hours=test_count*24)
        test_start_day = datetime.strftime(test_start_day, '%Y/%m/%d %H:%M')
        test_end_day = datetime.strftime(test_end_day, '%Y/%m/%d %H:%M')

        ans = fillna_data.loc[test_start_day:'2019/12/31 23:00', '西屯']
        test_true = pd.DataFrame(ans, columns=['西屯'])
        test_true_range = pd.date_range(test_start_day, '2019/12/31 23:00', freq='H')
        test_true['date'] = test_true_range
        test_true = test_true.reset_index()

        #plot line
        # fig = go.Figure()
        # fig.add_trace(go.Scatter(x=line_1['date'], y=line_1['line_1'], name='line1_pm2.5', line_color='red'))
        # fig.add_trace(go.Scatter(x=line_2['date'], y=line_2['line_2'], name='line2_pm2.5',))
        # fig.add_trace(go.Scatter(x=line_3['date'], y=line_3['line_3'], name='line3_pm2.5',))
        # fig.add_trace(go.Scatter(x=test_true['date'], y=test_true['西屯'], name="true_pm2.5", line_color='deepskyblue'))
        # fig.update_layout(title_text='Time Series with Rangeslider', xaxis_rangeslider_visible=True)
        # fig.write_html("cosine_data.html")
        # fig.show()