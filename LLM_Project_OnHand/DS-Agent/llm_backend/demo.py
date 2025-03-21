dic = {'finish_reason': 'stop'}

print(dic.get('finish_reason', False) == 'stop')
