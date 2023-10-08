import pandas as pd

pd.options.mode.chained_assignment = None


def fix_csv():
    df = pd.read_csv("daniel_aI_dataset.csv",
                     on_bad_lines='skip',
                     sep=':::')

    df = df.drop(columns=['index'])

    print(df.head())
    print(df.shape)

    for i, txt in enumerate(df['text']):
        df['text'][i] = txt.replace('/@newline/', '\n')[:-1]

    df.to_csv('daniel_ai_dataset_fixed.csv', sep='`')
    print(print(df.head()))


df = pd.read_csv("daniel_ai_dataset_fixed.csv",
                 sep='`')

#23042181

txt_set = ""
for user_id, user_txt in zip(df['user_id'], df['text']):
    if user_id: #== 23042181:
        txt_set = txt_set + "\n" + user_txt

plane_txt = open('full_chat_set.txt', 'x')
a = plane_txt.write(txt_set)
plane_txt.close()
print(a)
