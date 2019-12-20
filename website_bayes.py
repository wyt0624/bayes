import re
import redis
import shutil
import pymysql
import jieba
from numpy import *
import pickle  # 持久化
import os
from sklearn.feature_extraction.text import TfidfTransformer  # TF-IDF向量转换类
from sklearn.feature_extraction.text import TfidfVectorizer  # TF_IDF向量生成类
from sklearn.datasets.base import Bunch
from sklearn.naive_bayes import MultinomialNB  # 多项式贝叶斯算法
from threading import Thread
from sklearn.metrics import classification_report
class message:
    def __init__(self, telephone, qq, wechat, address,url):
        self.telephone = telephone
        self.qq = qq
        self.wechat = wechat
        self.address = address
        self.url = url


# 连接Redis
def connRedis():
    conn = redis.Redis(host='172.30.154.203', port=6377)
    return conn

#连接mysql
def conn_mysql():
    conn = pymysql.connect(host='172.30.154.204', port=3306, user='root', password='123456', database='crawler',charset='utf8')
    return conn

#读取文件
def readFile(path):

    with open(path, 'r', errors='ignore') as file:  # 文档中编码有些问题，所有用errors过滤错误
        try:
            content = file.read()
        except Exception as e:
            print(e)
        finally:
            return content

# 写入文件
def writeFile(path,content):
    if not os.path.exists(path):
        with open(path, 'w', errors='ignore') as file:
            try:
                file.write(content)
            except Exception as e:
                print(e)


# 保存文件
def saveFile(path, result):
    with open(path, 'w', errors='ignore') as file:
        file.write(result)


def segText(inputPath, resultPath):
    fatherLists = os.listdir(inputPath)  # 主目录
    for eachDir in fatherLists:  # 遍历主目录中各个文件夹
        eachPath = inputPath + eachDir + "/"  # 保存主目录中每个文件夹目录，便于遍历二级文件
        each_resultPath = resultPath + eachDir + "/"  # 分词结果文件存入的目录
        if not os.path.exists(each_resultPath):
            os.makedirs(each_resultPath)
        childLists = os.listdir(eachPath)  # 获取每个文件夹中的各个文件
        for eachFile in childLists:  # 遍历每个文件夹中的子文件
            eachPathFile = eachPath + eachFile  # 获得每个文件路径
            #  print(eachFile)
            content = readFile(eachPathFile)  # 调用上面函数读取内容
            # content = str(content)
            chinese = find_chinese(str(content))
            result = chinese.replace("\r\n", "").replace(" ", "")
                # .strip()  # 删除多余空行与空格
            # result = content.replace("\r\n","").strip()
            cutResult = jieba.cut(result)  # 默认方式分词，分词结果用空格隔开
            saveFile(each_resultPath + eachFile, " ".join(cutResult))  # 调用上面函数保存文件

def segText_test(inputPath, resultPath):
    fatherLists = os.listdir(inputPath)  # 主目录
    for eachDir in fatherLists:  # 遍历主目录中各个文件夹
        eachPath = inputPath + eachDir + "/"  # 保存主目录中每个文件夹目录，便于遍历二级文件
        each_resultPath = resultPath + eachDir + "/"  # 分词结果文件存入的目录
        if not os.path.exists(each_resultPath):
            os.makedirs(each_resultPath)
        childLists = os.listdir(eachPath)  # 获取每个文件夹中的各个文件
        for eachFile in childLists:  # 遍历每个文件夹中的子文件
            eachPathFile = eachPath + eachFile  # 获得每个文件路径
            #  print(eachFile)
            content = readFile(eachPathFile)  # 调用上面函数读取内容
            # content = str(content)
            chinese = find_chinese(str(content))
            result = chinese.replace("\r\n", "").replace(" ", "")
            # .strip()  # 删除多余空行与空格
            # result = content.replace("\r\n","").strip()

            cutResult = jieba.cut(result)  # 默认方式分词，分词结果用空格隔开
            saveFile(each_resultPath + eachFile, " ".join(cutResult))  # 调用上面函数保存文件
            try:
                os.remove(eachPath + eachFile)
            except Exception as e:
                print(e)

def find_chinese(content):
    pattern = re.compile(r'[^\u4e00-\u9fa5]')
    chinese = re.sub(pattern, '', content)
    return chinese


def get_message(content):
    chinese = re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])",' ', content)
    return chinese


def bunchSave(inputFile, outputFile):
    catelist = os.listdir(inputFile)
    bunch = Bunch(target_name=[], label=[], filenames=[], contents=[])
    bunch.target_name.extend(catelist)  # 将类别保存到Bunch对象中
    for eachDir in catelist:
        eachPath = inputFile + eachDir + "/"
        fileList = os.listdir(eachPath)
        for eachFile in fileList:  # 二级目录中的每个子文件
            fullName = eachPath + eachFile  # 二级目录子文件全路径
            bunch.label.append(eachDir)  # 当前分类标签
            bunch.filenames.append(fullName)  # 保存当前文件的路径
            bunch.contents.append(readFile(fullName).strip())  # 保存文件词向量
    with open(outputFile, 'wb') as file_obj:  # 持久化必须用二进制访问模式打开
        pickle.dump(bunch, file_obj)
        # pickle.dump(obj, file, [,protocol])函数的功能：将obj对象序列化存入已经打开的file中。
        # obj：想要序列化的obj对象。
        # file:文件名称。
        # protocol：序列化使用的协议。如果该项省略，则默认为0。如果为负值或HIGHEST_PROTOCOL，则使用最高的协议版本


def readBunch(path):
    with open(path, 'rb') as file:
        bunch = pickle.load(file)
        # pickle.load(file)
        # 函数的功能：将file中的对象序列化读出。
    return bunch


def writeBunch(path, bunchFile):
    with open(path, 'wb') as file:
        pickle.dump(bunchFile, file)


def getStopWord(inputFile):
    stopWordList = readFile(inputFile).splitlines()
    return stopWordList


def getTFIDFMat(inputPath, stopWordList, outputPath,
                tftfidfspace_path,tfidfspace_arr_path,tfidfspace_vocabulary_path):  # 求得TF-IDF向量
    bunch = readBunch(inputPath)
    tfidfspace = Bunch(target_name=bunch.target_name, label=bunch.label, filenames=bunch.filenames, tdm=[],
                       vocabulary={})
    '''读取tfidfspace'''
    tfidfspace_out = str(tfidfspace)
    saveFile(tftfidfspace_path, tfidfspace_out)
    # 初始化向量空间
    vectorizer = TfidfVectorizer(stop_words=stopWordList, sublinear_tf=True, max_df=0.5)
    transformer = TfidfTransformer()  # 该类会统计每个词语的TF-IDF权值
    # 文本转化为词频矩阵，单独保存字典文件
    tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)
    tfidfspace_arr = str(vectorizer.fit_transform(bunch.contents))
    saveFile(tfidfspace_arr_path, tfidfspace_arr)
    tfidfspace.vocabulary = vectorizer.vocabulary_  # 获取词汇
    tfidfspace_vocabulary = str(vectorizer.vocabulary_)
    saveFile(tfidfspace_vocabulary_path, tfidfspace_vocabulary)
    '''over'''
    writeBunch(outputPath, tfidfspace)


def getTestSpace(testSetPath, trainSpacePath, stopWordList, testSpacePath,
                 testSpace_path,testSpace_arr_path,trainbunch_vocabulary_path):
    bunch = readBunch(testSetPath)
    # 构建测试集TF-IDF向量空间
    testSpace = Bunch(target_name=bunch.target_name, label=bunch.label, filenames=bunch.filenames, tdm=[],
                      vocabulary={})
    '''
       读取testSpace
       '''
    testSpace_out = str(testSpace)
    saveFile(testSpace_path, testSpace_out)
    # 导入训练集的词袋
    trainbunch = readBunch(trainSpacePath)
    # 使用TfidfVectorizer初始化向量空间模型  使用训练集词袋向量
    vectorizer = TfidfVectorizer(stop_words=stopWordList, sublinear_tf=True, max_df=0.5,
                                 vocabulary=trainbunch.vocabulary)
    transformer = TfidfTransformer()
    testSpace.tdm = vectorizer.fit_transform(bunch.contents)
    dict = {}
    for i, j in zip(testSpace.tdm,testSpace.filenames):
        if i.data.__len__() == 0:
            file_name = str(j).split('/', 4)[4]
            print('沒有用文件' + file_name)
            updateInfo(9, file_name)
            dict[file_name] = 1
            os.remove(j)
            # testSpace.tdm.remove(i)
        # if os.path.exists(j):
        #     os.remove(j)
    # if testSpace.tdm.data.__len__() == 0:
    #     return 0
    testSpace.vocabulary = trainbunch.vocabulary
    testSpace_arr = str(testSpace.tdm)
    trainbunch_vocabulary = str(trainbunch.vocabulary)
    saveFile(testSpace_arr_path, testSpace_arr)
    saveFile(trainbunch_vocabulary_path, trainbunch_vocabulary)
    # 持久化
    writeBunch(testSpacePath, testSpace)
    return dict

def bayesAlgorithm(trainPath, testPath,tfidfspace_out_arr_path,
                   tfidfspace_out_word_path,testspace_out_arr_path,
                   testspace_out_word_apth,dict):
    trainSet = readBunch(trainPath)
    testSet = readBunch(testPath)
    clf = MultinomialNB(alpha=0.01).fit(trainSet.tdm, trainSet.label)

    # alpha:0.001 alpha 越小，迭代次数越多，精度越高
    # print(shape(trainSet.tdm))  #输出单词矩阵的类型
    # print(shape(testSet.tdm))
    '''处理bat文件'''
    tfidfspace_out_arr = str(trainSet.tdm)  # 处理
    tfidfspace_out_word = str(trainSet)
    saveFile(tfidfspace_out_arr_path, tfidfspace_out_arr)  # 矩阵形式的train_set.txt
    saveFile(tfidfspace_out_word_path, tfidfspace_out_word)  # 文本形式的train_set.txt

    testspace_out_arr = str(testSet)
    testspace_out_word = str(testSet.label)
    saveFile(testspace_out_arr_path, testspace_out_arr)
    saveFile(testspace_out_word_apth, testspace_out_word)


    predicted = clf.predict(testSet.tdm)
    # total = len(predicted)

    for flabel, fileName, expct_cate in zip(testSet.label, testSet.filenames, predicted):
        file_name = str(fileName).split('/',4)[4]
        if not file_name in dict:
            print(fileName, "预测类别：", expct_cate)
            updateInfo(expct_cate,file_name)
            if int(expct_cate) == 1 or int(expct_cate) == 2:
                try:
                    # 获取QQ微信联系方式
                    content = readFile('./bak/bak/'+file_name)
                    messgae_content = get_message(str(content))
                    telephone = get_telephone(messgae_content)
                    msg = message(telephone, '', '', '', file_name)
                    update_message(msg)
                    print(telephone)
                except Exception as e:
                    print(e)


            if not os.path.exists('./bak/'+ str(expct_cate)):
                os.makedirs('./bak/'+ str(expct_cate))
            if not os.path.exists('./bak/'+expct_cate + '/' + file_name):
                try:
                    shutil.copy(fileName, './bak/'+expct_cate)
                    os.remove(fileName)
                except Exception as e:
                    print(e)
                    # shutil.rmtree('./split/test_split/test/')
                # if os.path.exists(fileName):
            else:
                os.remove(fileName)
    # if os.path.exists('./split/'):
    #     shutil.rmtree('./split/')
    dict.clear()

def updateInfo(expct_cate,file_name):
    conn = conn_mysql()
    # total = len(predicted)
    cursor = conn.cursor()
    sql = 'update info set category = %s where url = %s'
    try:
        cursor.execute(sql, [int(expct_cate), file_name])
        conn.commit()
    except Exception as e:
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

def get_telephone(message):
    telephone = ''
    num = 0
    for i in re.finditer("(13[0-9]|14[15679]|15[0-3,5-9]|166|17[0-8]|18[0-9]|19[89])\\d{8}", message):
        if num == 0:
            telephone = telephone + i.group()
        else:
            if not telephone.__contains__(i.group()):
                telephone = telephone + ',' + i.group()
        num = num + 1
    return telephone

def train():
    split_datapath = "./split/split_data/"  # 对原始数据分词之后的数据路径
    datapath = "./data/"  #原始数据路径
    stopWord_path = "./stop/stopword.txt"#停用词路径
    train_dat_path = "./train_set.dat"  # 读取分词数据之后的词向量并保存为二进制文件
    tfidfspace_dat_path = "./tfidfspace.dat"  #tf-idf词频空间向量的dat文件
    tfidfspace_path = "./tfidfspace.txt"  # 将TF-IDF词向量保存为txt，方便查看
    tfidfspace_arr_path = "./tfidfspace_arr.txt"  # 将TF-IDF词频矩阵保存为txt，方便查看
    tfidfspace_vocabulary_path = "./tfidfspace_vocabulary.txt"  # 将分词的词汇统计信息保存为txt，方便查看
    #输入训练集
    segText(datapath,#读入数据
            split_datapath)#输出分词结果
    bunchSave(split_datapath,#读入分词结果
              train_dat_path)  # 输出分词向量
    stopWordList = getStopWord(stopWord_path)  # 获取停用词表
    getTFIDFMat(train_dat_path, #读入分词的词向量
                stopWordList,    #获取停用词表
                tfidfspace_dat_path, #tf-idf词频空间向量的dat文件
                tfidfspace_path, #输出词频信息txt文件
                tfidfspace_arr_path,#输出词频矩阵txt文件
                tfidfspace_vocabulary_path)  #输出单词txt文件

def test(stopWordList):

    stopWord_path = "./stop/stopword.txt"#停用词路径
    test_path = "./test/"            #测试集路径
    test_split_dat_path =  "./test_set.dat" #测试集分词bat文件路径
    testspace_dat_path ="./testspace.dat"   #测试集输出空间矩阵dat文件
    testSpace_path = "./testSpace.txt"  #测试集分词信息
    testSpace_arr_path = "./testSpace_arr.txt"  #测试集词频矩阵信息
    testspace_out_arr_path = "./testspace_out_arr.txt"     #测试集输出矩阵信息
    testspace_out_word_apth ="./testspace_out_word.txt"    #测试界单词信息
    tfidfspace_out_arr_path = "./tfidfspace_out_arr.txt"   #tfidf输出矩阵信息
    tfidfspace_out_word_path = "./tfidfspace_out_word.txt" #单词形式的txt
    tfidfspace_dat_path = "./tfidfspace.dat"  #tf-idf词频空间向量的dat文件
    test_split_path = './split/test_split/'   #测试集分词路径
    trainbunch_vocabulary_path = "./trainbunch_vocabulary.txt" #所有分词词频信息

    filename = redis_write_file(test_path)
    if not filename == '':
        # stopWordList = getStopWord(stopWord_path)  # 获取停用词表
        segText_test(test_path,
                test_split_path)  # 对测试集读入文件，输出分词结果
        bunchSave(test_split_path,
                  test_split_dat_path)  #

        dict = getTestSpace(test_split_dat_path,
                     tfidfspace_dat_path,
                     stopWordList,
                     testspace_dat_path,
                     testSpace_path,
                     testSpace_arr_path,
                     trainbunch_vocabulary_path)# 输入分词文件，停用词，词向量，输出特征空间(txt,dat文件都有)

        bayesAlgorithm(tfidfspace_dat_path,
                   testspace_dat_path,
                   tfidfspace_out_arr_path,
                   tfidfspace_out_word_path,
                   testspace_out_arr_path,
                   testspace_out_word_apth,dict)
        # else:
        #     print('沒有用文件' + filename)
        #     updateInfo(9, filename)
        #     if os.path.exists('./split/test_split/test/'+filename):
        #         os.remove('./split/test_split/test/'+filename)
    else:
        print('队列无数据')

def update_message(message):
    conn = conn_mysql()
    cursor = conn.cursor()
    sql = 'update info set telephone = %s, qq = %s, wx = %s, address=%s where url = %s'
    try:
        cursor.execute(sql, [message.telephone, message.qq, message.wechat, message.address, message.url])
        conn.commit()
    except Exception as e:
        conn.rollback()
    finally:
        cursor.close()
        conn.close()
'''
从Redis队列读取文件，落磁盘
'''
def redis_write_file(test_path):
    redis = connRedis()
    filename = ''
    for i in range(100):
        if redis.exists("analise_queue"):
            allcontext = redis.rpop("analise_queue")
            text = str(allcontext.decode('utf8'))
            filename = text.split("|", 1)[0]
            filecontent = text.split("|", 1)[1]
            try:
                writeFile(test_path + 'test/' + filename, filecontent)
                writeFile('./bak/bak/' + filename, filecontent)
            except Exception as e:
                print(e)
            if not os.path.exists('./bak/bak/'):
                os.mkdir('./bak/bak')
            # shutil.copy(test_path + 'test/' + filename, './bak/bak/'+filename)
    redis.close()
    return filename

'''
预测分类
'''
def test_start():
    stopWord_path = "./stop/stopword.txt"  # 停用词路径
    stopWordList = getStopWord(stopWord_path)
    test(stopWordList)


if __name__ == '__main__':
    # train()
    while True:
        test_start()




