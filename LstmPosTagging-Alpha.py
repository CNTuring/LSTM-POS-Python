#!/usr/bin/python2
#@Author: Pengcheng Zhu
#@Date:   6/15/2016
#@Copyright: IRC.Inc 2016
import codecs
import sys
import os
import logging
import ConfigParser
import glob
import numpy as np
from numpy import genfromtxt
from StringIO import StringIO
from netCDF4 import Dataset
"""
     This script is the main script for LSTM-RNN based POS Tagging, including:
  1.  [cwf]:create working folders
  2.  [pwt]:prepare word2vec training data and statistical corpus POS classes
  3.  [twv]:train word2vec
  4.  [cdn]:convert corpus data to number format, input is vector data and output is class number
  5.  [wnc]:write data to NC format (if u use currennt as LSTM-RNN toolkit)
  6.  [scf]: set network and other configs
  7.  [tnn]:train LSTM-RNN model
  8.  [tes]:test LSTM-RNN model, model input: nc file, model output: csv file
  9.  [CSV]:change csv file to POS Tags
  10. [ACC]:calculate accuracy

  Notes:
  1. toolkit :                       currennt, netcdf and corresponding libraies
  2. Operating System:               Linux
  3. Each component can be controled by buttons in config file (cwf=1,pwt=1,...)

"""
class LstmPosTagging(object):
    def __init__(self,logger,options):
        self.logger = logger
        self.parseConfig('config',options)
        self.vecdct = {}
        self.posdct = {}
        self.projectfolder = "."

    def parseConfig(self,confName,options):
        cf = ConfigParser.ConfigParser()
        cf.read(confName)
        count = 0
        s = "lstmpostagging"
        if cf.has_section(s):
            for o in cf.options(s):
                count += 1
                options[o] = cf.get(s,o)
        self.logger.info("Configs(%d options) Loaded ..." %count)

    def createworkfolder(self,projectfolder):
        self.logger.info('Start to create working folders.')

        if not os.path.exists('/'.join([projectfolder,'dct'])):
            os.mkdir('/'.join([projectfolder,'dct']))
        else:
            self.logger.error('Folder '+'/'.join([projectfolder,'dct'])+' has existed.')
        
        if not os.path.exists('/'.join([projectfolder,'data'])):
            os.mkdir('/'.join([projectfolder,'data']))
        else:
            self.logger.error('Folder '+'/'.join([projectfolder,'data'])+' has existed.')

        if not os.path.exists('/'.join([projectfolder,'autosave'])):
            os.mkdir('/'.join([projectfolder,'autosave']))
        else:
            self.logger.error('Folder '+'/'.join([projectfolder,'autosave'])+' has existed.')

        if not os.path.exists('/'.join([projectfolder,'configs'])):
            os.mkdir('/'.join([projectfolder,'configs']))
        else:
            self.logger.error('Folder '+'/'.join([projectfolder,'configs'])+' has existed.')

        if not os.path.exists('/'.join([projectfolder,'networks'])):
            os.mkdir('/'.join([projectfolder,'networks']))
        else:
            self.logger.error('Folder '+'/'.join([projectfolder,'networks'])+' has existed.')

        if not os.path.exists('/'.join([projectfolder,'tmp'])):
            os.mkdir('/'.join([projectfolder,'tmp']))
        else:
            self.logger.error('Folder '+'/'.join([projectfolder,'tmp'])+' has existed.')

        self.logger.info('All folders have been created.')

    def prepareVecTrainData(self,vectraincorpus):
        self.logger.info('Start to prepare word2vec training data ')
        posdct = {}
        with codecs.open(self.projectfolder+'/data/vectraindata','w','utf-8') as outf:
            for f in glob.glob(vectraincorpus+'/*.txt'):
                self.logger.info('Now processing file '+os.path.basename(f))
                with codecs.open(f,'r','gb18030') as inf:
                    line = inf.readline()
                    while(line):
                        sline = line.strip().split(' ')
                        for i in range(1,len(sline)):
                            if len(sline[i].split('/'))==2:
                                outf.write(sline[i].split('/')[0].replace('[','')+' ')
                                pos = sline[i].split('/')[1].split(']')[0]
                                if posdct.get(pos)==None:
                                    posdct[pos] = 1
                                else:
                                    posdct[pos] += 1
                        line = inf.readline()
        self.logger.info("All pos: "+" ".join(posdct.keys()))
        self.logger.info("POS categories:"+str(len(posdct.keys())))
        posdct = sorted(posdct.iteritems(),key=lambda asd:asd[1], reverse=True)
        with codecs.open(self.projectfolder+'/dct/posinfo','w','utf-8') as outf:
            num = 0
            for key in posdct:
                outf.write(key[0]+'\t'+str(key[1])+'\r\n')
                self.posdct[key[0]] = num
                num += 1

    def trainWord2Vec(self,options):
        self.logger.info('Start to train word2vec model.')
        """ training word2vec model using google word2vec toolkit"""
        word2vec = options['word2vec']
        traindata = self.projectfolder+'/data/vectraindata'
        output = self.projectfolder+'/dct/pkuvec'
        size = options['word2vecsize']
        command = ' '.join([word2vec,'-train',traindata,'-output',output,'-size',size,'-window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 20 -min-count 0'])
        os.system(command)
        self.logger.info('word2vec model has been trained.')

    def tardct(self,dctfile):
        self.logger.info('Form pos class dct from file.')
        with codecs.open(dctfile,'r','utf-8') as inf:
            dct = inf.readlines()
            for i,line in enumerate(dct):
                sline = line.strip().split('\t')
                self.posdct[sline[0]] = str(i)
        self.logger.info('pos class dct has been built.')

    def vectorfromfile(self,vecfile):
        self.logger.info('Get vectors from file.')
        with codecs.open(vecfile,'r','utf-8') as inf:
            line = inf.readline()
            line = inf.readline()
            while(line):
                sline = line.strip().split(' ')
                self.vecdct[sline[0].strip()]=' '.join(sline[1:])
                line = inf.readline()
        self.logger.info('Dct length:'+str(len(self.vecdct.keys())))
        self.logger.info('vector dct has been built.')

    def digitalize(self,filelist,outfile_pre):
        with codecs.open(outfile_pre+'input','w','utf-8') as ipos,codecs.open(outfile_pre+'output','w','utf-8') as opos,codecs.open(outfile_pre+'frame','w','utf-8') as frame:
            for f in filelist:
                self.logger.info('Now processing file :'+os.path.basename(f))
                with codecs.open(f,'r','gb18030') as inf:
                    line = inf.readline()
                    while(line):
                        sline = line.strip().split(' ')
                        itemnum = 0
                        if len(sline)>2:
                            for i in range(1,len(sline)):
                                item = sline[i].split('/')
                                if len(item) == 2:
                                    if self.vecdct.get(item[0].replace('[','').strip()) != None and self.posdct.get(item[1].split(']')[0].strip()) != None:
                                        ipos.write(self.vecdct.get(item[0].replace('[','').strip())+'\n')
                                        opos.write(str(self.posdct.get(item[1].split(']')[0].strip()))+'\n')
                                        itemnum = itemnum + 1
                                    else:
                                        self.logger.error('Item '+sline[i]+' can not find in either vec dictionary or pos dictionary.')
                            frame.write(str(itemnum)+'\n')
                        line = inf.readline()

    def convert2num(self,options):
        if len(self.posdct.keys()) == 0:
            self.tardct(self.projectfolder+'/dct/posinfo')
        if len(self.vecdct.keys()) == 0:
            self.vectorfromfile(self.projectfolder+'/dct/pkuvec')
        """prepare trainning data """
        trainlist = options['trainlist'].strip().split(',')
        for i in range(0,len(trainlist)):
            trainlist[i] = options['corpus'] +'/'+trainlist[i].strip()
        self.digitalize(trainlist,self.projectfolder+'/data/train')
        """prepare validation data """
        validationlist = options['validationlist'].strip().split(',')
        for i in range(0,len(validationlist)):
            validationlist[i] = options['corpus'] + '/' + validationlist[i].strip()
        self.digitalize(validationlist,self.projectfolder+'/data/validation')
        """prepare testing data """
        testlist = options['testlist'].strip().split(',')
        for i in range(0,len(testlist)):
            testlist[i] = options['corpus'] + '/'+ testlist[i].strip()
        self.digitalize(testlist,self.projectfolder+'/data/test')
        self.logger.info('All data has been converted to number format.')

    def write2nc(self,options):
        feature = options['feature']
        postag =  options['target']
        frame = options['frame']
        outname = options['outname']
        numSeqs =0
        numTimesteps =0
        inputPattSize = 100
        numLabels = 116
        framedata = codecs.open(frame,'r','utf-8').readlines()
        numSeqs = len(framedata)
        inlength = 0
        outlength = 0
        maxSeqTagLength = 0
        with codecs.open(feature,'r','utf-8') as fidfea,codecs.open(postag,'r','utf-8') as fidtar:
            inline = fidfea.readline()
            while(inline):
                inlength = inlength + 1
                inline = fidfea.readline()
            outline = fidtar.readline()
            while(outline):
                outlength = outlength + 1
                outline = fidtar.readline()
        if inlength != outlength:
            self.logger.error('Feature length and label length is not equal, please check data first.')
            sys.exit()
        
        numTimesteps = inlength
        maxSeqTagLength = 9
        if len(str(numSeqs))>9:
            self.logger.error('maxSeqTagLength is higher than 9. Please reset maxSeqTagLength to '+str(numSeqs))
            sys.exit()
        ## Create netcdf file , define dimensions and variables
        ncid = Dataset(outname,'w',format="NETCDF4")
        ncid.createDimension("numSeqs",numSeqs)
        ncid.createDimension("numTimesteps",numTimesteps)
        ncid.createDimension("inputPattSize",inputPattSize)
        ncid.createDimension("maxSeqTagLength",maxSeqTagLength)
        ncid.createDimension("numLabels",numLabels)

        seqTagsId = ncid.createVariable("seqTags","c",('numSeqs','maxSeqTagLength'))
        seqLengthId = ncid.createVariable("seqLengths","i","numSeqs")
        inputsId = ncid.createVariable("inputs","f",('numTimesteps','inputPattSize'))
        targetClassesId = ncid.createVariable("targetClasses","i","numTimesteps")

        fileindex = 0
        frameindex = 0
        with codecs.open(feature,'r','utf-8') as fidfea, codecs.open(postag,'r','utf-8') as fidtar,codecs.open(frame,'r','utf-8') as fidframe:
            ffinfo = fidframe.readline()
            while(ffinfo):
                filelength = int(ffinfo.strip())
                filename = "%09d"%(fileindex+1) 
                for i in range(0,filelength):
                    inputdata = fidfea.readline()
                    targetdata = fidtar.readline()
                    mat_lab = np.mat(np.genfromtxt(StringIO(inputdata.strip()),delimiter=" "))
                    mat_class = targetdata.strip()
                    inputsId[frameindex:frameindex+1,:] = mat_lab
                    targetClassesId[frameindex] = mat_class
                    frameindex = frameindex + 1
                    seqTagsId[fileindex:fileindex+1,0:maxSeqTagLength] = filename
                    seqLengthId[fileindex:fileindex+1] = filelength
                ffinfo = fidframe.readline()
                fileindex = fileindex + 1
        ncid.close()
        
    def network(self):
        self.logger.info('Writing network.')
        with codecs.open(self.projectfolder+'/networks/network.jsn','w','utf-8') as outf:
            outf.write('{\n')
            outf.write('    "layers":[\n')
            outf.write('    {\n')
            outf.write('        "size":100,\n')
            outf.write('        "name":"input_layer",\n')
            outf.write('        "type":"input"\n')
            outf.write('    },\n')
            outf.write('    {\n')
            outf.write('        "size":256,\n')
            outf.write('        "name":"hidden_layer_1",\n')
            outf.write('        "bias":1.0,\n')
            outf.write('        "type":"feedforward_logistic"\n')
            outf.write('    },\n')
            outf.write('    {\n')
            outf.write('        "size":256,\n')
            outf.write('        "name":"hidden_layer_2",\n')
            outf.write('        "bias":1.0,\n')
            outf.write('        "type":"feedforward_logistic"\n')
            outf.write('    },\n')
            outf.write('    {\n')
            outf.write('        "size":256,\n')
            outf.write('        "name":"hidden_layer_3",\n')
            outf.write('        "bias":1.0,\n')
            outf.write('        "type":"blstm"\n')
            outf.write('    },\n')
            outf.write('    {\n')
            outf.write('        "size":256,\n')
            outf.write('        "name":"hidden_layer_4",\n')
            outf.write('        "bias":1.0,\n')
            outf.write('        "type":"blstm"\n')
            outf.write('    },\n')
            outf.write('    {\n')
            outf.write('        "size":116,\n')
            outf.write('        "name":"output_layer",\n')
            outf.write('        "bias":1.0,\n')
            outf.write('        "type":"softmax"\n')
            outf.write('    },\n')
            outf.write('    {\n')
            outf.write('        "size":116,\n')
            outf.write('        "name":"postoutput_layer",\n')
            outf.write('        "type":"multiclass_classification"\n')
            outf.write('    }\n')
            outf.write('    ]\n')
            outf.write('}\n')

    def trainconfig(self):
        self.logger.info('Writing train config.')
        with codecs.open(self.projectfolder+'/configs/config.cfg','w','utf-8') as outf:
            outf.write('network = '+self.projectfolder+'/networks/network.jsn\n')
            outf.write('cuda = true\n')
            outf.write('list_devices = false\n')
            outf.write('parallel_sequences = 10\n')
            outf.write('random_seed = 0\n')
            outf.write('train = true\n')
            outf.write('stochastic = true\n')
            outf.write('shuffle_fractions = true\n')
            outf.write('shuffle_sequences = false\n')
            outf.write('max_epochs = 300\n')
            outf.write('max_epochs_no_best = 20\n')
            outf.write('validate_every = 2\n')
            outf.write('test_every = 2\n')
            outf.write('optimizer = steepest_descent\n')
            outf.write('learning_rate = 1e-4\n')
            outf.write('momentum = 0.9\n')
            outf.write('weight_noise_sigma = 0.0\n')
            outf.write('save_network = '+self.projectfolder+'/autosave/trained_network.jsn\n')
            outf.write('autosave = false\n')
            outf.write('autosave_best = false\n')
            outf.write('autosave_prefix = '+self.projectfolder+'/autosave/mynn\n')
            outf.write('train_file = '+self.projectfolder+'/data/trainset.nc\n')
            outf.write('test_file = '+self.projectfolder+'/data/testset.nc\n')
            outf.write('val_file = '+self.projectfolder+'/data/validationset.nc\n')
            outf.write('train_fraction = 1\n')
            outf.write('test_fraction = 1\n')
            outf.write('val_fraction = 1\n')
            outf.write('truncate_seq = 0\n')
            outf.write('input_noise_sigma = 0\n')
            outf.write('input_right_context = 0\n')
            outf.write('input_left_context = 0\n')
            outf.write('output_time_lag = 0\n')
            outf.write('cache_path = '+self.projectfolder+'/tmp/\n')
            outf.write('weights_dist = normal\n')
            outf.write('weights_normal_mean = 0\n')

    def testconfig(self):
        self.logger.info('Writing test config.')
        with codecs.open(self.projectfolder+'/configs/test.cfg','w','utf-8') as outf:
            outf.write('train = false\n')
            outf.write('network = '+self.projectfolder+'/autosave/trained_network.jsn\n')
            outf.write('ff_output_format = csv\n')
            outf.write('ff_input_file = '+self.projectfolder+'/data/testset.nc\n')
            outf.write('parallel_sequences = 10\n')
            outf.write('cache_path = '+self.projectfolder+'/tmp/\n')
            outf.write('revert_std = false\n')
            outf.write('input_left_context = 0\n')
            outf.write('input_right_context = 0\n')

    def dealwithcsv(self,csvfolder,framefile,numfile,tagfile):
        if len(self.posdct.keys()) == 0:
            self.tardct(self.projectfolder+'/dct/posinfo')
        if len(self.vecdct.keys()) == 0:
            self.vectorfromfile(self.projectfolder+'/dct/pkuvec')
        self.logger.info('Start to deal with csv file')
        posdct = sorted(self.posdct.iteritems(),key=lambda asd:asd[1], reverse=True)
        with codecs.open(numfile,'w','utf-8') as numo,codecs.open(tagfile,'w','utf-8') as tago:
            frame = codecs.open(framefile,'r','utf-8').readlines()
            for i in range(0,len(frame)):
                filename = csvfolder+'/'+'%09d'%(i+1)+'.csv'
                tago.write(filename+'\n')
                lines = genfromtxt(filename,delimiter=';')
                if str(len(lines)) != frame[i].strip():
                    self.logger.debug('File: '+filename+'  ground truth frame number is '+frame[i]+' but csv line is '+str(len(lines)))
                    numo.write(str(np.argmax(lines))+'\n')
                    tago.write(posdct[np.argmax(lines)][0]+'\n')
                else:
                    for line in lines:
                        numo.write(str(np.argmax(line))+'\n')
                        tago.write(posdct[np.argmax(line)][0]+'\n')
        self.logger.info('Done testnumfile and testtagfile is there.')

    def calculate(self,groundfile,predictfile):
        ground = codecs.open(groundfile,'r','utf-8').readlines()
        predict = codecs.open(predictfile,'r','utf-8').readlines()
        correct = 0
        if len(ground) != len(predict):
            self.logger.error('The length of ground truth and predict file is not equal.')
            sys.exit()

        for i in range(0,len(ground)):
            if ground[i] == predict[i]:
                correct += 1
        acc = float(correct)/float(len(ground))
        self.logger.info('correct items length is '+str(correct))
        self.logger.info('all items length is '+str(len(ground)))
        self.logger.info('Pos acc is '+str(acc))

    def process(self,options):
        self.projectfolder = options['projectfolder']
        """control the process based on buttons """

        if options['cwf'] == '1':
            self.createworkfolder(options['projectfolder'])
        if options['pwt'] == '1':
            self.prepareVecTrainData(options['corpus'])
        if options['twv'] == '1':
            self.trainWord2Vec(options)
        if options['cdn'] == '1':
            self.convert2num(options)
        if options['wnc'] == '1':
            self.logger.info('Start to write nc file.')
            ## Write train set number data to nc file.
            trainoptions = {}
            trainoptions['feature'] = self.projectfolder+'/data/traininput'
            trainoptions['target'] = self.projectfolder +'/data/trainoutput'
            trainoptions['frame'] = self.projectfolder +'/data/trainframe'
            trainoptions['outname'] = self.projectfolder +'/data/trainset.nc'
            self.write2nc(trainoptions)
            ## Write validation set number data to nc file.
            validationoptions = {}
            validationoptions['feature'] = self.projectfolder+'/data/validationinput'
            validationoptions['target'] = self.projectfolder+'/data/validationoutput'
            validationoptions['frame'] = self.projectfolder+'/data/validationframe'
            validationoptions['outname'] = self.projectfolder+'/data/validationset.nc'
            self.write2nc(validationoptions)
            ## Write test set number data to nc file.
            testoptions ={}
            testoptions['feature'] = self.projectfolder+'/data/testinput'
            testoptions['target'] = self.projectfolder+'/data/testoutput'
            testoptions['frame'] = self.projectfolder+'/data/testframe'
            testoptions['outname'] = self.projectfolder+'/data/testset.nc'
            self.write2nc(testoptions)
            self.logger.info('NC files has prepared well.')

        if options['scf'] == '1':
            self.network()
            self.trainconfig()
            self.testconfig()

        if options['tnn'] == '1':
            self.logger.info('Start to training LSTM-RNN POS Tagging Model.')
            command = options['currennt']+' '+self.projectfolder+'/configs/config.cfg'
            os.system(command)
            self.logger.info('LSTM-RNN POS Tagging Model has been trained well.')
            
        if options['tes'] == '1':
            self.logger.info('Start to testing LSTM-RNN POS Tagging Model.')
            command = options['currennt']+' '+self.projectfolder+'/configs/test.cfg'
            os.system(command)
            self.logger.info('CSV files has generated in ff_output.csv folder.')
            
        if options['csv'] == '1':
            self.dealwithcsv('ff_output.csv',self.projectfolder+'/data/testframe',self.projectfolder+'/data/testnumfile',self.projectfolder+'/data/testtagfile')

        if options['acc'] == '1':
            self.calculate(self.projectfolder+'/data/testoutput',self.projectfolder+'/data/testnumfile')

        self.logger.info('All works have done, please enjoy LSTM-RNN POS tagging. ')

if __name__=='__main__':
    import sys
    import logging
    import logging.config
    from argparse import ArgumentParser

    formatter = logging.Formatter("*%(levelname)s* | %(filename)s | %(message)s")
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)

    file_handler = logging.FileHandler('log','w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    parser = ArgumentParser(description="Hi boy, lst's try lstm-rnn based pos tagging.")
    parser.add_argument("-c1","--cwf",action="store",dest="cwf",nargs="?",default="1",help="cwf")
    parser.add_argument("-p","--pwt",action="store",dest="pwt",nargs="?",default="1",help="pwt")
    parser.add_argument("-t1","--twv",action="store",dest="twv",nargs="?",default="1",help="twv")
    parser.add_argument("-c2","--cdn",action="store",dest="cdn",nargs="?",default="1",help="cdn")
    parser.add_argument("-w","--wnc",action="store",dest="wnc",nargs="?",default="1",help="wnc")
    parser.add_argument("-s","--scf",action="store",dest="scf",nargs="?",default="1",help="scf")
    parser.add_argument("-t2","--tnn",action="store",dest="tnn",nargs="?",default="1",help="tnn")
    parser.add_argument("-t3","--tes",action="store",dest="tes",nargs="?",default="1",help="tes")
    parser.add_argument("-c3","--csv",action="store",dest="csv",nargs="?",default="1",help="csv")
    parser.add_argument("-a","--acc",action="store",dest="acc",nargs="?",default="1",help="acc")

    args = parser.parse_args()
    options = vars(args)

    lstmpos = LstmPosTagging(logger,options)
    lstmpos.process(options)
