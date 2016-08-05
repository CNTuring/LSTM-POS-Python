#!/usr/bin/python2
#@Author: Pengcheng Zhu
#@Date: 6/16/2016
#@Copyright: Nuance.Inc 2016
import codecs
import os
import logging
import numpy as np
from numpy import genfromtxt
from StringIO import StringIO
from netCDF4 import Dataset
import ConfigParser
"""
      This script is the main script for testing online, one sentence in , one sentence with tag out
      Format: python LstmPosTaggingonline-Alpha.py sentencefile outfilename
      Example:
            sentencefile:
            outfilename:
"""
class LstmPosTagOnline(object):
    def __init__(self,logger,options):
        self.logger = logger
        self.parseConfig('config',options)
        self.vecdct = {}
        self.posdct = {}

    def parseConfig(self,confName,options):
        cf = ConfigParser.ConfigParser()
        cf.read(confName)
        count = 0
        s = "lstmpostaggingonline"
        if cf.has_section(s):
            for o in cf.options(s):
                count += 1
                options[o] = cf.get(s,o)
        self.logger.info("Configs(%d options) Loaded ..." % count)

    def checkfile(self):
        if not os.path.exists('autosave/trained_network.jsn'):
            self.logger.error('Please put trained well network in autosave/trained_network.jsn')
            sys.exit()
        
        if not os.path.exists('dct/pkuvec'):
            self.logger.error('word2vec dictionary can not be found, please put it into dct folder and rename filename to pkuvec')
            sys.exit()
        
        if not os.path.exists('dct/posinfo'):
            self.logger.error('pos dictionary can not be found, please put it into dct folder and rename filename to posinfo')
            sys.exit()
            
    def loaddct(self):
        self.logger.info('Get vectors from file.')
        with codecs.open('dct/pkuvec','r','utf-8') as inf:
            line = inf.readline()
            line = inf.readline()
            while(line):
                sline = line.strip().split(' ')
                self.vecdct[sline[0].strip()]=' '.join(sline[1:])
                line = inf.readline()
        self.logger.info('Vec dct length is :'+str(len(self.vecdct.keys())))
        self.logger.info('Vector dct has been loaded.')
        self.logger.info('Form pos class dct from file.')
        with codecs.open('dct/posinfo','r','utf-8') as inf:
            dct = inf.readlines()
            for i,line in enumerate(dct):
                sline = line.strip().split('\t')
                self.posdct[sline[0]] = str(i)
        self.logger.info('Pos class dct has been loaded.')
    
    def digitalize(self,sentencefile):
        with codecs.open('data/onlineinput','w','utf-8') as ipos, codecs.open('data/onlineframe','w','utf-8') as frame:
            self.logger.info('Now, conver test sentence to digital format.')
            with codecs.open(sentencefile,'r','utf-8') as inf:
                line = inf.readline()
                while(line):
                    sline = line.strip().split(' ')
                    for i in range(0,len(sline)):
                        item = sline[i]
                        if self.vecdct.get(item.strip()) != None:
                            ipos.write(self.vecdct.get(item.strip())+'\n')
                        else:
                            ipos.write(self.vecdct.get('</s>')+'\n')
                    frame.write(str(len(sline))+'\n')
                    line = inf.readline()
            self.logger.info('All data has been conveted to digital format.')
            
    def write2nc(self):
        feature = 'data/onlineinput'
        frame = 'data/onlineframe'
        outname = 'data/online.nc'
        numSeqs =0
        numTimesteps =0
        inputPattSize = 100
        numLabels = 116
        framedata = codecs.open(frame,'r','utf-8').readlines()
        numSeqs = len(framedata)
        inlength = 0
        outlength = 0
        maxSeqTagLength = 0
        with codecs.open(feature,'r','utf-8') as fidfea:
            inline = fidfea.readline()
            while(inline):
                inlength = inlength + 1
                inline = fidfea.readline()
        
        numTimesteps = inlength
        maxSeqTagLength = 10
        if len(str(numSeqs))>10:
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
        with codecs.open(feature,'r','utf-8') as fidfea, codecs.open(frame,'r','utf-8') as fidframe:
            ffinfo = fidframe.readline()
            while(ffinfo):
                filelength = int(ffinfo.strip())
                filename = '1'+"%09d"%(fileindex+1) 
                for i in range(0,filelength):
                    inputdata = fidfea.readline()
                    mat_lab = np.mat(np.genfromtxt(StringIO(inputdata.strip()),delimiter=" "))
                    inputsId[frameindex:frameindex+1,:] = mat_lab
                    targetClassesId[frameindex] = 0
                    frameindex = frameindex + 1
                    seqTagsId[fileindex:fileindex+1,0:maxSeqTagLength] = filename
                    seqLengthId[fileindex:fileindex+1] = filelength
                ffinfo = fidframe.readline()
                fileindex = fileindex + 1
        ncid.close()
    def testconfig(self):
        self.logger.info('Writing test config.')
        with codecs.open('configs/online.cfg','w','utf-8') as outf:
            outf.write('train = false\n')
            outf.write('network = autosave/trained_network.jsn\n')
            outf.write('ff_output_format = csv\n')
            outf.write('ff_input_file = data/online.nc\n')
            outf.write('cache_path = tmp/\n')
            outf.write('revert_std = false\n')
            outf.write('input_left_context = 0\n')
            outf.write('input_right_context = 0\n')
            
    def dealwithcsv(self,csvfolder,framefile,sentencefile,outfilename):
        self.logger.info('Start to convert csv file to outfile.')
        posdct = sorted(self.posdct.iteritems(),key=lambda asd:asd[1], reverse=True)
        with codecs.open(sentencefile,'r','utf-8') as sen,codecs.open(outfilename,'w','utf-8') as outf:
            frame = codecs.open(framefile,'r','utf-8').readlines()
            for i in range(0,len(frame)):
                filename = csvfolder + '/1'+"%09d"%(i+1)+'.csv'
                lines =  genfromtxt(filename,delimiter=';')
                sline = sen.readline().strip().split(' ')
                if len(sline) != len(lines):
                    self.error('sentencefile length and csv file length is not equal.')
                    
                if str(len(lines)) != frame[i].strip():
                    self.logger.error('File: '+filename+' frame number is '+frame[i]+' but csv line is '+str(len(lines)))
                    outf.write(sline[0]+posdct[np.argmax(lines)][0]+'\n')
                else:
                    for j,line in enumerate(lines):
                        outf.write(sline[j]+'/'+posdct[np.argmax(line)][0]+' ')
                    outf.write('\n')
        self.logger.info('Done!please check outfile.')
        
    def process(self,options):
        self.checkfile()
        self.loaddct()
        self.digitalize(options['sentencefile'])
        self.write2nc()
        self.testconfig()
        self.logger.info('Running currennnt to get output csv.')
        command = options['currennt']+' configs/online.cfg'
        os.system(command)
        self.logger.info('csv files has generated.')
        self.dealwithcsv('ff_output.csv','data/onlineframe',options['sentencefile'],options['outfilename'])
        self.logger.info('Hey, enjoy research, enjoy high-tech.')
        
    
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
    parser.add_argument("-c","--cur",action="store",dest="currennt",nargs="?",default="",help="currennt")
    parser.add_argument("-s","--sen",action="store",dest="sentencefile",nargs="?",default="",help="sentencefile")
    parser.add_argument("-o","--out",action="store",dest="outfilename",nargs="?",default="",help="outfilename")
    
    args = parser.parse_args()
    options = vars(args)
    
    lstmpos = LstmPosTagOnline(logger,options)
    lstmpos.process(options)