# LSTM-POS-Python
using LSTM POS to train and predict Chinese POS tags
LSTM-RNN POS Tagging is written in Python and runs on Linux. It requires a CUDA-capable graphics crad with a compute capability of 1.3 or higher (i.e. at
least a GeForce 210, GT 220/40, FX380 LP, 1800M, 370/380M or NVS 2/3100M)

This is the first alpha beta. Research use only.

If u find some problem, please contact author : 
Name:       Pengcheng Zhu 
Position:   TTS Research && Development Engineer
Email:      Pengcheng.Zhu@nuance.com / Pczhu@nwpu-aslp.org

+===============================================================================+
| Currennt Install                                                              |
+===============================================================================+
Building on Linux requires the following:
* CUDA Toolkit 5.0
* GCC 4.6 or higher
* NetCDF scientific data library
* Boost library 1.48 or higher
Link: https://sourceforge.net/projects/currennt/files/?source=navbar

To build Currennt execute the following commands:
  cd currennt<br>
  mkdir build && cd build<br>
  cmake ..<br>
  make<br>

+===============================================================================+
| Word2vec                                                                      |
+===============================================================================+
Source file link :http://pan.baidu.com/s/1qYnFOZi
using command :
make <br>
to compile, and exe word2vec can running.<br>

+===============================================================================+
| Python Toolkit Emviroment                                                     |
+===============================================================================+
To run this script successfully we need toolkits as following:
* StringIO
* numpy
* netCDF4
* logging
* ConfigParser

All toolkit can be setup by pip,except netCDF4:
Format: pip install [toolkit name]

netCDF4 Setup:
ref:https://pythonhosted.org/cdb_query/install_source.html
commands:<br>
  pip install numpy<br>
  pip install sqlalchemy<br>
  pip install Cython<br>
  pip install python-dateutil<br>
   wget --no-check-certificate \<br>
      https://pypi.python.org/packages/source/v/virtualenv/virtualenv-1.11.2.tar.gz<br>
  tar xvfz virtualenv-1.11.2.tar.gz<br>
  cd virtualenv-1.11.2<br>
  python virtualenv.py $HOME/python<br>
  source $HOME/python/bin/activate<br>
  pip install numpy<br>
  pip install sqlalchemy<br>
  pip install Cython<br>
  pip install python-dateutil<br>
  export USE_NCCONFIG=1;pip install netcdf4<br>
If there has some problems,please use:<br>
  pip install h5py<br>
  pip install netcdf4<br>

+===============================================================================+
| Trainning Model and test                                                      |
+===============================================================================+
1. put corpus in folder 'corpus'<br>
2. set configs in file config:<br>
   Example:<br>
   [lstmpostagging]<br>
   word2vec=./toolkit/word2vec<br>
   currennt=./toolkit/currennt<br>
   word2vecsize=100<br>
   trainlist=1998-01.txt,1998-02.txt<br>
   validationlist=1998-03.txt<br>
   testlist=1998-04.txt<br>
   corpus=corpus<br>
   projectfolder=.<br>
   CWF=1<br>
   PWT=1<br>
   TWV=1<br>
   CDN=1<br>
   WNC=1<br>
   SCF=1<br>
   TNN=1<br>
   TES=1<br>
   CSV=1<br>
   ACC=1<br>
   Note:<br>
     1.  [cwf]:create working folders<br>
     2.  [pwt]:prepare word2vec training data and statistical corpus POS classes<br>
     3.  [twv]:train word2vec<br>
     4.  [cdn]:convert corpus data to number format, input is vector data and output is class number<br>
     5.  [wnc]:write data to NC format (if u use currennt as LSTM-RNN toolkit)<br>
     6.  [scf]: set network and other configs<br>
     7.  [tnn]:train LSTM-RNN model<br>
     8.  [tes]:test LSTM-RNN model, model input: nc file, model output: csv file<br>
     9.  [CSV]:change csv file to POS Tags<br>
     10. [ACC]:calculate accuracy<br>
     11. [word2vec]:word2vec toolkit position<br>
     12. [currennt]:currennt postion<br>
     13. [word2vecsize]: vector size in word2vec <br>
     14. [trainlist]:train file names, each file use ',' separating<br>
     15. [validationlist]:validation file names,each file use ',' separating<br>
     16. [testlist]:test file names, each file use ',' separating<br>
     17. [corpus]:corpus folder<br>
     18. [projectfolder]: whole project prosition<br>
3. Command:<br>
\>source $HOME/python/bin/activate<br>
\>python LstmPosTagging-Alpha.py<br>

+===============================================================================+
| Using trained well model to do test                                           |
+===============================================================================+
1. make sure model in 'autosave/trained_network.jsn'<br>
2. make sure word2vec dictionary in 'dct/pkuvec'<br>
3. make sure pos list in 'dct/posinfo'<br>
4. configs:<br>
   Example:<br>
        [lstmpostaggingonline]<br>
        currennt=./toolkit/currennt<br>
        sentencefile=data/onlinetest.txt<br>
        outfilename=data/onlineout.txt<br>
   Note:<br>
   [currennt]:     currennt position<br>
   [sentencefile]: input text file<br>
   [outfilename]:  output file <br>
5. command:<br>
  source $HOME/python/bin/activate<br>
  python LstmPosTaggingonline-Alpha.py<br>
Tip: U can use log to debug.<br>
