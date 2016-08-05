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
\> cd currennt
\> mkdir build && cd build
\> cmake ..
\> make

+===============================================================================+
| Word2vec                                                                      |
+===============================================================================+
Source file link :http://pan.baidu.com/s/1qYnFOZi
using command :
make 
to compile, and exe word2vec can running.

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
commands:
\>pip install numpy
\>pip install sqlalchemy
\>pip install Cython
\>pip install python-dateutil
\> wget --no-check-certificate \
      https://pypi.python.org/packages/source/v/virtualenv/virtualenv-1.11.2.tar.gz
\>tar xvfz virtualenv-1.11.2.tar.gz
\>cd virtualenv-1.11.2
\>python virtualenv.py $HOME/python
\>source $HOME/python/bin/activate
\>pip install numpy
\>pip install sqlalchemy
\>pip install Cython
\>pip install python-dateutil
\>export USE_NCCONFIG=1;pip install netcdf4
If there has some problems,please use:
\>pip install h5py
\>pip install netcdf4

+===============================================================================+
| Trainning Model and test                                                      |
+===============================================================================+
1. put corpus in folder 'corpus'
2. set configs in file config:
   Example:
   [lstmpostagging]
   word2vec=./toolkit/word2vec
   currennt=./toolkit/currennt
   word2vecsize=100
   trainlist=1998-01.txt,1998-02.txt
   validationlist=1998-03.txt
   testlist=1998-04.txt
   corpus=corpus
   projectfolder=.
   CWF=1
   PWT=1
   TWV=1
   CDN=1
   WNC=1
   SCF=1
   TNN=1
   TES=1
   CSV=1
   ACC=1
   Note:
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
     11. [word2vec]:word2vec toolkit position
     12. [currennt]:currennt postion
     13. [word2vecsize]: vector size in word2vec 
     14. [trainlist]:train file names, each file use ',' separating
     15. [validationlist]:validation file names,each file use ',' separating
     16. [testlist]:test file names, each file use ',' separating
     17. [corpus]:corpus folder
     18. [projectfolder]: whole project prosition
3. Command:
\>source $HOME/python/bin/activate
\>python LstmPosTagging-Alpha.py

+===============================================================================+
| Using trained well model to do test                                           |
+===============================================================================+
1. make sure model in 'autosave/trained_network.jsn'
2. make sure word2vec dictionary in 'dct/pkuvec'
3. make sure pos list in 'dct/posinfo'
4. configs:
   Example:
        [lstmpostaggingonline]
        currennt=./toolkit/currennt
        sentencefile=data/onlinetest.txt
        outfilename=data/onlineout.txt
   Note:
   [currennt]:     currennt position
   [sentencefile]: input text file
   [outfilename]:  output file 
5. command:
\>source $HOME/python/bin/activate
\>python LstmPosTaggingonline-Alpha.py
Tip: U can use log to debug.
