#include <fstream>
#include <cstdlib>
#include <cmath>
#include "pthread.h"

#include "bpn.h"
#include "bpnlayer.h"

BPN::~BPN() {
  if(size >0) {
    if(layers[size-1]->size >0)
      delete [] train_output;

    for(unsigned i=0; i<size; ++i) {
      delete layers[i];
    }

    delete [] layers;
  }

  for(unsigned i=0; i <= threads; ++i) {
    delete li[i];
  }

  delete [] li;
  delete [] thread_id;
}

BPN::BPN(unsigned *sizes, bool *biases, outFunction *functions, unsigned layersNumber, double et, double alph, double o_factor, unsigned thrs) {
  threads = thrs;
  minchunk = 64;
  size = layersNumber;
  alpha = alph;
  eta = et;
  scale_factor = o_factor;
  initial_scale = 0.03f;

  if(size > 0) {
    layers = new bpnLayer*[size];

    layers[0] = new bpnLayer(sizes[0], 0, false, functions[0], threads + 1);   //  input layer - no lower layer

    for(unsigned i=1; i<size; ++i) {
      layers[i] = new bpnLayer(sizes[i], sizes[i-1], biases[i], functions[i], threads + 1);
    }

    unsigned lowerSize = layers[size-1]->size;

    if(lowerSize >0) {
      train_output = new double[lowerSize];
      for(unsigned j=0; j < lowerSize; ++j) {
	train_output[j] = 0.0f;
      }
    }

    InitializeWeights();
  }

  li = new LayerThread*[threads + 1];
  for(unsigned i=0; i <= threads; ++i) {
    li[i] = new LayerThread;
  }

  thread_id = new pthread_t[threads];
}

BPN::BPN(const char* file, unsigned thrs) {
  threads = thrs;
  minchunk = 64;
  unsigned i;

  li = new LayerThread*[threads];
  for(unsigned i=0; i <= threads; ++i) {
    li[i] = new LayerThread;
  }

  thread_id = new pthread_t[threads];

  std::ifstream fin;
  fin.open(file, std::ifstream::in);

  readString(fin);
  fin>>initial_scale;

  readString(fin);
  fin>>eta;

  readString(fin);
  fin>>alpha;

  readString(fin);
  fin>>scale_factor;

  readString(fin);
  fin>>size;

  if(size >0) {
    unsigned f, layersize, prevsize=0;
    bool bias;

    //  input layer - no bias
    readString(fin);
    readString(fin);
    fin>>layersize;

    readString(fin);
    fin>>bias;

    readString(fin);
    fin>>f;

    bias = false;
    layers = new bpnLayer*[size];
    layers[0] = new bpnLayer(layersize, prevsize, bias, (outFunction) f, threads + 1);
    prevsize = layersize;

    for(i=1; i<size; ++i) {  //  itterate all layer above input
      readString(fin);
      readString(fin);
      fin>>layersize;

      readString(fin);
      fin>>bias;

      readString(fin);
      fin>>f;

      layers[i] = new bpnLayer(layersize, prevsize, bias, (outFunction) f, threads + 1);
      prevsize = layersize;
    }

    unsigned lowerSize = layers[size-1]->size;

    if(lowerSize > 0)
      {
	train_output = new double[lowerSize];
	for(i=0; i < lowerSize; ++i) {
	  train_output[i] = 0.0f;
	}
      }

    char ch;
    fin>>ch;

    if(ch =='x') {
      fin.close();
      InitializeWeights();
      return;
    }

    for(i=0; i<size; ++i) {
      LoadLayer(layers[i], fin);
    }
  }

  fin.close();
}

void BPN::LoadLayer(bpnLayer* layer, std::ifstream& fin) {
  unsigned i, j;
  unsigned sizeL = layer->size;
  unsigned lowerSize = layer->lowerSize;

  if(layer->bias) {
    for(i=0; i < sizeL; ++i) {
      double *weight = layer->weights[i];

      for(j=0; j < lowerSize; ++j) {
	fin>>weight[j];
      }

      fin>>layer->biases[i];
    }
  }
  else {
    for(i=0; i < sizeL; ++i) {
      double *weight = layer->weights[i];

      for(j=0; j < lowerSize; ++j) {
	fin>>weight[j];
      }
    }
  }
}

std::string BPN::readString(std::ifstream& fin) {
  char ch;
  std::string acc="";

  fin.get(ch);
  do {
    fin.get(ch);

    if(ch != ' ')
      acc += ch;
  }
  while(ch != ' ');

  return acc;
}

bool BPN::SaveToFile(const char* file) {
  unsigned i;
  std::ofstream fout;

  try {
    fout.open(file, std::ofstream::out);

    fout<<" init_scale: "<<initial_scale<<" eta: "<<eta<<" alpha: "<<alpha<<" scale_factor: "<<scale_factor<<" layers_num: "<<size;

    for(i=0; i<size; ++i) {
      bpnLayer *l = layers[i];
      fout<<" layer"<<i+1<<": size "<<l->size<<" bias "<<l->bias<<" func "<<(int)l->func;
    }

    fout<<"i\n";
  }
  catch(std::exception ex) {
    return false;
  }

  for(i=0; i<size; ++i) {
    if(!SaveLayer(layers[i], fout)) {
      fout.close();
      return false;
    }
  }

  fout.close();
  return true;
}

bool BPN::SaveLayer(const bpnLayer* layer, std::ofstream& fout) {
  unsigned i, j;
  unsigned sizeL = layer->size;
  unsigned lowerSize = layer->lowerSize;
  double **weigs = layer->weights;

  if(layer->bias) {
    double *bias = layer->biases;

    for(i=0; i < sizeL; ++i) {
      double *weight = weigs[i];

      try {
	for(j=0; j < lowerSize; ++j) {
	  fout<<weight[j]<<" ";
	}

	fout<<bias[i]<<" "<<"\n";
      }
      catch(std::exception ex) {
	return false;
      }
    }
  }
  else {
    for(i=0; i < sizeL; ++i) {
      double *weight = weigs[i];

      try {
	for(j=0; j < lowerSize; ++j) {
	  fout<<weight[j]<<" ";
	}
      }
      catch(std::exception ex) {
	return false;
      }

      fout<<"\n";
    }
  }

  return true;
}

void BPN::InitializeWeights() {
  unsigned i, j, k, hiddenNum =0;

  for(i=1; i < size-1; ++i) {
    hiddenNum += layers[i]->size;
  }

  //    double scale = (double) (pow((double) (0.7f * (double) hiddenNum),
  //                                  (double) (1.0f / (double) layers[0]->size)))/scale_factor;
  //
  //    scale = 0.03f;

  for(i=1; i < size; ++i) {  //  itterate layers above input
    bpnLayer *l = layers[i];
    unsigned sizeL = l->size;
    unsigned lowerSize = l->lowerSize;
    double **weigs = l->weights;

    if(l->bias) {
      for(j=0; j < sizeL; ++j) {
	double *weight = weigs[j];

	for(k=0; k < lowerSize; ++k) {
	  weight[k] = randomNum(-initial_scale, initial_scale);
	}

	l->biases[j] = randomNum(-initial_scale, initial_scale);
      }
    }
    else {
      for(j=0; j < sizeL; ++j) {
	double *weight = weigs[j];

	for(k=0; k < lowerSize; ++k) {
	  weight[k] = randomNum(-initial_scale, initial_scale);
	}
      }
    }
  }
}

bool BPN::Train(double *input, double* output) {
  Run(input);
  double (*derivate) (double); //  pointer to derivate funtion
  bpnLayer *l = layers[size-1];
  bpnLayer *under_l = layers[size-2];
  unsigned sizeL = l->size;
  unsigned lowerSize = under_l->size;
  double *prods = l->products[0];
  double *under_prods = under_l->products[0];
  double *errs = l->errors;
  double **delts = l->deltas;

  switch(l->func) {
  case sigmoid:
    derivate = DerivateSigmoid;
    break;
  case sigmoid2:
    derivate = DerivateSigmoid2;
    break;
  default:
    derivate = DerivateLinear;
  }

  for(unsigned i=0; i < sizeL; ++i) { //  fill output errors
    errs[i] = (double)(*derivate)(prods[i]/scale_factor)
      * (output[i] - prods[i])/scale_factor;

    double *delta = delts[i];
    double error = errs[i];

    for(unsigned j=0; j < lowerSize; ++j) {
      delta[j] = under_prods[j]*eta*error + alpha*delta[j];
    }
  }

  return Train();
}

void BPN::Run(const double* input) {
  double *prods = layers[0]->products[0];
  unsigned sizeL = layers[0]->size;

  for(unsigned i=0; i < sizeL; ++i) {   //  fill input
    prods[i] = input[i];
  }

  Run();
}

void BPN::Run(const double* input, unsigned threadid) {
  double *prods = layers[0]->products[threadid];
  unsigned sizeL = layers[0]->size;

  for(unsigned i=0; i < sizeL; ++i) {   //  fill input
    prods[i] = input[i];
  }

  Run(threadid);
}

void BPN::PrepareFromFEN(const char* fen, bool training) {
  PrepareFromFEN(fen, 0, training);
}

void BPN::PrepareFromFEN(const char* fen, unsigned threadid, bool training) {
  int i=0;
  int j=0;
  char ch;
  double *prods = layers[0]->products[threadid];

  while(i<256 && j<100) {	//	this is for 64 squares
    ch = fen[j];

    switch(ch) {
    case 'P':	//	white pawn - code 0001
      prods[i++] = 0.0f;
      prods[i++] = 0.0f;
      prods[i++] = 0.0f;
      prods[i++] = 1.0f;
      break;
    case 'p':	//	black pawn - code 1001
      prods[i++] = 1.0f;
      prods[i++] = 0.0f;
      prods[i++] = 0.0f;
      prods[i++] = 1.0f;
      break;
    case 'N':	//	white kNight - code 0010
      prods[i++] = 0.0f;
      prods[i++] = 0.0f;
      prods[i++] = 1.0f;
      prods[i++] = 0.0f;
      break;
    case 'n':	//	black kNight - code 1010
      prods[i++] = 1.0f;
      prods[i++] = 0.0f;
      prods[i++] = 1.0f;
      prods[i++] = 0.0f;
      break;
    case 'B':	//	white bishop - code 0011
      prods[i++] = 0.0f;
      prods[i++] = 0.0f;
      prods[i++] = 1.0f;
      prods[i++] = 1.0f;
      break;
    case 'b':	//	black bishop - code 1011
      prods[i++] = 1.0f;
      prods[i++] = 0.0f;
      prods[i++] = 1.0f;
      prods[i++] = 1.0f;
      break;
    case 'R':	//	white rook - code 0100
      prods[i++] = 0.0f;
      prods[i++] = 1.0f;
      prods[i++] = 0.0f;
      prods[i++] = 0.0f;
      break;
    case 'r':	//	black rook - code 1100
      prods[i++] = 1.0f;
      prods[i++] = 1.0f;
      prods[i++] = 0.0f;
      prods[i++] = 0.0f;
      break;
    case 'Q':	//	white queen - code 0101
      prods[i++] = 0.0f;
      prods[i++] = 1.0f;
      prods[i++] = 0.0f;
      prods[i++] = 0.0f;
      break;
    case 'q':	//	black queen - code 1101
      prods[i++] = 1.0f;
      prods[i++] = 1.0f;
      prods[i++] = 0.0f;
      prods[i++] = 1.0f;
      break;
    case 'K':	//	white king - code 0110
      prods[i++] = 0.0f;
      prods[i++] = 1.0f;
      prods[i++] = 1.0f;
      prods[i++] = 0.0f;
      break;
    case 'k':	//	black king - code 1110
      prods[i++] = 1.0f;
      prods[i++] = 1.0f;
      prods[i++] = 1.0f;
      prods[i++] = 0.0f;
      break;
    }

    if(ch <= '8' && ch>= '1') {
      int emptySquares = ch - '0';
      emptySquares *= 4;

      for(int k=0; k < emptySquares; ++k) {
	prods[i++] = 1.0f;
      }
    }

    ++j;
  }

  ch = fen[j];
  while(ch == ' ') {
    ch = fen[++j];
  }

  if(ch == 'w' || ch == 'W') prods[256] = 1.0f;	//	set active colour bit - i == 256
  else prods[256] = 0.0f;

  do {
    ++j;
    ch = fen[j];
  }while(ch == ' ');

  for(i=257; i<261; ++i)
    prods[i] = 0.0f;

  while(ch != '-' && ch != ' ') {
    switch(ch) {
    case 'K':	//	White can castle kingside
      prods[257] = 1.0f;
      break;
    case 'Q':	//	White can castle queenside
      prods[258] = 1.0f;
      break;
    case 'k':	//	Black can castle kingside
      prods[259] = 1.0f;
      break;
    case 'q':	//	Black can castle queenside
      prods[260] = 1.0f;
      break;
    }

    ++j;
    ch = fen[j];
  }

  while(fen[j] == ' ')
    ++j;

  if(fen[j] =='-')
    prods[261] = 0.0f;
  else    //  en passant possible
    prods[261] = 1.0f;

  if (training) {
    while(fen[j] != '[')
      ++j;

    train_output[0] = atof((const char*)(fen + j + 1));
  }
}

void* BPN::UnitThreadFuncBias (void *arg) {
  double net;
  LayerThread *lt = (LayerThread*) arg;
  unsigned k, layer = lt->layer;
  bpnLayer *l = lt->bp->layers[layer];
  unsigned lowerSize = l->lowerSize;
  unsigned end = lt->end;
  bpnLayer *under_l = lt->bp->layers[layer - 1];
  double (*apply) (double);
  apply = lt->apply;
  double **weigs = l->weights;
  double *prods = l->products[0];
  double *under_prods = under_l->products[0];
  double *bias = l->biases;

  for(unsigned j = lt->start; j < end; ++j) {   //  for each unit - compute product
    net = 0;

    double *weight = weigs[j];
    for(k=0; k < lowerSize; ++k) {
      net += (double)weight[k] * under_prods[k];
    }

    net += bias[j];
    prods[j] = (*apply)(net);
  }

  return (void*) true;
}

void* BPN::UnitThreadFunc (void *arg) {
  double net;
  LayerThread *lt = (LayerThread*) arg;
  unsigned k, layer = lt->layer;
  bpnLayer *l = lt->bp->layers[layer];
  unsigned lowerSize = l->lowerSize;
  unsigned end = lt->end;
  bpnLayer *under_l = lt->bp->layers[layer - 1];
  double (*apply) (double);
  apply = lt->apply;
  double **weigs = l->weights;
  double *prods = l->products[0];
  double *under_prods = under_l->products[0];

  for(unsigned j = lt->start; j < end; ++j) {   //  for each unit - compute product
    net = 0;

    double *weight = weigs[j];
    for(k=0; k < lowerSize; ++k) {
      net += (double)weight[k] * under_prods[k];
    }

    prods[j] = (*apply)(net);
  }

  return (void*) true;
}

void* BPN::UnitThreadFuncBiasScale (void *arg) {
  double net;
  LayerThread *lt = (LayerThread*) arg;
  unsigned k, layer = lt->layer;
  bpnLayer *l = lt->bp->layers[layer];
  unsigned lowerSize = l->lowerSize;
  unsigned end = lt->end;
  bpnLayer *under_l = lt->bp->layers[layer - 1];
  double (*apply) (double);
  apply = lt->apply;
  double scale_factor = lt->bp->scale_factor;
  double **weigs = l->weights;
  double *prods = l->products[0];
  double *under_prods = under_l->products[0];
  double *bias = l->biases;

  for(unsigned j = lt->start; j < end; ++j) {   //  for each unit - compute product
    net = 0;

    double *weight = weigs[j];
    for(k=0; k < lowerSize; ++k) {
      net += (double)weight[k] * under_prods[k];
    }

    net += bias[j];
    prods[j] = (*apply)(net) * scale_factor;
  }

  return (void*) true;
}

void* BPN::UnitThreadFuncScale (void *arg) {
  double net;
  LayerThread *lt = (LayerThread*) arg;
  unsigned k, layer = lt->layer;
  bpnLayer *l = lt->bp->layers[layer];
  unsigned lowerSize = l->lowerSize;
  unsigned end = lt->end;
  bpnLayer *under_l = lt->bp->layers[layer - 1];
  double (*apply) (double);
  apply = lt->apply;
  double scale_factor = lt->bp->scale_factor;
  double **weigs = l->weights;
  double *prods = l->products[0];
  double *under_prods = under_l->products[0];

  for(unsigned j = lt->start; j < end; ++j) {   //  for each unit - compute product
    net = 0;

    double *weight = weigs[j];
    for(k=0; k < lowerSize; ++k) {
      net += (double)weight[k] * under_prods[k];
    }

    prods[j] = (*apply)(net) * scale_factor;
  }

  return (void*) true;
}

bool BPN::DoThreading(unsigned layer, double (*apply) (double), void* (*thrFunc)(void*)) {
  unsigned i, layer_size = layers[layer]->size;
  unsigned nthreads = layer_size > threads ? threads : layer_size - 1;
  unsigned chunk = layer_size / (nthreads+1);
  bool retval, ok = true;

  if(nthreads && chunk < minchunk) { // decrease number of threads for bigger chunks
    if(layer_size < 2*minchunk) {
      nthreads = 0;
    }
    else {
      nthreads = layer_size/minchunk - 1;
      chunk = layer_size / (nthreads+1);
    }
  }

  //pthread_t thread_id[threads];

  unsigned prev_end = 0;
  for(i=0; i<nthreads; ++i) {	// create threads
    LayerThread *lt = li[i];

    lt->layer = layer;
    lt->apply = apply;
    lt->start = prev_end;	//i*chunk
    prev_end += chunk;
    lt->end = prev_end;	//(i+1)*chunk
    lt->bp = this;

    pthread_create(&thread_id[i], NULL, thrFunc, lt);
  }

  // finish what's left in the main thread
  if (prev_end < layer_size) {	// li[nthreads]->start < li[nthreads]->end
    LayerThread *lt = li[nthreads];

    lt->start = prev_end; //nthreads*chunk
    lt->end = layer_size;
    lt->layer = layer;
    lt->apply = apply;
    lt->bp = this;

    retval = (*thrFunc)(lt);
    ok = ok && retval;
  }

  for(i=0; i<nthreads; ++i) {
    pthread_join(thread_id[i], (void**) &retval);
    ok = ok && retval;
  }

  return ok;
}

// threaded version
void BPN::Run() {//  assume that input is already placed in the first layer
  double (*apply) (double); //  pointer to output funtion

  for(unsigned i=1; i<size-1; ++i) { //  for each layer above input
    //  determine output function before iterating through units - it's always same function for the whole layer
    switch(layers[i]->func) {
    case sigmoid:
      apply = ApplySigmoid;
      break;
    case sigmoid2:
      apply = ApplySigmoid2;
      break;
    default:
      apply = ApplyLinear;
    }

    if(layers[i]->bias)
      DoThreading(i, apply, UnitThreadFuncBias);
    else
      DoThreading(i, apply, UnitThreadFunc);
  }

  //  same for output layer - though add scale factor
  switch(layers[size-1]->func) {
  case sigmoid:
    apply = ApplySigmoid;
    break;
  case sigmoid2:
    apply = ApplySigmoid2;
    break;
  default:
    apply = ApplyLinear;
  }

  if(layers[size-1]->bias)
    DoThreading(size-1, apply, UnitThreadFuncBiasScale);
  else
    DoThreading(size-1, apply, UnitThreadFuncScale);
}

void BPN::Run(unsigned threadid) { //  assume that input is already placed in the first layer
  unsigned j, k;
  double net, (*apply) (double); //  pointer to output funtion

  for(unsigned i=1; i<size-1; ++i) {  //  for each layer above input
    //  determine output function before iterating through units - it's always same function for the whole layer
    switch(layers[i]->func) {
    case sigmoid:
      apply = ApplySigmoid;
      break;
    case sigmoid2:
      apply = ApplySigmoid2;
      break;
    default:
      apply = ApplyLinear;
    }

    bpnLayer *l = layers[i];
    double *l_prods = l->products[threadid];
    double *under_prods = layers[i-1]->products[threadid];
    unsigned sizeL = l->size;
    unsigned lowerSize = l->lowerSize;

    if(layers[i]->bias) {
      for(j=0; j < sizeL; ++j) {  //  for each unit - compute product
	net = 0;
	double *weight = l->weights[j];

	for(k=0; k < lowerSize; ++k) {
	  net += (double)weight[k] * under_prods[k];
	}

	net += l->biases[j];
	l_prods[j] = (*apply)(net);
      }
    }
    else {
      for(j=0; j < sizeL; ++j) {  //  for each unit - compute product
	net = 0;
	double *weight = l->weights[j];

	for(k=0; k < lowerSize; ++k) {
	  net += weight[k] * under_prods[k];
	}

	l_prods[j] = (*apply)(net);
      }
    }
  }

  //  same for output layer - though add scale factor
  switch(layers[size-1]->func) {
  case sigmoid:
    apply = ApplySigmoid;
    break;
  case sigmoid2:
    apply = ApplySigmoid2;
    break;
  default:
    apply = ApplyLinear;
  }

  bpnLayer *l = layers[size-1];
  double *l_prods = l->products[threadid];
  double *under_prods = layers[size-2]->products[threadid];
  unsigned sizeL = l->size;
  unsigned lowerSize = l->lowerSize;

  if(l->bias) {
    for(j=0; j < sizeL; ++j) {  //  for each unit - compute product
      net = 0;
      double *weight = l->weights[j];

      for(k=0; k < lowerSize; ++k) {
	net += (double)weight[k] * under_prods[k];
      }

      net += l->biases[j];
      l_prods[j] = (*apply)(net)*scale_factor;
    }
  }
  else {
    for(j=0; j < sizeL; ++j) {  //  for each unit - compute product
      net = 0;
      double *weight = l->weights[j];

      for(k=0; k < lowerSize; ++k) {
	net += (double)weight[k] * under_prods[k];
      }

      l_prods[j] = (*apply)(net)*scale_factor;
    }
  }
}

bool BPN::Train(const char* fen) {
  Run(fen, true);
  double (*derivate) (double); //  pointer to derivate funtion
  bpnLayer *l = layers[size-1];
  bpnLayer *under_l = layers[size-2];
  unsigned sizeL = l->size;
  unsigned lowerSize = under_l->size;
  double *prods = l->products[0];
  double *under_prods = under_l->products[0];
  double *errs = l->errors;
  double **delts = l->deltas;

  switch(l->func) {
  case sigmoid:
    derivate = DerivateSigmoid;
    break;
  case sigmoid2:
    derivate = DerivateSigmoid2;
    break;
  default:
    derivate = DerivateLinear;
  }

  unsigned j;
  for(unsigned i=0; i < sizeL; ++i) { //  fill output errors
    errs[i] = (double)(*derivate)(prods[i]/scale_factor)
      * (train_output[i] - prods[i])/scale_factor;

    double *delta = delts[i];
    double error = errs[i];

    for(j=0; j < lowerSize; ++j) {
      delta[j] = under_prods[j]*eta*error + alpha*delta[j];
    }
  }

  return Train();
}

void* BPN::UnitThreadFuncTrain (void *arg) {
  double err;
  LayerThread *lt = (LayerThread*) arg;
  unsigned k, layer = lt->layer;
  bpnLayer *l = lt->bp->layers[layer];
  bpnLayer *under_l = lt->bp->layers[layer - 1];
  bpnLayer *upper_l = lt->bp->layers[layer + 1];
  unsigned upperSize = upper_l->size;
  unsigned lowerSize = under_l->size;
  unsigned end = lt->end;
  double eta = lt->bp->eta, alpha = lt->bp->alpha;
  double (*apply) (double);
  apply = lt->apply;
  double *errs = l->errors;
  double *upper_errs = upper_l->errors;
  double *prods = l->products[0];
  double *under_prods = under_l->products[0];
  double **delts = l->deltas;
  double **weigs = upper_l->weights;

  for(unsigned j = lt->start; j < end; ++j) {   //  for each unit - compute product
    err = 0.0f;

    try {
      for(k=0; k < upperSize; ++k) {   //  for each unit in the upper layer
	err += upper_errs[k] * weigs[k][j];
      }

      l->errors[j] = (double)(*apply)(prods[j])*err;  //  compute error

      double *delta = delts[j];
      double error = errs[j];

      for(k=0; k < lowerSize; ++k) {
	delta[k] = under_prods[k]*eta*error + alpha*delta[k];
      }
    }
    catch(std::exception ex) {
      return (void*) false;
    }
  }

  return (void*) true;
}

void* BPN::UnitThreadFuncRenew (void *arg) {
  LayerThread *lt = (LayerThread*) arg;
  unsigned k, layer = lt->layer;
  bpnLayer *l = lt->bp->layers[layer];
  unsigned lowerSize = l->lowerSize;
  unsigned end = lt->end;
  double **weigs = l->weights;
  double **delts = l->deltas;

  try {
    for(unsigned j = lt->start; j < end; ++j) {   //  for each unit
      double *weight = weigs[j];
      double *delta = delts[j];

      for(k=0; k < lowerSize; ++k) {  //  for each weight - renew
	weight[k] += (double)delta[k];
	if(std::isnan(weight[k]))
	  return (void*) false;
      }
    }
  }
  catch(std::exception ex) {
    return (void*) false;
  }

  return (void*) true;
}

bool BPN::Train() {		// threaded version
  unsigned i;
  double (*derivate) (double); //  pointer to derivate funtion
  bool ok;

  for(i=size-2; i>0; --i) {   //  compute errors and weights' changes - itterate layers from top to bottom
    //  determine derivate function before iterating through units - it's always same derivate for the whole layer
    switch(layers[i]->func) {
    case sigmoid:
      derivate = DerivateSigmoid;
      break;
    case sigmoid2:
      derivate = DerivateSigmoid2;
      break;
    default:
      derivate = DerivateLinear;
    }

    ok = DoThreading(i, derivate, UnitThreadFuncTrain);
    if (!ok) return false;
  }

  for(i=1; i<size; ++i) {  //  for each layer (above input)
    ok = DoThreading(i, NULL, UnitThreadFuncRenew);
    if (!ok) return false;
  }

  return true;
}

/*
  bool BPN::Train() {
  unsigned i, j, k;
  double err;
  double (*derivate) (double); //  pointer to derivate funtion

  for(i=size-2; i>0; --i) {   //  compute errors and weights' changes - itterate layers from top to bottom
  bpnLayer *l = layers[i];
  bpnLayer *upper_l = layers[i+1];
  bpnLayer *under_l = layers[i-1];
  unsigned sizeL = l->size;
  unsigned lowerSize = l->lowerSize;
  unsigned upperSize = upper_l->size;
  double *errs = l->errors;
  double *up_errs = upper_l->errors;
  double **up_weigs = upper_l->weights;
  double *prods = l->products[0];
  double *un_prods = under_l->products[0];
  double **delts = l->deltas;

  //  determine derivate function before iterating through units - it's always same derivate for the whole layer
  switch(l->func) {
  case sigmoid:
  derivate = DerivateSigmoid;
  break;
  case sigmoid2:
  derivate = DerivateSigmoid2;
  break;
  default:
  derivate = DerivateLinear;
  }

  for(j=0; j < sizeL; ++j) {  //  for each unit in the layer
  err = 0.0f;

  try {
  for(k=0; k < upperSize; ++k) {   //  for each unit in the upper layer
  err += up_errs[k] * up_weigs[k][j];
  }

  errs[j] = (double)(*derivate)(prods[j]) * err; //  compute error

  double *delta = delts[j];

  for(k=0; k < lowerSize; ++k) {
  delta[k] = un_prods[k]*eta*errs[j] + alpha*delta[k];
  //layers[i]->weights[j][k] += layers[i]->deltas[j][k];  //  can't do it here because of the lower layer errors
  }
  }
  catch(std::exception ex) {
  return false;
  }
  }
  }

  try {
  for(i=1; i<size; ++i) {  //  for each layer (above input)
  bpnLayer *l = layers[i];
  unsigned sizeL = l->size;
  unsigned lowerSize = l->lowerSize;
  double **delts = l->deltas;
  double **weigs = l->weights;

  for(j=0; j < sizeL; ++j) {   //  for each unit
  double *weight = weigs[j];
  double *delta = delts[j];

  for(k=0; k < lowerSize; ++k) {  //  for each weight - renew
  weight[k] += (double)delta[k];
  if(std::isnan(weight[k])) return false;
  }
  }
  }
  }
  catch(std::exception ex) {
  return false;
  }

  return true;
  }
*/

void BPN::Run(const char* fen, bool training) {
  PrepareFromFEN(fen, training);
  Run();
}

void BPN::Run(const char* fen, unsigned threadid) {
  PrepareFromFEN(fen, threadid, false);
  Run(threadid);
}

double BPN::ApplyLinear(double val) {
  return val;
}

double BPN::ApplySigmoid(double val) {
  return (double)1/(1+exp(-val));
}

double BPN::ApplySigmoid2(double val) {
  return (double)2 / (1 + exp(-2 * val)) - 1;
}

double BPN::DerivateLinear(double) {
  return 1;
}

double BPN::DerivateSigmoid(double val) {
  return val*(1-val);
}

double BPN::DerivateSigmoid2(double val) {
  return (double) 1 - (pow(val, 2));
}

double BPN::randomNum(double minv, double maxv) {
  return (double)minv + (maxv - minv)*rand()/(RAND_MAX+1.0f);
}
