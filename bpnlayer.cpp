#include "bpnlayer.h"

bpnLayer::bpnLayer(unsigned sizeNum, unsigned lowerSizeNum, bool with_bias, outFunction type, unsigned thrs) {
  threads = thrs;
  size = sizeNum;
  lowerSize = lowerSizeNum;
  func = type;
  bias = with_bias;

  if(size >0) {
    unsigned i, j;

    weights = new double*[size];
    deltas = new double*[size];

    if(lowerSize >0) {
      for(i=0; i<size; ++i) {
	weights[i] = new double[lowerSize];
	deltas[i] = new double[lowerSize];

	for(j=0; j<lowerSize; ++j) {
	  deltas[i][j] = 0;
	}
      }
    }
    else {   //  lowerSize ==0 -> input layer
      for(i=0; i<size; ++i) {
	weights[i] = new double[1];  //  just to keep structure integrity
	weights[i][0] = 1;
	deltas[i] = new double[1];
	deltas[i][0] = 0;
	lowerSize = 1;
      }
    }

    if(bias) biases = new double[size];

    errors = new double[size];
    // products = new double[size];

    products = new double*[threads];

    for(i=0; i < threads; ++i) {
      products[i] = new double[size];
    }
  }
}

bpnLayer::~bpnLayer() {
  if(size >0) {
    if(lowerSize >0) delete [] errors;

    unsigned i;

    for (i=0; i < threads; ++i) {
      delete products[i];
    }

    delete [] products;

    if(bias) delete [] biases;

    for(i=0; i<size; ++i) {
      delete deltas[i];
      delete weights[i];
    }

    delete [] deltas;
    delete [] weights;
  }
}

void bpnLayer::ChangeThreads (unsigned thr) {
  unsigned i;

  for (i=0; i < threads; ++i) {
    delete products[i];
  }

  delete [] products;

  threads = thr;

  products = new double*[threads];

  for(i=0; i < threads; ++i) {
    products[i] = new double[size];
  }
}
