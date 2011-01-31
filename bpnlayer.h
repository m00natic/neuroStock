#ifndef BPNLAYER_H
#define BPNLAYER_H

enum outFunction {
  linear=0,
  sigmoid=1,
  sigmoid2=2
};


class bpnLayer {
 public:
  bpnLayer(unsigned, unsigned, bool, outFunction, unsigned);
  ~bpnLayer();
  void ChangeThreads(unsigned);

  unsigned threads;
  unsigned size;
  unsigned lowerSize;
  bool bias;
  double **weights;
  double **deltas;
  double *biases;
  double **products;
  double *errors;
  outFunction func;
};

#endif // BPNLAYER_H
