#ifndef BPN_H
#define BPN_H

#include <fstream>
#include "bpnlayer.h"

class BPN;

struct LayerThread {
  unsigned layer, start, end;
  BPN *bp;
  double (*apply) (double);
};

class BPN {
 public:
  BPN(const char*, unsigned);
  BPN();
  ~BPN();

  void Run(const char*, unsigned);
  bool Train(const char*);
  bool SaveToFile(const char*);
  void ChangeThreads(unsigned);

 private:
  void constructor(unsigned*, bool*, outFunction*, unsigned, double, double, double, unsigned);
  void ConstructDefault(unsigned);
  void InitializeWeights();

  void Run(const char*, bool);

  bool SaveLayer(const bpnLayer*, std::ofstream&);
  std::string readString(std::ifstream&);
  void LoadLayer(bpnLayer*, std::ifstream&);
  void PrepareFromFEN(const char*, bool);
  void PrepareFromFEN(const char*, unsigned, bool);
  void Run();
  void Run(unsigned);
  bool Train();
  bool DoThreading(unsigned, double (*) (double), void* (*)(void*));

  static double randomNum(double, double);
  static double ApplyLinear(double);
  static double ApplySigmoid(double);
  static double ApplySigmoid2(double);
  static double DerivateLinear(double);
  static double DerivateSigmoid(double);
  static double DerivateSigmoid2(double);

  static void* UnitThreadFuncBias (void*);
  static void* UnitThreadFunc (void*);
  static void* UnitThreadFuncBiasScale (void*);
  static void* UnitThreadFuncScale (void*);

  static void* UnitThreadFuncTrain (void*);
  static void* UnitThreadFuncRenew (void*);

 private:
  double initial_scale;
  LayerThread **li; // thread info
  pthread_t *thread_ids;
  unsigned threads; // number of additional threads, should be # cores -1

 public:
  unsigned minchunk; // minimum number of layer units processed by thread
  unsigned size;
  bpnLayer **layers;
  double* train_output;
  double scale_factor;
  double alpha;    //  momentum
  double eta;  //  learning rate
};

#endif // BPN_H
