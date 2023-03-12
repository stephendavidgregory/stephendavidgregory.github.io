

// capture-recapture model
#include <TMB.hpp>

/* implement the vector - matrix product */
template<class Type>
vector<Type> multvecmat(vector<Type> A, matrix<Type> B) {
  int nrowb = B.rows();
  int ncolb = B.cols(); 
  vector<Type> C(ncolb);
  for (int i = 0; i < ncolb; i++) {
    C(i) = Type(0);
    for (int k = 0; k < nrowb; k++) {
      C(i) += A(k) * B(k, i);
    }
  }
  return C;
}

template<class Type>
Type objective_function<Type>::operator() () {

  // data
  DATA_IMATRIX(ch);
  int n_individuals = ch.rows();  
  int n_occasions = ch.cols();
  // DATA_VECTOR(fii);
  
  // parameters
  PARAMETER_VECTOR(b);
  
  // transformations
  int npar = b.size();
  vector<Type> par(npar);
  for (int i = 0; i < npar; i++) {
    par(i) = Type(1.0) / (Type(1.0) + exp(-b(i)));
  }
  Type phi = par(0);
  Type psi = par(1);
  Type p = par(2);
  
  // observation matrix
  matrix<Type> A(3, 3);
  A(0, 0) = Type(0.0);
  A(0, 1) = phi;
  A(0, 2) = Type(1.0) - phi;
  A(1, 0) = Type(0.0);
  A(1, 1) = psi;
  A(1, 2) = Type(1.0) - psi;
  A(2, 0) = Type(0.0);
  A(2, 1) = Type(0.0);
  A(2, 2) = Type(1.0);
  
  // observation matrix
  matrix<Type> B(3, 3);
  B(0, 0) = Type(0.0);
  B(0, 1) = Type(0.0);
  B(0, 2) = Type(1.0);
  B(1, 0) = Type(0.0);
  B(1, 1) = p;
  B(1, 2) = Type(1.0) - p;
  B(2, 0) = Type(0.0);
  B(2, 1) = Type(0.0);
  B(2, 2) = Type(1.0);
  
  // likelihood
  Type ll;
  Type nll;
  for (int i = 0; i < n_individuals; i++) {
    vector<Type> foo(3);
    foo(0) = Type(1.0);
    foo(1) = Type(0.0);
    foo(2) = Type(0.0);
    for (int j = 1; j < n_occasions; j++) {
      foo = multvecmat(foo, A) * vector<Type> (B.col(ch(i, j)));
    }
    ll += log(sum(foo));
  }
  
  // negative loglikelihood
  nll = -ll;
  
  // return
  return nll;
  
}
