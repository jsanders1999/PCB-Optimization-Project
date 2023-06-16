#include <iostream>
#include <cmath>
#include <gsl/gsl_integration.h>

double f(double x, void* params) {
  double alpha = *(double*)params;
  double f = std::log(alpha * x) / std::sqrt(x);
  return f;
}

int main() {
  gsl_integration_workspace* w = gsl_integration_workspace_alloc(1000);

  double result, error;
  double expected = -4.0;
  double alpha = 1.0;

  gsl_function F;
  F.function = &f;
  F.params = &alpha;

  gsl_integration_qags(&F, 0, 1, 0, 1e-7, 1000, w, &result, &error);

  std::cout << "result          = " << result << std::endl;
  std::cout << "exact result    = " << expected << std::endl;
  std::cout << "estimated error = " << error << std::endl;
  std::cout << "actual error    = " << result - expected << std::endl;
  std::cout << "intervals       = " << w->size << std::endl;

  gsl_integration_workspace_free(w);

  return 0;
}
