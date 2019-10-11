#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
// GSL lib includes
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv2.h>


int ode_func (double x, const double y[], double f[], void *params)
{

    double mu = *(int *)params;
    f[0] = (x + 2 * y[0]) / (1 + mu * mu);
    return GSL_SUCCESS;
}

void calc_cauchy_problem(double x_start, double x_end, double y_start,
                         int count, int param1, int param2) {

#pragma omp parallel for
    for(int param = param1; param < param2; param++) {
        gsl_odeiv2_system sys = {ode_func, NULL, 1, &param};

        gsl_odeiv2_driver * d =
                gsl_odeiv2_driver_alloc_y_new (&sys, gsl_odeiv2_step_rk8pd,
                                               1e-6, 1e-6, 0.0);
        int i;
        double x = x_start, x1 = x_end;
        double y[1] = { y_start };

        for (i = 1; i <= count; i++)
        {
            double xi = i * x1 / count;
            int status = gsl_odeiv2_driver_apply (d, &x, xi, y);

            if (status != GSL_SUCCESS)
            {
                printf ("error, return value=%d\n", status);
                break;
            }

//            printf ("%d %d %.5e %.5e\n", omp_get_thread_num(), param, x, y[0]);
        }

        gsl_odeiv2_driver_free (d);
        }
    }

int main() {
    double start_time = omp_get_wtime();
    double x_start = 0;
    double x_end = 10;
    double y_start = 0;
    const int count = 100000;
    int param1 = 1;
    int param2 = 20;
    calc_cauchy_problem(x_start, x_end, y_start, count, param1, param2);
    printf("Elapsed time = %f\n", omp_get_wtime() - start_time);
    return 0;
}