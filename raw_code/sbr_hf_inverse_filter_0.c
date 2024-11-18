static void sbr_hf_inverse_filter(float (*alpha0)[2], float (*alpha1)[2],

                                  const float X_low[32][40][2], int k0)

{

    int k;

    for (k = 0; k < k0; k++) {

        float phi[3][2][2], dk;



        autocorrelate(X_low[k], phi, 0);

        autocorrelate(X_low[k], phi, 1);

        autocorrelate(X_low[k], phi, 2);



        dk =  phi[2][1][0] * phi[1][0][0] -

             (phi[1][1][0] * phi[1][1][0] + phi[1][1][1] * phi[1][1][1]) / 1.000001f;



        if (!dk) {

            alpha1[k][0] = 0;

            alpha1[k][1] = 0;

        } else {

            float temp_real, temp_im;

            temp_real = phi[0][0][0] * phi[1][1][0] -

                        phi[0][0][1] * phi[1][1][1] -

                        phi[0][1][0] * phi[1][0][0];

            temp_im   = phi[0][0][0] * phi[1][1][1] +

                        phi[0][0][1] * phi[1][1][0] -

                        phi[0][1][1] * phi[1][0][0];



            alpha1[k][0] = temp_real / dk;

            alpha1[k][1] = temp_im   / dk;

        }



        if (!phi[1][0][0]) {

            alpha0[k][0] = 0;

            alpha0[k][1] = 0;

        } else {

            float temp_real, temp_im;

            temp_real = phi[0][0][0] + alpha1[k][0] * phi[1][1][0] +

                                       alpha1[k][1] * phi[1][1][1];

            temp_im   = phi[0][0][1] + alpha1[k][1] * phi[1][1][0] -

                                       alpha1[k][0] * phi[1][1][1];



            alpha0[k][0] = -temp_real / phi[1][0][0];

            alpha0[k][1] = -temp_im   / phi[1][0][0];

        }



        if (alpha1[k][0] * alpha1[k][0] + alpha1[k][1] * alpha1[k][1] >= 16.0f ||

           alpha0[k][0] * alpha0[k][0] + alpha0[k][1] * alpha0[k][1] >= 16.0f) {

            alpha1[k][0] = 0;

            alpha1[k][1] = 0;

            alpha0[k][0] = 0;

            alpha0[k][1] = 0;

        }

    }

}
