float ff_amr_set_fixed_gain(float fixed_gain_factor, float fixed_mean_energy,

                            float *prediction_error, float energy_mean,

                            const float *pred_table)

{

    // Equations 66-69:

    // ^g_c = ^gamma_gc * 100.05 (predicted dB + mean dB - dB of fixed vector)

    // Note 10^(0.05 * -10log(average x2)) = 1/sqrt((average x2)).

    float val = fixed_gain_factor *

        ff_exp10(0.05 *

              (avpriv_scalarproduct_float_c(pred_table, prediction_error, 4) +

               energy_mean)) /

        sqrtf(fixed_mean_energy);



    // update quantified prediction error energy history

    memmove(&prediction_error[0], &prediction_error[1],

            3 * sizeof(prediction_error[0]));

    prediction_error[3] = 20.0 * log10f(fixed_gain_factor);



    return val;

}
