int ff_celp_lp_synthesis_filter(int16_t *out,

                                const int16_t* filter_coeffs,

                                const int16_t* in,

                                int buffer_length,

                                int filter_length,

                                int stop_on_overflow,

                                int rounder)

{

    int i,n;



    // Avoids a +1 in the inner loop.

    filter_length++;



    for (n = 0; n < buffer_length; n++) {

        int sum = rounder;

        for (i = 1; i < filter_length; i++)

            sum -= filter_coeffs[i-1] * out[n-i];



        sum = (sum >> 12) + in[n];



        if (sum + 0x8000 > 0xFFFFU) {

            if (stop_on_overflow)

                return 1;

            sum = (sum >> 31) ^ 32767;

        }

        out[n] = sum;

    }



    return 0;

}
