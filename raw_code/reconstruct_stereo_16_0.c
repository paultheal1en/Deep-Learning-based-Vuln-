static void reconstruct_stereo_16(int32_t *buffer[MAX_CHANNELS],

                                  int16_t *buffer_out,

                                  int numchannels, int numsamples,

                                  uint8_t interlacing_shift,

                                  uint8_t interlacing_leftweight)

{

    int i;

    if (numsamples <= 0)

        return;



    /* weighted interlacing */

    if (interlacing_leftweight) {

        for (i = 0; i < numsamples; i++) {

            int32_t a, b;



            a = buffer[0][i];

            b = buffer[1][i];



            a -= (b * interlacing_leftweight) >> interlacing_shift;

            b += a;



            buffer_out[i*numchannels] = b;

            buffer_out[i*numchannels + 1] = a;

        }



        return;

    }



    /* otherwise basic interlacing took place */

    for (i = 0; i < numsamples; i++) {

        int16_t left, right;



        left = buffer[0][i];

        right = buffer[1][i];



        buffer_out[i*numchannels] = left;

        buffer_out[i*numchannels + 1] = right;

    }

}
