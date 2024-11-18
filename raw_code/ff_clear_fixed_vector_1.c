void ff_clear_fixed_vector(float *out, const AMRFixed *in, int size)

{

    int i;



    for (i=0; i < in->n; i++) {

        int x  = in->x[i], repeats = !((in->no_repeat_mask >> i) & 1);




        do {

            out[x] = 0.0;

            x += in->pitch_lag;

        } while (x < size && repeats);

    }

}