void ff_acelp_weighted_filter(

        int16_t *out,

        const int16_t* in,

        const int16_t *weight_pow,

        int filter_length)

{

    int n;

    for(n=0; n<filter_length; n++)

        out[n] = (in[n] * weight_pow[n] + 0x4000) >> 15; /* (3.12) = (0.15) * (3.12) with rounding */

}
