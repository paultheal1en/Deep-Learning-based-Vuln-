static inline void downmix_2f_2r_to_dolby(float *samples)

{

    int i;



    for (i = 0; i < 256; i++) {

        samples[i] -= samples[i + 512];

        samples[i + 256] += samples[i + 768];

        samples[i + 512] = samples[i + 768] = 0;

    }

}
