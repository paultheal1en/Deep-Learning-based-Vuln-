static inline void downmix_dualmono_to_mono(float *samples)

{

    int i;



    for (i = 0; i < 256; i++) {

        samples[i] += samples[i + 256];

        samples[i + 256] = 0;

    }

}
