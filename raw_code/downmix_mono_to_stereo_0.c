static inline void downmix_mono_to_stereo(float *samples)

{

    int i;



    for (i = 0; i < 256; i++)

        samples[i + 256] = samples[i];

}
