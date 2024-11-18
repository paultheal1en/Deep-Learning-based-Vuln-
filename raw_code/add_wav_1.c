static void add_wav(int16_t *dest, int n, int skip_first, int *m,

                    const int16_t *s1, const int8_t *s2, const int8_t *s3)

{

    int i;

    int v[3];



    v[0] = 0;

    for (i=!skip_first; i<3; i++)

        v[i] = (gain_val_tab[n][i] * m[i]) >> gain_exp_tab[n];





        dest[i] = (s1[i]*v[0] + s2[i]*v[1] + s3[i]*v[2]) >> 12;




