static void noise_scale(int *coefs, int scale, int band_energy, int len)

{

    int ssign = scale < 0 ? -1 : 1;

    int s = FFABS(scale);

    unsigned int round;

    int i, out, c = exp2tab[s & 3];

    int nlz = 0;



    while (band_energy > 0x7fff) {

        band_energy >>= 1;

        nlz++;

    }

    c /= band_energy;

    s = 21 + nlz - (s >> 2);



    if (s > 0) {

        round = 1 << (s-1);

        for (i=0; i<len; i++) {

            out = (int)(((int64_t)coefs[i] * c) >> 32);

            coefs[i] = ((int)(out+round) >> s) * ssign;

        }

    }

    else {

        s = s + 32;

        round = 1 << (s-1);

        for (i=0; i<len; i++) {

            out = (int)((int64_t)((int64_t)coefs[i] * c + round) >> s);

            coefs[i] = out * ssign;

        }

    }

}
