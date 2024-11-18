static int eval_refl(int *refl, const int16_t *coefs, RA144Context *ractx)

{

    int b, c, i;

    unsigned int u;

    int buffer1[10];

    int buffer2[10];

    int *bp1 = buffer1;

    int *bp2 = buffer2;



    for (i=0; i < 10; i++)

        buffer2[i] = coefs[i];



    refl[9] = bp2[9];



    if ((unsigned) bp2[9] + 0x1000 > 0x1fff) {

        av_log(ractx, AV_LOG_ERROR, "Overflow. Broken sample?\n");

        return 1;

    }



    for (c=8; c >= 0; c--) {

        b = 0x1000-((bp2[c+1] * bp2[c+1]) >> 12);



        if (!b)

            b = -2;



        for (u=0; u<=c; u++)

            bp1[u] = ((bp2[u] - ((refl[c+1] * bp2[c-u]) >> 12)) * (0x1000000 / b)) >> 12;



        refl[c] = bp1[c];



        if ((unsigned) bp1[c] + 0x1000 > 0x1fff)

            return 1;



        FFSWAP(int *, bp1, bp2);

    }

    return 0;

}
