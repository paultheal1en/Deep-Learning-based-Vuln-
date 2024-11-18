static int _decode_exponents(int expstr, int ngrps, uint8_t absexp, uint8_t *gexps, uint8_t *dexps)

{

    int exps;

    int i = 0;



    while (ngrps--) {

        exps = gexps[i++];



        absexp += exp_1[exps];

        assert(absexp <= 24);

        switch (expstr) {

            case AC3_EXPSTR_D45:

                *(dexps++) = absexp;

                *(dexps++) = absexp;

            case AC3_EXPSTR_D25:

                *(dexps++) = absexp;

            case AC3_EXPSTR_D15:

                *(dexps++) = absexp;

        }

        absexp += exp_2[exps];

        assert(absexp <= 24);

        switch (expstr) {

            case AC3_EXPSTR_D45:

                *(dexps++) = absexp;

                *(dexps++) = absexp;

            case AC3_EXPSTR_D25:

                *(dexps++) = absexp;

            case AC3_EXPSTR_D15:

                *(dexps++) = absexp;

        }



        absexp += exp_3[exps];

        assert(absexp <= 24);

        switch (expstr) {

            case AC3_EXPSTR_D45:

                *(dexps++) = absexp;

                *(dexps++) = absexp;

            case AC3_EXPSTR_D25:

                *(dexps++) = absexp;

            case AC3_EXPSTR_D15:

                *(dexps++) = absexp;

        }

    }



    return 0;

}
