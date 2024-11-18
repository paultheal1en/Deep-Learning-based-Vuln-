static int adaptive_cb_search(const int16_t *adapt_cb, float *work,

                              const float *coefs, float *data)

{

    int i, best_vect;

    float score, gain, best_score, best_gain;

    float exc[BLOCKSIZE];



    gain = best_score = 0;

    for (i = BLOCKSIZE / 2; i <= BUFFERSIZE; i++) {

        create_adapt_vect(exc, adapt_cb, i);

        get_match_score(work, coefs, exc, NULL, NULL, data, &score, &gain);

        if (score > best_score) {

            best_score = score;

            best_vect = i;

            best_gain = gain;

        }

    }

    if (!best_score)

        return 0;



    /**

     * Re-calculate the filtered vector from the vector with maximum match score

     * and remove its contribution from input data.

     */

    create_adapt_vect(exc, adapt_cb, best_vect);

    ff_celp_lp_synthesis_filterf(work, coefs, exc, BLOCKSIZE, LPC_ORDER);

    for (i = 0; i < BLOCKSIZE; i++)

        data[i] -= best_gain * work[i];

    return best_vect - BLOCKSIZE / 2 + 1;

}
