static void generate_noise(G723_1_Context *p)

{

    int i, j, idx, t;

    int off[SUBFRAMES];

    int signs[SUBFRAMES / 2 * 11], pos[SUBFRAMES / 2 * 11];

    int tmp[SUBFRAME_LEN * 2];

    int16_t *vector_ptr;

    int64_t sum;

    int b0, c, delta, x, shift;



    p->pitch_lag[0] = cng_rand(&p->cng_random_seed, 21) + 123;

    p->pitch_lag[1] = cng_rand(&p->cng_random_seed, 19) + 123;



    for (i = 0; i < SUBFRAMES; i++) {

        p->subframe[i].ad_cb_gain = cng_rand(&p->cng_random_seed, 50) + 1;

        p->subframe[i].ad_cb_lag  = cng_adaptive_cb_lag[i];

    }



    for (i = 0; i < SUBFRAMES / 2; i++) {

        t = cng_rand(&p->cng_random_seed, 1 << 13);

        off[i * 2]     =   t       & 1;

        off[i * 2 + 1] = ((t >> 1) & 1) + SUBFRAME_LEN;

        t >>= 2;

        for (j = 0; j < 11; j++) {

            signs[i * 11 + j] = (t & 1) * 2 - 1 << 14;

            t >>= 1;

        }

    }



    idx = 0;

    for (i = 0; i < SUBFRAMES; i++) {

        for (j = 0; j < SUBFRAME_LEN / 2; j++)

            tmp[j] = j;

        t = SUBFRAME_LEN / 2;

        for (j = 0; j < pulses[i]; j++, idx++) {

            int idx2 = cng_rand(&p->cng_random_seed, t);



            pos[idx]  = tmp[idx2] * 2 + off[i];

            tmp[idx2] = tmp[--t];

        }

    }



    vector_ptr = p->audio + LPC_ORDER;

    memcpy(vector_ptr, p->prev_excitation,

           PITCH_MAX * sizeof(*p->excitation));

    for (i = 0; i < SUBFRAMES; i += 2) {

        ff_g723_1_gen_acb_excitation(vector_ptr, vector_ptr,

                                     p->pitch_lag[i >> 1], &p->subframe[i],

                                     p->cur_rate);

        ff_g723_1_gen_acb_excitation(vector_ptr + SUBFRAME_LEN,

                                     vector_ptr + SUBFRAME_LEN,

                                     p->pitch_lag[i >> 1], &p->subframe[i + 1],

                                     p->cur_rate);



        t = 0;

        for (j = 0; j < SUBFRAME_LEN * 2; j++)

            t |= FFABS(vector_ptr[j]);

        t = FFMIN(t, 0x7FFF);

        if (!t) {

            shift = 0;

        } else {

            shift = -10 + av_log2(t);

            if (shift < -2)

                shift = -2;

        }

        sum = 0;

        if (shift < 0) {

           for (j = 0; j < SUBFRAME_LEN * 2; j++) {

               t      = vector_ptr[j] << -shift;

               sum   += t * t;

               tmp[j] = t;

           }

        } else {

           for (j = 0; j < SUBFRAME_LEN * 2; j++) {

               t      = vector_ptr[j] >> shift;

               sum   += t * t;

               tmp[j] = t;

           }

        }



        b0 = 0;

        for (j = 0; j < 11; j++)

            b0 += tmp[pos[(i / 2) * 11 + j]] * signs[(i / 2) * 11 + j];

        b0 = b0 * 2 * 2979LL + (1 << 29) >> 30; // approximated division by 11



        c = p->cur_gain * (p->cur_gain * SUBFRAME_LEN >> 5);

        if (shift * 2 + 3 >= 0)

            c >>= shift * 2 + 3;

        else

            c <<= -(shift * 2 + 3);

        c = (av_clipl_int32(sum << 1) - c) * 2979LL >> 15;



        delta = b0 * b0 * 2 - c;

        if (delta <= 0) {

            x = -b0;

        } else {

            delta = square_root(delta);

            x     = delta - b0;

            t     = delta + b0;

            if (FFABS(t) < FFABS(x))

                x = -t;

        }

        shift++;

        if (shift < 0)

           x >>= -shift;

        else

           x <<= shift;

        x = av_clip(x, -10000, 10000);



        for (j = 0; j < 11; j++) {

            idx = (i / 2) * 11 + j;

            vector_ptr[pos[idx]] = av_clip_int16(vector_ptr[pos[idx]] +

                                                 (x * signs[idx] >> 15));

        }



        /* copy decoded data to serve as a history for the next decoded subframes */

        memcpy(vector_ptr + PITCH_MAX, vector_ptr,

               sizeof(*vector_ptr) * SUBFRAME_LEN * 2);

        vector_ptr += SUBFRAME_LEN * 2;

    }

    /* Save the excitation for the next frame */

    memcpy(p->prev_excitation, p->audio + LPC_ORDER + FRAME_LEN,

           PITCH_MAX * sizeof(*p->excitation));

}
