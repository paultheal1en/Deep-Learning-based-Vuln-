static int decorrelate(TAKDecContext *s, int c1, int c2, int length)

{

    GetBitContext *gb = &s->gb;

    int32_t *p1       = s->decoded[c1] + 1;

    int32_t *p2       = s->decoded[c2] + 1;

    int i;

    int dshift, dfactor;



    switch (s->dmode) {

    case 1: /* left/side */

        for (i = 0; i < length; i++) {

            int32_t a = p1[i];

            int32_t b = p2[i];

            p2[i]     = a + b;

        }

        break;

    case 2: /* side/right */

        for (i = 0; i < length; i++) {

            int32_t a = p1[i];

            int32_t b = p2[i];

            p1[i]     = b - a;

        }

        break;

    case 3: /* side/mid */

        for (i = 0; i < length; i++) {

            int32_t a = p1[i];

            int32_t b = p2[i];

            a        -= b >> 1;

            p1[i]     = a;

            p2[i]     = a + b;

        }

        break;

    case 4: /* side/left with scale factor */

        FFSWAP(int32_t*, p1, p2);

    case 5: /* side/right with scale factor */

        dshift  = get_bits_esc4(gb);

        dfactor = get_sbits(gb, 10);

        for (i = 0; i < length; i++) {

            int32_t a = p1[i];

            int32_t b = p2[i];

            b         = dfactor * (b >> dshift) + 128 >> 8 << dshift;

            p1[i]     = b - a;

        }

        break;

    case 6:

        FFSWAP(int32_t*, p1, p2);

    case 7: {

        int length2, order_half, filter_order, dval1, dval2;

        int tmp, x, code_size;



        if (length < 256)

            return AVERROR_INVALIDDATA;



        dshift       = get_bits_esc4(gb);

        filter_order = 8 << get_bits1(gb);

        dval1        = get_bits1(gb);

        dval2        = get_bits1(gb);



        AV_ZERO128(s->filter + 8);

        for (i = 0; i < filter_order; i++) {

            if (!(i & 3))

                code_size = 14 - get_bits(gb, 3);

            s->filter[i] = get_sbits(gb, code_size);

        }



        order_half = filter_order / 2;

        length2    = length - (filter_order - 1);



        /* decorrelate beginning samples */

        if (dval1) {

            for (i = 0; i < order_half; i++) {

                int32_t a = p1[i];

                int32_t b = p2[i];

                p1[i]     = a + b;

            }

        }



        /* decorrelate ending samples */

        if (dval2) {

            for (i = length2 + order_half; i < length; i++) {

                int32_t a = p1[i];

                int32_t b = p2[i];

                p1[i]     = a + b;

            }

        }





        for (i = 0; i < filter_order; i++)

            s->residues[i] = *p2++ >> dshift;



        p1 += order_half;

        x = FF_ARRAY_ELEMS(s->residues) - filter_order;

        for (; length2 > 0; length2 -= tmp) {

            tmp = FFMIN(length2, x);



            for (i = 0; i < tmp; i++)

                s->residues[filter_order + i] = *p2++ >> dshift;



            for (i = 0; i < tmp; i++) {

                int v = 1 << 9;



                v += s->adsp.scalarproduct_int16(&s->residues[i], s->filter, 16);

                v = (av_clip_intp2(v >> 10, 13) << dshift) - *p1;

                *p1++ = v;

            }



            memcpy(s->residues, &s->residues[tmp], 2 * filter_order);

        }



        emms_c();

        break;

    }

    }



    return 0;

}
