static int decorrelate(TAKDecContext *s, int c1, int c2, int length)

{

    GetBitContext *gb = &s->gb;

    int32_t *p1       = s->decoded[c1] + (s->dmode > 5);

    int32_t *p2       = s->decoded[c2] + (s->dmode > 5);

    int32_t bp1       = p1[0];

    int32_t bp2       = p2[0];

    int i;

    int dshift, dfactor;



    length += s->dmode < 6;



    switch (s->dmode) {

    case 1: /* left/side */

        s->tdsp.decorrelate_ls(p1, p2, length);

        break;

    case 2: /* side/right */

        s->tdsp.decorrelate_sr(p1, p2, length);

        break;

    case 3: /* side/mid */

        s->tdsp.decorrelate_sm(p1, p2, length);

        break;

    case 4: /* side/left with scale factor */

        FFSWAP(int32_t*, p1, p2);

        FFSWAP(int32_t, bp1, bp2);

    case 5: /* side/right with scale factor */

        dshift  = get_bits_esc4(gb);

        dfactor = get_sbits(gb, 10);

        s->tdsp.decorrelate_sf(p1, p2, length, dshift, dfactor);

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



                if (filter_order == 16) {

                    v += s->adsp.scalarproduct_int16(&s->residues[i], s->filter,

                                                     filter_order);

                } else {

                    v += s->residues[i + 7] * s->filter[7] +

                         s->residues[i + 6] * s->filter[6] +

                         s->residues[i + 5] * s->filter[5] +

                         s->residues[i + 4] * s->filter[4] +

                         s->residues[i + 3] * s->filter[3] +

                         s->residues[i + 2] * s->filter[2] +

                         s->residues[i + 1] * s->filter[1] +

                         s->residues[i    ] * s->filter[0];

                }



                v = (av_clip_intp2(v >> 10, 13) << dshift) - *p1;

                *p1++ = v;

            }



            memmove(s->residues, &s->residues[tmp], 2 * filter_order);

        }



        emms_c();

        break;

    }

    }



    if (s->dmode > 0 && s->dmode < 6) {

        p1[0] = bp1;

        p2[0] = bp2;

    }



    return 0;

}
