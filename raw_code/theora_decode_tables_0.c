static int theora_decode_tables(AVCodecContext *avctx, GetBitContext gb)

{

    Vp3DecodeContext *s = avctx->priv_data;

    int i, n, matrices;



    if (s->theora >= 0x030200) {

        n = get_bits(&gb, 3);

        /* loop filter limit values table */

        for (i = 0; i < 64; i++)

            s->filter_limit_values[i] = get_bits(&gb, n);

    }



    if (s->theora >= 0x030200)

        n = get_bits(&gb, 4) + 1;

    else

        n = 16;

    /* quality threshold table */

    for (i = 0; i < 64; i++)

        s->coded_ac_scale_factor[i] = get_bits(&gb, n);



    if (s->theora >= 0x030200)

        n = get_bits(&gb, 4) + 1;

    else

        n = 16;

    /* dc scale factor table */

    for (i = 0; i < 64; i++)

        s->coded_dc_scale_factor[i] = get_bits(&gb, n);



    if (s->theora >= 0x030200)

        matrices = get_bits(&gb, 9) + 1;

    else

        matrices = 3;

    if (matrices != 3) {

        av_log(avctx,AV_LOG_ERROR, "unsupported matrices: %d\n", matrices);

//        return -1;

    }

    /* y coeffs */

    for (i = 0; i < 64; i++)

        s->coded_intra_y_dequant[i] = get_bits(&gb, 8);



    /* uv coeffs */

    for (i = 0; i < 64; i++)

        s->coded_intra_c_dequant[i] = get_bits(&gb, 8);



    /* inter coeffs */

    for (i = 0; i < 64; i++)

        s->coded_inter_dequant[i] = get_bits(&gb, 8);



    /* skip unknown matrices */

    n = matrices - 3;

    while(n--)

        for (i = 0; i < 64; i++)

            skip_bits(&gb, 8);



    for (i = 0; i <= 1; i++) {

        for (n = 0; n <= 2; n++) {

            int newqr;

            if (i > 0 || n > 0)

                newqr = get_bits(&gb, 1);

            else

                newqr = 1;

            if (!newqr) {

                if (i > 0)

                    get_bits(&gb, 1);

            }

            else {

                int qi = 0;

                skip_bits(&gb, av_log2(matrices-1)+1);

                while (qi < 63) {

                    qi += get_bits(&gb, av_log2(63-qi)+1) + 1;

                    skip_bits(&gb, av_log2(matrices-1)+1);

                }

                if (qi > 63) {

                    av_log(avctx, AV_LOG_ERROR, "invalid qi %d > 63\n", qi);

                    return -1;

                }

            }

        }

    }



    /* Huffman tables */

    for (s->hti = 0; s->hti < 80; s->hti++) {

        s->entries = 0;

        s->huff_code_size = 1;

        if (!get_bits(&gb, 1)) {

            s->hbits = 0;

            read_huffman_tree(avctx, &gb);

            s->hbits = 1;

            read_huffman_tree(avctx, &gb);

        }

    }



    s->theora_tables = 1;



    return 0;

}
