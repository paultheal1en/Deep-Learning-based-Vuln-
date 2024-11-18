static int theora_decode_tables(AVCodecContext *avctx, GetBitContext *gb)

{

    Vp3DecodeContext *s = avctx->priv_data;

    int i, n, matrices, inter, plane;



    if (s->theora >= 0x030200) {

        n = get_bits(gb, 3);

        /* loop filter limit values table */

        if (n) {

        for (i = 0; i < 64; i++) {

            s->filter_limit_values[i] = get_bits(gb, n);

            if (s->filter_limit_values[i] > 127) {

                av_log(avctx, AV_LOG_ERROR, "filter limit value too large (%i > 127), clamping\n", s->filter_limit_values[i]);

                s->filter_limit_values[i] = 127;







    if (s->theora >= 0x030200)

        n = get_bits(gb, 4) + 1;

    else

        n = 16;

    /* quality threshold table */

    for (i = 0; i < 64; i++)

        s->coded_ac_scale_factor[i] = get_bits(gb, n);



    if (s->theora >= 0x030200)

        n = get_bits(gb, 4) + 1;

    else

        n = 16;

    /* dc scale factor table */

    for (i = 0; i < 64; i++)

        s->coded_dc_scale_factor[i] = get_bits(gb, n);



    if (s->theora >= 0x030200)

        matrices = get_bits(gb, 9) + 1;

    else

        matrices = 3;



    if(matrices > 384){

        av_log(avctx, AV_LOG_ERROR, "invalid number of base matrixes\n");

        return -1;




    for(n=0; n<matrices; n++){

        for (i = 0; i < 64; i++)

            s->base_matrix[n][i]= get_bits(gb, 8);




    for (inter = 0; inter <= 1; inter++) {

        for (plane = 0; plane <= 2; plane++) {

            int newqr= 1;

            if (inter || plane > 0)

                newqr = get_bits1(gb);

            if (!newqr) {

                int qtj, plj;

                if(inter && get_bits1(gb)){

                    qtj = 0;

                    plj = plane;

                }else{

                    qtj= (3*inter + plane - 1) / 3;

                    plj= (plane + 2) % 3;


                s->qr_count[inter][plane]= s->qr_count[qtj][plj];

                memcpy(s->qr_size[inter][plane], s->qr_size[qtj][plj], sizeof(s->qr_size[0][0]));

                memcpy(s->qr_base[inter][plane], s->qr_base[qtj][plj], sizeof(s->qr_base[0][0]));

            } else {

                int qri= 0;

                int qi = 0;



                for(;;){

                    i= get_bits(gb, av_log2(matrices-1)+1);

                    if(i>= matrices){

                        av_log(avctx, AV_LOG_ERROR, "invalid base matrix index\n");

                        return -1;


                    s->qr_base[inter][plane][qri]= i;

                    if(qi >= 63)

                        break;

                    i = get_bits(gb, av_log2(63-qi)+1) + 1;

                    s->qr_size[inter][plane][qri++]= i;

                    qi += i;




                if (qi > 63) {

                    av_log(avctx, AV_LOG_ERROR, "invalid qi %d > 63\n", qi);

                    return -1;


                s->qr_count[inter][plane]= qri;






    /* Huffman tables */

    for (s->hti = 0; s->hti < 80; s->hti++) {

        s->entries = 0;

        s->huff_code_size = 1;

        if (!get_bits1(gb)) {

            s->hbits = 0;

            if(read_huffman_tree(avctx, gb))

                return -1;

            s->hbits = 1;

            if(read_huffman_tree(avctx, gb))

                return -1;





    s->theora_tables = 1;



    return 0;
