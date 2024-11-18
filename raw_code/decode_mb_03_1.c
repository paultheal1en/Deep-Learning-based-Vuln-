static int decode_mb(MadContext *s, AVFrame *frame, int inter)

{

    int mv_map = 0;

    int mv_x, mv_y;

    int j;



    if (inter) {

        int v = decode210(&s->gb);

        if (v < 2) {

            mv_map = v ? get_bits(&s->gb, 6) : 63;

            mv_x = decode_motion(&s->gb);

            mv_y = decode_motion(&s->gb);

        }

    }



    for (j=0; j<6; j++) {

        if (mv_map & (1<<j)) {  // mv_x and mv_y are guarded by mv_map

            int add = 2*decode_motion(&s->gb);

            if (s->last_frame->data[0])

                comp_block(s, frame, s->mb_x, s->mb_y, j, mv_x, mv_y, add);

        } else {

            s->dsp.clear_block(s->block);

            if(decode_block_intra(s, s->block) < 0)

                return -1;

            idct_put(s, frame, s->block, s->mb_x, s->mb_y, j);

        }

    }

    return 0;

}
