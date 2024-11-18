static void tgq_decode_mb(TgqContext *s, int mb_y, int mb_x, const uint8_t **bs, const uint8_t *buf_end){

    int mode;

    int i;

    int8_t dc[6];



    mode = bytestream_get_byte(bs);

    if (mode>buf_end-*bs) {

        av_log(s->avctx, AV_LOG_ERROR, "truncated macroblock\n");

        return;

    }



    if (mode>12) {

        GetBitContext gb;

        init_get_bits(&gb, *bs, mode*8);

        for(i=0; i<6; i++)

            tgq_decode_block(s, s->block[i], &gb);

        tgq_idct_put_mb(s, s->block, mb_x, mb_y);

    }else{

        if (mode==3) {

            memset(dc, (*bs)[0], 4);

            dc[4] = (*bs)[1];

            dc[5] = (*bs)[2];

        }else if (mode==6) {

            memcpy(dc, *bs, 6);

        }else if (mode==12) {

            for(i=0; i<6; i++)

                dc[i] = (*bs)[i*2];

        }else{

            av_log(s->avctx, AV_LOG_ERROR, "unsupported mb mode %i\n", mode);

        }

        tgq_idct_put_mb_dconly(s, mb_x, mb_y, dc);

    }

    *bs += mode;

}
