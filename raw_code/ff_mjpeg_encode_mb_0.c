void ff_mjpeg_encode_mb(MpegEncContext *s, int16_t block[12][64])

{

    int i;

    if (s->chroma_format == CHROMA_444) {

        encode_block(s, block[0], 0);

        encode_block(s, block[2], 2);

        encode_block(s, block[4], 4);

        encode_block(s, block[8], 8);

        encode_block(s, block[5], 5);

        encode_block(s, block[9], 9);



        if (16*s->mb_x+8 < s->width) {

            encode_block(s, block[1], 1);

            encode_block(s, block[3], 3);

            encode_block(s, block[6], 6);

            encode_block(s, block[10], 10);

            encode_block(s, block[7], 7);

            encode_block(s, block[11], 11);

        }

    } else {

        for(i=0;i<5;i++) {

            encode_block(s, block[i], i);

        }

        if (s->chroma_format == CHROMA_420) {

            encode_block(s, block[5], 5);

        } else {

            encode_block(s, block[6], 6);

            encode_block(s, block[5], 5);

            encode_block(s, block[7], 7);

        }

    }

}
