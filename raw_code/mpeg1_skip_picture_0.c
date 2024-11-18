static void mpeg1_skip_picture(MpegEncContext *s, int pict_num)

{

    assert(s->codec_id == CODEC_ID_MPEG1VIDEO); // mpeg2 can do these repeat things



    /* mpeg1 picture header */

    put_header(s, PICTURE_START_CODE);

    /* temporal reference */

    put_bits(&s->pb, 10, pict_num & 0x3ff); 

    

    put_bits(&s->pb, 3, P_TYPE);

    put_bits(&s->pb, 16, 0xffff); /* non constant bit rate */

    

    put_bits(&s->pb, 1, 1); /* integer coordinates */

    put_bits(&s->pb, 3, 1); /* forward_f_code */

    

    put_bits(&s->pb, 1, 0); /* extra bit picture */

    

    /* only one slice */

    put_header(s, SLICE_MIN_START_CODE);

    put_bits(&s->pb, 5, 1); /* quantizer scale */

    put_bits(&s->pb, 1, 0); /* slice extra information */

    

    encode_mb_skip_run(s, 0);

    

    /* empty macroblock */

    put_bits(&s->pb, 3, 1); /* motion only */

    

    /* zero motion x & y */

    put_bits(&s->pb, 1, 1); 

    put_bits(&s->pb, 1, 1); 



    /* output a number of empty slice */

    encode_mb_skip_run(s, s->mb_width * s->mb_height - 2);

    

    /* empty macroblock */

    put_bits(&s->pb, 3, 1); /* motion only */

    

    /* zero motion x & y */

    put_bits(&s->pb, 1, 1); 

    put_bits(&s->pb, 1, 1); 

}
