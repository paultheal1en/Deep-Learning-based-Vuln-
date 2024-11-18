int intel_h263_decode_picture_header(MpegEncContext *s)

{

    int format;



    /* picture header */

    if (get_bits_long(&s->gb, 22) != 0x20) {

        av_log(s->avctx, AV_LOG_ERROR, "Bad picture start code\n");

        return -1;

    }

    s->picture_number = get_bits(&s->gb, 8); /* picture timestamp */



    if (get_bits1(&s->gb) != 1) {

        av_log(s->avctx, AV_LOG_ERROR, "Bad marker\n");

        return -1;	/* marker */

    }

    if (get_bits1(&s->gb) != 0) {

        av_log(s->avctx, AV_LOG_ERROR, "Bad H263 id\n");

        return -1;	/* h263 id */

    }

    skip_bits1(&s->gb);	/* split screen off */

    skip_bits1(&s->gb);	/* camera  off */

    skip_bits1(&s->gb);	/* freeze picture release off */



    format = get_bits(&s->gb, 3);

    if (format != 7) {

        av_log(s->avctx, AV_LOG_ERROR, "Intel H263 free format not supported\n");

        return -1;

    }

    s->h263_plus = 0;



    s->pict_type = I_TYPE + get_bits1(&s->gb);

    

    s->unrestricted_mv = get_bits1(&s->gb); 

    s->h263_long_vectors = s->unrestricted_mv;



    if (get_bits1(&s->gb) != 0) {

        av_log(s->avctx, AV_LOG_ERROR, "SAC not supported\n");

        return -1;	/* SAC: off */

    }

    if (get_bits1(&s->gb) != 0) {

        s->obmc= 1;

        av_log(s->avctx, AV_LOG_ERROR, "Advanced Prediction Mode not supported\n");

//        return -1;	/* advanced prediction mode: off */

    }

    if (get_bits1(&s->gb) != 0) {

        av_log(s->avctx, AV_LOG_ERROR, "PB frame mode no supported\n");

        return -1;	/* PB frame mode */

    }



    /* skip unknown header garbage */

    skip_bits(&s->gb, 41);



    s->qscale = get_bits(&s->gb, 5);

    skip_bits1(&s->gb);	/* Continuous Presence Multipoint mode: off */



    /* PEI */

    while (get_bits1(&s->gb) != 0) {

        skip_bits(&s->gb, 8);

    }

    s->f_code = 1;



    s->y_dc_scale_table=

    s->c_dc_scale_table= ff_mpeg1_dc_scale_table;



    return 0;

}
