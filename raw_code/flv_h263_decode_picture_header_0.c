int flv_h263_decode_picture_header(MpegEncContext *s)

{

    int format, width, height;



    /* picture header */

    if (get_bits_long(&s->gb, 17) != 1) {

        av_log(s->avctx, AV_LOG_ERROR, "Bad picture start code\n");

        return -1;

    }

    format = get_bits(&s->gb, 5);

    if (format != 0 && format != 1) {

        av_log(s->avctx, AV_LOG_ERROR, "Bad picture format\n");

        return -1;

    }

    s->h263_flv = format+1;

    s->picture_number = get_bits(&s->gb, 8); /* picture timestamp */

    format = get_bits(&s->gb, 3);

    switch (format) {

    case 0:

        width = get_bits(&s->gb, 8);

        height = get_bits(&s->gb, 8);

        break;

    case 1:

        width = get_bits(&s->gb, 16);

        height = get_bits(&s->gb, 16);

        break;

    case 2:

        width = 352;

        height = 288;

        break;

    case 3:

        width = 176;

        height = 144;

        break;

    case 4:

        width = 128;

        height = 96;

        break;

    case 5:

        width = 320;

        height = 240;

        break;

    case 6:

        width = 160;

        height = 120;

        break;

    default:

        width = height = 0;

        break;

    }

    if ((width == 0) || (height == 0))

        return -1;

    s->width = width;

    s->height = height;



    s->pict_type = I_TYPE + get_bits(&s->gb, 2);

    if (s->pict_type > P_TYPE)

        s->pict_type = P_TYPE;

    skip_bits1(&s->gb);	/* deblocking flag */

    s->qscale = get_bits(&s->gb, 5);



    s->h263_plus = 0;



    s->unrestricted_mv = 1;

    s->h263_long_vectors = 0;



    /* PEI */

    while (get_bits1(&s->gb) != 0) {

        skip_bits(&s->gb, 8);

    }

    s->f_code = 1;



    if(s->avctx->debug & FF_DEBUG_PICT_INFO){

        av_log(s->avctx, AV_LOG_DEBUG, "%c esc_type:%d, qp:%d num:%d\n",

               av_get_pict_type_char(s->pict_type), s->h263_flv-1, s->qscale, s->picture_number);

    }

    

    s->y_dc_scale_table=

    s->c_dc_scale_table= ff_mpeg1_dc_scale_table;



    return 0;

}
