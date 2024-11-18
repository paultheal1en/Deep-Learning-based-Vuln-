static int svq1_decode_frame_header(AVCodecContext *avctx, AVFrame *frame)

{

    SVQ1Context *s = avctx->priv_data;

    GetBitContext *bitbuf = &s->gb;

    int frame_size_code;

    int width  = s->width;

    int height = s->height;



    skip_bits(bitbuf, 8); /* temporal_reference */



    /* frame type */

    s->nonref = 0;

    switch (get_bits(bitbuf, 2)) {

    case 0:

        frame->pict_type = AV_PICTURE_TYPE_I;

        break;

    case 2:

        s->nonref = 1;

    case 1:

        frame->pict_type = AV_PICTURE_TYPE_P;

        break;

    default:

        av_log(avctx, AV_LOG_ERROR, "Invalid frame type.\n");

        return AVERROR_INVALIDDATA;

    }



    if (frame->pict_type == AV_PICTURE_TYPE_I) {

        /* unknown fields */

        if (s->frame_code == 0x50 || s->frame_code == 0x60) {

            int csum = get_bits(bitbuf, 16);



            csum = ff_svq1_packet_checksum(bitbuf->buffer,

                                           bitbuf->size_in_bits >> 3,

                                           csum);



            av_dlog(avctx, "%s checksum (%02x) for packet data\n",

                    (csum == 0) ? "correct" : "incorrect", csum);

        }



        if ((s->frame_code ^ 0x10) >= 0x50) {

            uint8_t msg[256];



            svq1_parse_string(bitbuf, msg);



            av_log(avctx, AV_LOG_INFO,

                   "embedded message:\n%s\n", (char *)msg);

        }



        skip_bits(bitbuf, 2);

        skip_bits(bitbuf, 2);

        skip_bits1(bitbuf);



        /* load frame size */

        frame_size_code = get_bits(bitbuf, 3);



        if (frame_size_code == 7) {

            /* load width, height (12 bits each) */

            width  = get_bits(bitbuf, 12);

            height = get_bits(bitbuf, 12);



            if (!width || !height)

                return AVERROR_INVALIDDATA;

        } else {

            /* get width, height from table */

            width  = ff_svq1_frame_size_table[frame_size_code][0];

            height = ff_svq1_frame_size_table[frame_size_code][1];

        }

    }



    /* unknown fields */

    if (get_bits1(bitbuf)) {

        skip_bits1(bitbuf);    /* use packet checksum if (1) */

        skip_bits1(bitbuf);    /* component checksums after image data if (1) */



        if (get_bits(bitbuf, 2) != 0)

            return AVERROR_INVALIDDATA;

    }



    if (get_bits1(bitbuf)) {

        skip_bits1(bitbuf);

        skip_bits(bitbuf, 4);

        skip_bits1(bitbuf);

        skip_bits(bitbuf, 2);



        if (skip_1stop_8data_bits(bitbuf) < 0)

            return AVERROR_INVALIDDATA;

    }



    s->width  = width;

    s->height = height;

    return 0;

}
