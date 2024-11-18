static int svq1_decode_frame_header(GetBitContext *bitbuf, MpegEncContext *s)

{

    int frame_size_code;



    skip_bits(bitbuf, 8); /* temporal_reference */



    /* frame type */

    s->pict_type = get_bits(bitbuf, 2) + 1;

    if (s->pict_type == 4)

        return AVERROR_INVALIDDATA;



    if (s->pict_type == AV_PICTURE_TYPE_I) {

        /* unknown fields */

        if (s->f_code == 0x50 || s->f_code == 0x60) {

            int csum = get_bits(bitbuf, 16);



            csum = ff_svq1_packet_checksum(bitbuf->buffer,

                                           bitbuf->size_in_bits >> 3,

                                           csum);



            av_dlog(s->avctx, "%s checksum (%02x) for packet data\n",

                    (csum == 0) ? "correct" : "incorrect", csum);

        }



        if ((s->f_code ^ 0x10) >= 0x50) {

            uint8_t msg[256];



            svq1_parse_string(bitbuf, msg);



            av_log(s->avctx, AV_LOG_ERROR,

                   "embedded message: \"%s\"\n", (char *)msg);

        }



        skip_bits(bitbuf, 2);

        skip_bits(bitbuf, 2);

        skip_bits1(bitbuf);



        /* load frame size */

        frame_size_code = get_bits(bitbuf, 3);



        if (frame_size_code == 7) {

            /* load width, height (12 bits each) */

            s->width  = get_bits(bitbuf, 12);

            s->height = get_bits(bitbuf, 12);



            if (!s->width || !s->height)

                return AVERROR_INVALIDDATA;

        } else {

            /* get width, height from table */

            s->width  = ff_svq1_frame_size_table[frame_size_code].width;

            s->height = ff_svq1_frame_size_table[frame_size_code].height;

        }

    }



    /* unknown fields */

    if (get_bits1(bitbuf) == 1) {

        skip_bits1(bitbuf);    /* use packet checksum if (1) */

        skip_bits1(bitbuf);    /* component checksums after image data if (1) */



        if (get_bits(bitbuf, 2) != 0)

            return AVERROR_INVALIDDATA;

    }



    if (get_bits1(bitbuf) == 1) {

        skip_bits1(bitbuf);

        skip_bits(bitbuf, 4);

        skip_bits1(bitbuf);

        skip_bits(bitbuf, 2);



        while (get_bits1(bitbuf) == 1)

            skip_bits(bitbuf, 8);

    }



    return 0;

}
