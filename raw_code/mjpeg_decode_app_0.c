static int mjpeg_decode_app(MJpegDecodeContext *s)

{

    int len, id, i;



    len = get_bits(&s->gb, 16);

    if (len < 5)

        return AVERROR_INVALIDDATA;

    if (8 * len > get_bits_left(&s->gb))

        return AVERROR_INVALIDDATA;



    id   = get_bits_long(&s->gb, 32);

    id   = av_be2ne32(id);

    len -= 6;



    if (s->avctx->debug & FF_DEBUG_STARTCODE)

        av_log(s->avctx, AV_LOG_DEBUG, "APPx %8X\n", id);



    /* Buggy AVID, it puts EOI only at every 10th frame. */

    /* Also, this fourcc is used by non-avid files too, it holds some

       information, but it's always present in AVID-created files. */

    if (id == AV_RL32("AVI1")) {

        /* structure:

            4bytes      AVI1

            1bytes      polarity

            1bytes      always zero

            4bytes      field_size

            4bytes      field_size_less_padding

        */

            s->buggy_avid = 1;

        i = get_bits(&s->gb, 8); len--;

        av_log(s->avctx, AV_LOG_DEBUG, "polarity %d\n", i);

#if 0

        skip_bits(&s->gb, 8);

        skip_bits(&s->gb, 32);

        skip_bits(&s->gb, 32);

        len -= 10;

#endif

        goto out;

    }



//    len -= 2;



    if (id == AV_RL32("JFIF")) {

        int t_w, t_h, v1, v2;

        skip_bits(&s->gb, 8); /* the trailing zero-byte */

        v1 = get_bits(&s->gb, 8);

        v2 = get_bits(&s->gb, 8);

        skip_bits(&s->gb, 8);



        s->avctx->sample_aspect_ratio.num = get_bits(&s->gb, 16);

        s->avctx->sample_aspect_ratio.den = get_bits(&s->gb, 16);



        if (s->avctx->debug & FF_DEBUG_PICT_INFO)

            av_log(s->avctx, AV_LOG_INFO,

                   "mjpeg: JFIF header found (version: %x.%x) SAR=%d/%d\n",

                   v1, v2,

                   s->avctx->sample_aspect_ratio.num,

                   s->avctx->sample_aspect_ratio.den);



        t_w = get_bits(&s->gb, 8);

        t_h = get_bits(&s->gb, 8);

        if (t_w && t_h) {

            /* skip thumbnail */

            if (len -10 - (t_w * t_h * 3) > 0)

                len -= t_w * t_h * 3;

        }

        len -= 10;

        goto out;

    }



    if (id == AV_RL32("Adob") && (get_bits(&s->gb, 8) == 'e')) {

        if (s->avctx->debug & FF_DEBUG_PICT_INFO)

            av_log(s->avctx, AV_LOG_INFO, "mjpeg: Adobe header found\n");

        skip_bits(&s->gb, 16); /* version */

        skip_bits(&s->gb, 16); /* flags0 */

        skip_bits(&s->gb, 16); /* flags1 */

        skip_bits(&s->gb,  8); /* transform */

        len -= 7;

        goto out;

    }



    if (id == AV_RL32("LJIF")) {

        if (s->avctx->debug & FF_DEBUG_PICT_INFO)

            av_log(s->avctx, AV_LOG_INFO,

                   "Pegasus lossless jpeg header found\n");

        skip_bits(&s->gb, 16); /* version ? */

        skip_bits(&s->gb, 16); /* unknwon always 0? */

        skip_bits(&s->gb, 16); /* unknwon always 0? */

        skip_bits(&s->gb, 16); /* unknwon always 0? */

        switch (get_bits(&s->gb, 8)) {

        case 1:

            s->rgb         = 1;

            s->pegasus_rct = 0;

            break;

        case 2:

            s->rgb         = 1;

            s->pegasus_rct = 1;

            break;

        default:

            av_log(s->avctx, AV_LOG_ERROR, "unknown colorspace\n");

        }

        len -= 9;

        goto out;

    }



    /* Apple MJPEG-A */

    if ((s->start_code == APP1) && (len > (0x28 - 8))) {

        id   = get_bits_long(&s->gb, 32);

        id   = av_be2ne32(id);

        len -= 4;

        /* Apple MJPEG-A */

        if (id == AV_RL32("mjpg")) {

#if 0

            skip_bits(&s->gb, 32); /* field size */

            skip_bits(&s->gb, 32); /* pad field size */

            skip_bits(&s->gb, 32); /* next off */

            skip_bits(&s->gb, 32); /* quant off */

            skip_bits(&s->gb, 32); /* huff off */

            skip_bits(&s->gb, 32); /* image off */

            skip_bits(&s->gb, 32); /* scan off */

            skip_bits(&s->gb, 32); /* data off */

#endif

            if (s->avctx->debug & FF_DEBUG_PICT_INFO)

                av_log(s->avctx, AV_LOG_INFO, "mjpeg: Apple MJPEG-A header found\n");

        }

    }



out:

    /* slow but needed for extreme adobe jpegs */

    if (len < 0)

        av_log(s->avctx, AV_LOG_ERROR,

               "mjpeg: error, decode_app parser read over the end\n");

    while (--len > 0)

        skip_bits(&s->gb, 8);



    return 0;

}
