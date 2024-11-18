static av_cold int theora_decode_init(AVCodecContext *avctx)

{

    Vp3DecodeContext *s = avctx->priv_data;

    GetBitContext gb;

    int ptype;

    const uint8_t *header_start[3];

    int header_len[3];

    int i;



    avctx->pix_fmt = AV_PIX_FMT_YUV420P;



    s->theora = 1;



    if (!avctx->extradata_size) {

        av_log(avctx, AV_LOG_ERROR, "Missing extradata!\n");

        return -1;

    }



    if (avpriv_split_xiph_headers(avctx->extradata, avctx->extradata_size,

                                  42, header_start, header_len) < 0) {

        av_log(avctx, AV_LOG_ERROR, "Corrupt extradata\n");

        return -1;

    }



    for (i = 0; i < 3; i++) {

        if (header_len[i] <= 0)

            continue;

        init_get_bits8(&gb, header_start[i], header_len[i]);



        ptype = get_bits(&gb, 8);



        if (!(ptype & 0x80)) {

            av_log(avctx, AV_LOG_ERROR, "Invalid extradata!\n");

//          return -1;

        }



        // FIXME: Check for this as well.

        skip_bits_long(&gb, 6 * 8); /* "theora" */



        switch (ptype) {

        case 0x80:

            if (theora_decode_header(avctx, &gb) < 0)

                return -1;

            break;

        case 0x81:

// FIXME: is this needed? it breaks sometimes

//            theora_decode_comments(avctx, gb);

            break;

        case 0x82:

            if (theora_decode_tables(avctx, &gb))

                return -1;

            break;

        default:

            av_log(avctx, AV_LOG_ERROR,

                   "Unknown Theora config packet: %d\n", ptype & ~0x80);

            break;

        }

        if (ptype != 0x81 && 8 * header_len[i] != get_bits_count(&gb))

            av_log(avctx, AV_LOG_WARNING,

                   "%d bits left in packet %X\n",

                   8 * header_len[i] - get_bits_count(&gb), ptype);

        if (s->theora < 0x030200)

            break;

    }



    return vp3_decode_init(avctx);

}
