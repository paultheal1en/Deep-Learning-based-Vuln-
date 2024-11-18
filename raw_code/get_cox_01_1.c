static int get_cox(Jpeg2000DecoderContext *s, Jpeg2000CodingStyle *c)

{

    uint8_t byte;



    if (s->buf_end - s->buf < 5)

        return AVERROR(EINVAL);

    /*  nreslevels = number of resolution levels

                   = number of decomposition level +1 */

    c->nreslevels = bytestream_get_byte(&s->buf) + 1;



    if (c->nreslevels > JPEG2000_MAX_RESLEVELS)

        return AVERROR_INVALIDDATA;



    /* compute number of resolution levels to decode */

    if (c->nreslevels < s->reduction_factor)

        c->nreslevels2decode = 1;

    else

        c->nreslevels2decode = c->nreslevels - s->reduction_factor;



    c->log2_cblk_width  = bytestream_get_byte(&s->buf) + 2; // cblk width

    c->log2_cblk_height = bytestream_get_byte(&s->buf) + 2; // cblk height



    if (c->log2_cblk_width > 10 || c->log2_cblk_height > 10 ||

        c->log2_cblk_width + c->log2_cblk_height > 12) {

        av_log(s->avctx, AV_LOG_ERROR, "cblk size invalid\n");

        return AVERROR_INVALIDDATA;

    }



    c->cblk_style = bytestream_get_byte(&s->buf);

    if (c->cblk_style != 0) { // cblk style

        avpriv_request_sample(s->avctx, "Support for extra cblk styles");

        return AVERROR_PATCHWELCOME;

    }

    c->transform = bytestream_get_byte(&s->buf); // DWT transformation type

    /* set integer 9/7 DWT in case of BITEXACT flag */

    if ((s->avctx->flags & CODEC_FLAG_BITEXACT) && (c->transform == FF_DWT97))

        c->transform = FF_DWT97_INT;



    if (c->csty & JPEG2000_CSTY_PREC) {

        int i;

        for (i = 0; i < c->nreslevels; i++) {

            byte = bytestream_get_byte(&s->buf);

            c->log2_prec_widths[i]  =  byte       & 0x0F;    // precinct PPx

            c->log2_prec_heights[i] = (byte >> 4) & 0x0F;    // precinct PPy

        }

    }

    return 0;

}
