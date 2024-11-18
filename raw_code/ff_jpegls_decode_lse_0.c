int ff_jpegls_decode_lse(MJpegDecodeContext *s)

{

    int id;

    int tid, wt, maxtab, i, j;



    int len = get_bits(&s->gb, 16);  /* length: FIXME: verify field validity */

    id = get_bits(&s->gb, 8);



    switch (id) {

    case 1:

        s->maxval = get_bits(&s->gb, 16);

        s->t1     = get_bits(&s->gb, 16);

        s->t2     = get_bits(&s->gb, 16);

        s->t3     = get_bits(&s->gb, 16);

        s->reset  = get_bits(&s->gb, 16);



//        ff_jpegls_reset_coding_parameters(s, 0);

        //FIXME quant table?

        break;

    case 2:

        s->palette_index = 0;

    case 3:

        tid= get_bits(&s->gb, 8);

        wt = get_bits(&s->gb, 8);



        if (len < 5)

            return AVERROR_INVALIDDATA;



        if (wt < 1 || wt > MAX_COMPONENTS) {

            avpriv_request_sample(s->avctx, "wt %d", wt);

            return AVERROR_PATCHWELCOME;

        }



        if (!s->maxval)

            maxtab = 255;

        else if ((5 + wt*(s->maxval+1)) < 65535)

            maxtab = s->maxval;

        else

            maxtab = 65530/wt - 1;



        if(s->avctx->debug & FF_DEBUG_PICT_INFO) {

            av_log(s->avctx, AV_LOG_DEBUG, "LSE palette %d tid:%d wt:%d maxtab:%d\n", id, tid, wt, maxtab);

        }

        if (maxtab >= 256) {

            avpriv_request_sample(s->avctx, ">8bit palette");

            return AVERROR_PATCHWELCOME;

        }

        maxtab = FFMIN(maxtab, (len - 5) / wt + s->palette_index);



        if (s->palette_index > maxtab)

            return AVERROR_INVALIDDATA;



        if ((s->avctx->pix_fmt == AV_PIX_FMT_GRAY8 || s->avctx->pix_fmt == AV_PIX_FMT_PAL8) &&

            (s->picture_ptr->format == AV_PIX_FMT_GRAY8 || s->picture_ptr->format == AV_PIX_FMT_PAL8)) {

            uint32_t *pal = s->picture_ptr->data[1];

            s->picture_ptr->format =

            s->avctx->pix_fmt = AV_PIX_FMT_PAL8;

            for (i=s->palette_index; i<=maxtab; i++) {

                pal[i] = 0;

                for (j=0; j<wt; j++) {

                    pal[i] |= get_bits(&s->gb, 8) << (8*(wt-j-1));

                }

            }

            s->palette_index = i;

        }

        break;

    case 4:

        avpriv_request_sample(s->avctx, "oversize image");

        return AVERROR(ENOSYS);

    default:

        av_log(s->avctx, AV_LOG_ERROR, "invalid id %d\n", id);

        return AVERROR_INVALIDDATA;

    }

    av_dlog(s->avctx, "ID=%i, T=%i,%i,%i\n", id, s->t1, s->t2, s->t3);



    return 0;

}
