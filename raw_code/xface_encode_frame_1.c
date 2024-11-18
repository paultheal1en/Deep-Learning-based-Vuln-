static int xface_encode_frame(AVCodecContext *avctx, AVPacket *pkt,

                              const AVFrame *frame, int *got_packet)

{

    XFaceContext *xface = avctx->priv_data;

    ProbRangesQueue pq = {{ 0 }, 0};

    uint8_t bitmap_copy[XFACE_PIXELS];

    BigInt b = {0};

    int i, j, k, ret = 0;

    const uint8_t *buf;

    uint8_t *p;

    char intbuf[XFACE_MAX_DIGITS];



    if (avctx->width || avctx->height) {

        if (avctx->width != XFACE_WIDTH || avctx->height != XFACE_HEIGHT) {

            av_log(avctx, AV_LOG_ERROR,

                   "Size value %dx%d not supported, only accepts a size of %dx%d\n",

                   avctx->width, avctx->height, XFACE_WIDTH, XFACE_HEIGHT);

            return AVERROR(EINVAL);

        }

    }

    avctx->width  = XFACE_WIDTH;

    avctx->height = XFACE_HEIGHT;



    /* convert image from MONOWHITE to 1=black 0=white bitmap */

    buf = frame->data[0];

    i = j = 0;

    do {

        for (k = 0; k < 8; k++)

            xface->bitmap[i++] = (buf[j]>>(7-k))&1;

        if (++j == XFACE_WIDTH/8) {

            buf += frame->linesize[0];

            j = 0;

        }

    } while (i < XFACE_PIXELS);



    /* create a copy of bitmap */

    memcpy(bitmap_copy, xface->bitmap, XFACE_PIXELS);

    ff_xface_generate_face(xface->bitmap, bitmap_copy);



    encode_block(xface->bitmap,                         16, 16, 0, &pq);

    encode_block(xface->bitmap + 16,                    16, 16, 0, &pq);

    encode_block(xface->bitmap + 32,                    16, 16, 0, &pq);

    encode_block(xface->bitmap + XFACE_WIDTH * 16,      16, 16, 0, &pq);

    encode_block(xface->bitmap + XFACE_WIDTH * 16 + 16, 16, 16, 0, &pq);

    encode_block(xface->bitmap + XFACE_WIDTH * 16 + 32, 16, 16, 0, &pq);

    encode_block(xface->bitmap + XFACE_WIDTH * 32,      16, 16, 0, &pq);

    encode_block(xface->bitmap + XFACE_WIDTH * 32 + 16, 16, 16, 0, &pq);

    encode_block(xface->bitmap + XFACE_WIDTH * 32 + 32, 16, 16, 0, &pq);



    while (pq.prob_ranges_idx > 0)

        push_integer(&b, pq.prob_ranges[--pq.prob_ranges_idx]);



    /* write the inverted big integer in b to intbuf */

    i = 0;


    while (b.nb_words) {

        uint8_t r;

        ff_big_div(&b, XFACE_PRINTS, &r);


        intbuf[i++] = r + XFACE_FIRST_PRINT;

    }



    if ((ret = ff_alloc_packet2(avctx, pkt, i+2)) < 0)

        return ret;



    /* revert the number, and close the buffer */

    p = pkt->data;

    while (--i >= 0)

        *(p++) = intbuf[i];

    *(p++) = '\n';

    *(p++) = 0;



    pkt->flags |= AV_PKT_FLAG_KEY;

    *got_packet = 1;



    return 0;

}