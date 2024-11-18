static int svq1_decode_frame(AVCodecContext *avctx, void *data,

                             int *got_frame, AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size       = avpkt->size;

    SVQ1Context     *s = avctx->priv_data;

    AVFrame       *cur = data;

    uint8_t *current;

    int result, i, x, y, width, height;

    svq1_pmv *pmv;



    /* initialize bit buffer */

    init_get_bits(&s->gb, buf, buf_size * 8);



    /* decode frame header */

    s->frame_code = get_bits(&s->gb, 22);



    if ((s->frame_code & ~0x70) || !(s->frame_code & 0x60))

        return AVERROR_INVALIDDATA;



    /* swap some header bytes (why?) */

    if (s->frame_code != 0x20) {

        uint32_t *src = (uint32_t *)(buf + 4);



        for (i = 0; i < 4; i++)

            src[i] = ((src[i] << 16) | (src[i] >> 16)) ^ src[7 - i];

    }



    result = svq1_decode_frame_header(avctx, cur);



    if (result != 0) {

        av_dlog(avctx, "Error in svq1_decode_frame_header %i\n", result);

        return result;

    }



    result = ff_set_dimensions(avctx, s->width, s->height);

    if (result < 0)

        return result;



    if ((avctx->skip_frame >= AVDISCARD_NONREF && s->nonref) ||

        (avctx->skip_frame >= AVDISCARD_NONKEY &&

         cur->pict_type != AV_PICTURE_TYPE_I) ||

        avctx->skip_frame >= AVDISCARD_ALL)

        return buf_size;



    result = ff_get_buffer(avctx, cur, s->nonref ? 0 : AV_GET_BUFFER_FLAG_REF);

    if (result < 0)

        return result;



    pmv = av_malloc((FFALIGN(s->width, 16) / 8 + 3) * sizeof(*pmv));

    if (!pmv)

        return AVERROR(ENOMEM);



    /* decode y, u and v components */

    for (i = 0; i < 3; i++) {

        int linesize = cur->linesize[i];

        if (i == 0) {

            width    = FFALIGN(s->width,  16);

            height   = FFALIGN(s->height, 16);

        } else {

            if (avctx->flags & CODEC_FLAG_GRAY)

                break;

            width    = FFALIGN(s->width  / 4, 16);

            height   = FFALIGN(s->height / 4, 16);

        }



        current = cur->data[i];



        if (cur->pict_type == AV_PICTURE_TYPE_I) {

            /* keyframe */

            for (y = 0; y < height; y += 16) {

                for (x = 0; x < width; x += 16) {

                    result = svq1_decode_block_intra(&s->gb, &current[x],

                                                     linesize);

                    if (result != 0) {

                        av_log(avctx, AV_LOG_INFO,

                               "Error in svq1_decode_block %i (keyframe)\n",

                               result);

                        goto err;

                    }

                }

                current += 16 * linesize;

            }

        } else {

            /* delta frame */

            uint8_t *previous = s->prev->data[i];

            if (!previous ||

                s->prev->width != s->width || s->prev->height != s->height) {

                av_log(avctx, AV_LOG_ERROR, "Missing reference frame.\n");

                result = AVERROR_INVALIDDATA;

                goto err;

            }



            memset(pmv, 0, ((width / 8) + 3) * sizeof(svq1_pmv));



            for (y = 0; y < height; y += 16) {

                for (x = 0; x < width; x += 16) {

                    result = svq1_decode_delta_block(avctx, &s->hdsp,

                                                     &s->gb, &current[x],

                                                     previous, linesize,

                                                     pmv, x, y, width, height);

                    if (result != 0) {

                        av_dlog(avctx,

                                "Error in svq1_decode_delta_block %i\n",

                                result);

                        goto err;

                    }

                }



                pmv[0].x     =

                    pmv[0].y = 0;



                current += 16 * linesize;

            }

        }

    }



    if (!s->nonref) {

        av_frame_unref(s->prev);

        result = av_frame_ref(s->prev, cur);

        if (result < 0)

            goto err;

    }



    *got_frame = 1;

    result     = buf_size;



err:

    av_free(pmv);

    return result;

}
