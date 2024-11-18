static av_cold int g722_encode_init(AVCodecContext * avctx)

{

    G722Context *c = avctx->priv_data;

    int ret;



    if (avctx->channels != 1) {

        av_log(avctx, AV_LOG_ERROR, "Only mono tracks are allowed.\n");

        return AVERROR_INVALIDDATA;

    }



    c->band[0].scale_factor = 8;

    c->band[1].scale_factor = 2;

    c->prev_samples_pos = 22;



    if (avctx->trellis) {

        int frontier = 1 << avctx->trellis;

        int max_paths = frontier * FREEZE_INTERVAL;

        int i;

        for (i = 0; i < 2; i++) {

            c->paths[i] = av_mallocz(max_paths * sizeof(**c->paths));

            c->node_buf[i] = av_mallocz(2 * frontier * sizeof(**c->node_buf));

            c->nodep_buf[i] = av_mallocz(2 * frontier * sizeof(**c->nodep_buf));

            if (!c->paths[i] || !c->node_buf[i] || !c->nodep_buf[i]) {

                ret = AVERROR(ENOMEM);

                goto error;

            }

        }

    }



    if (avctx->frame_size) {

        /* validate frame size */

        if (avctx->frame_size & 1 || avctx->frame_size > MAX_FRAME_SIZE) {

            int new_frame_size;



            if (avctx->frame_size == 1)

                new_frame_size = 2;

            else if (avctx->frame_size > MAX_FRAME_SIZE)

                new_frame_size = MAX_FRAME_SIZE;

            else

                new_frame_size = avctx->frame_size - 1;



            av_log(avctx, AV_LOG_WARNING, "Requested frame size is not "

                   "allowed. Using %d instead of %d\n", new_frame_size,

                   avctx->frame_size);

            avctx->frame_size = new_frame_size;

        }

    } else {

        /* This is arbitrary. We use 320 because it's 20ms @ 16kHz, which is

           a common packet size for VoIP applications */

        avctx->frame_size = 320;

    }

    avctx->delay = 22;



    if (avctx->trellis) {

        /* validate trellis */

        if (avctx->trellis < MIN_TRELLIS || avctx->trellis > MAX_TRELLIS) {

            int new_trellis = av_clip(avctx->trellis, MIN_TRELLIS, MAX_TRELLIS);

            av_log(avctx, AV_LOG_WARNING, "Requested trellis value is not "

                   "allowed. Using %d instead of %d\n", new_trellis,

                   avctx->trellis);

            avctx->trellis = new_trellis;

        }

    }



    return 0;

error:

    g722_encode_close(avctx);

    return ret;

}
