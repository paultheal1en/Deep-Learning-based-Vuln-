static int execute_decode_slices(H264Context *h, int context_count)

{

    MpegEncContext *const s     = &h->s;

    AVCodecContext *const avctx = s->avctx;

    H264Context *hx;

    int i;



    if (s->avctx->hwaccel ||

        s->avctx->codec->capabilities & CODEC_CAP_HWACCEL_VDPAU)

        return 0;

    if (context_count == 1) {

        return decode_slice(avctx, &h);

    } else {


        for (i = 1; i < context_count; i++) {

            hx                    = h->thread_context[i];

            hx->s.err_recognition = avctx->err_recognition;

            hx->s.error_count     = 0;

            hx->x264_build        = h->x264_build;

        }



        avctx->execute(avctx, decode_slice, h->thread_context,

                       NULL, context_count, sizeof(void *));



        /* pull back stuff from slices to master context */

        hx                   = h->thread_context[context_count - 1];

        s->mb_x              = hx->s.mb_x;

        s->mb_y              = hx->s.mb_y;

        s->droppable         = hx->s.droppable;

        s->picture_structure = hx->s.picture_structure;

        for (i = 1; i < context_count; i++)

            h->s.error_count += h->thread_context[i]->s.error_count;

    }



    return 0;

}