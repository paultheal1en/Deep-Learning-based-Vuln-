int ff_h264_execute_decode_slices(H264Context *h, unsigned context_count)

{

    AVCodecContext *const avctx = h->avctx;

    H264SliceContext *sl;

    int i;



    av_assert0(context_count && h->slice_ctx[context_count - 1].mb_y < h->mb_height);



    if (h->avctx->hwaccel ||

        h->avctx->codec->capabilities & CODEC_CAP_HWACCEL_VDPAU)

        return 0;

    if (context_count == 1) {

        int ret = decode_slice(avctx, &h->slice_ctx[0]);

        h->mb_y = h->slice_ctx[0].mb_y;

        return ret;

    } else {

        av_assert0(context_count > 0);

        for (i = 1; i < context_count; i++) {

            sl                 = &h->slice_ctx[i];

            if (CONFIG_ERROR_RESILIENCE) {

                sl->er.error_count = 0;

            }

        }



        avctx->execute(avctx, decode_slice, h->slice_ctx,

                       NULL, context_count, sizeof(h->slice_ctx[0]));



        /* pull back stuff from slices to master context */

        sl                   = &h->slice_ctx[context_count - 1];

        h->mb_y              = sl->mb_y;

        if (CONFIG_ERROR_RESILIENCE) {

            for (i = 1; i < context_count; i++)

                h->slice_ctx[0].er.error_count += h->slice_ctx[i].er.error_count;

        }

    }



    return 0;

}
