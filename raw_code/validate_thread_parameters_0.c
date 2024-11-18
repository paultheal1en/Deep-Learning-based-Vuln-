static void validate_thread_parameters(AVCodecContext *avctx)

{

    int frame_threading_supported = (avctx->codec->capabilities & CODEC_CAP_FRAME_THREADS)

                                && !(avctx->flags & CODEC_FLAG_TRUNCATED)

                                && !(avctx->flags & CODEC_FLAG_LOW_DELAY)

                                && !(avctx->flags2 & CODEC_FLAG2_CHUNKS);

    if (avctx->thread_count == 1) {

        avctx->active_thread_type = 0;

    } else if (frame_threading_supported && (avctx->thread_type & FF_THREAD_FRAME)) {

        avctx->active_thread_type = FF_THREAD_FRAME;

    } else {

        avctx->active_thread_type = FF_THREAD_SLICE;

    }

}
