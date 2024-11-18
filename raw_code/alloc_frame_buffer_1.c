static int alloc_frame_buffer(MpegEncContext *s, Picture *pic)

{

    int r;



    if (s->avctx->hwaccel) {

        assert(!pic->f.hwaccel_picture_private);

        if (s->avctx->hwaccel->priv_data_size) {

            pic->f.hwaccel_picture_private = av_mallocz(s->avctx->hwaccel->priv_data_size);

            if (!pic->f.hwaccel_picture_private) {

                av_log(s->avctx, AV_LOG_ERROR, "alloc_frame_buffer() failed (hwaccel private data allocation)\n");

                return -1;

            }

        }

    }



    if (s->codec_id != AV_CODEC_ID_WMV3IMAGE &&

        s->codec_id != AV_CODEC_ID_VC1IMAGE  &&

        s->codec_id != AV_CODEC_ID_MSS2)

        r = ff_thread_get_buffer(s->avctx, &pic->f);

    else

        r = avcodec_default_get_buffer(s->avctx, &pic->f);



    if (r < 0 || !pic->f.type || !pic->f.data[0]) {

        av_log(s->avctx, AV_LOG_ERROR, "get_buffer() failed (%d %d %p)\n",

               r, pic->f.type, pic->f.data[0]);

        av_freep(&pic->f.hwaccel_picture_private);

        return -1;

    }



    if (s->linesize && (s->linesize   != pic->f.linesize[0] ||

                        s->uvlinesize != pic->f.linesize[1])) {

        av_log(s->avctx, AV_LOG_ERROR,

               "get_buffer() failed (stride changed)\n");

        free_frame_buffer(s, pic);

        return -1;

    }



    if (pic->f.linesize[1] != pic->f.linesize[2]) {

        av_log(s->avctx, AV_LOG_ERROR,

               "get_buffer() failed (uv stride mismatch)\n");

        free_frame_buffer(s, pic);

        return -1;

    }



    return 0;

}
