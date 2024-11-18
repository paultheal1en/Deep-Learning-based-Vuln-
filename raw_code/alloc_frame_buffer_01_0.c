static int alloc_frame_buffer(AVCodecContext *avctx,  Picture *pic,

                              MotionEstContext *me, ScratchpadContext *sc,

                              int chroma_x_shift, int chroma_y_shift,

                              int linesize, int uvlinesize)

{

    int edges_needed = av_codec_is_encoder(avctx->codec);

    int r, ret;



    pic->tf.f = pic->f;

    if (avctx->codec_id != AV_CODEC_ID_WMV3IMAGE &&

        avctx->codec_id != AV_CODEC_ID_VC1IMAGE  &&

        avctx->codec_id != AV_CODEC_ID_MSS2) {

        if (edges_needed) {

            pic->f->width  = avctx->width  + 2 * EDGE_WIDTH;

            pic->f->height = avctx->height + 2 * EDGE_WIDTH;

        }



        r = ff_thread_get_buffer(avctx, &pic->tf,

                                 pic->reference ? AV_GET_BUFFER_FLAG_REF : 0);

    } else {

        pic->f->width  = avctx->width;

        pic->f->height = avctx->height;

        pic->f->format = avctx->pix_fmt;

        r = avcodec_default_get_buffer2(avctx, pic->f, 0);

    }



    if (r < 0 || !pic->f->buf[0]) {

        av_log(avctx, AV_LOG_ERROR, "get_buffer() failed (%d %p)\n",

               r, pic->f->data[0]);

        return -1;

    }



    if (edges_needed) {

        int i;

        for (i = 0; pic->f->data[i]; i++) {

            int offset = (EDGE_WIDTH >> (i ? chroma_y_shift : 0)) *

                         pic->f->linesize[i] +

                         (EDGE_WIDTH >> (i ? chroma_x_shift : 0));

            pic->f->data[i] += offset;

        }

        pic->f->width  = avctx->width;

        pic->f->height = avctx->height;

    }



    if (avctx->hwaccel) {

        assert(!pic->hwaccel_picture_private);

        if (avctx->hwaccel->frame_priv_data_size) {

            pic->hwaccel_priv_buf = av_buffer_allocz(avctx->hwaccel->frame_priv_data_size);

            if (!pic->hwaccel_priv_buf) {

                av_log(avctx, AV_LOG_ERROR, "alloc_frame_buffer() failed (hwaccel private data allocation)\n");

                return -1;

            }

            pic->hwaccel_picture_private = pic->hwaccel_priv_buf->data;

        }

    }



    if (linesize && (linesize   != pic->f->linesize[0] ||

                     uvlinesize != pic->f->linesize[1])) {

        av_log(avctx, AV_LOG_ERROR,

               "get_buffer() failed (stride changed)\n");

        ff_mpeg_unref_picture(avctx, pic);

        return -1;

    }



    if (pic->f->linesize[1] != pic->f->linesize[2]) {

        av_log(avctx, AV_LOG_ERROR,

               "get_buffer() failed (uv stride mismatch)\n");

        ff_mpeg_unref_picture(avctx, pic);

        return -1;

    }



    if (!sc->edge_emu_buffer &&

        (ret = ff_mpeg_framesize_alloc(avctx, me, sc,

                                       pic->f->linesize[0])) < 0) {

        av_log(avctx, AV_LOG_ERROR,

               "get_buffer() failed to allocate context scratch buffers.\n");

        ff_mpeg_unref_picture(avctx, pic);

        return ret;

    }



    return 0;

}
