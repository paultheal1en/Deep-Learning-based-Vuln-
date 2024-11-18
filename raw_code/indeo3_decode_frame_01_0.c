static int indeo3_decode_frame(AVCodecContext *avctx,

                               void *data, int *data_size,

                               unsigned char *buf, int buf_size)

{

    Indeo3DecodeContext *s=avctx->priv_data;

    unsigned char *src, *dest;

    int y;



    /* no supplementary picture */

    if (buf_size == 0) {

        return 0;

    }



    iv_decode_frame(s, buf, buf_size);



    if(s->frame.data[0])

        avctx->release_buffer(avctx, &s->frame);



    s->frame.reference = 0;

    if(avctx->get_buffer(avctx, &s->frame) < 0) {

        av_log(s->avctx, AV_LOG_ERROR, "get_buffer() failed\n");

        return -1;

    }



    src = s->cur_frame->Ybuf;

    dest = s->frame.data[0];

    for (y = 0; y < s->height; y++) {

      memcpy(dest, src, s->cur_frame->y_w);

      src += s->cur_frame->y_w;

      dest += s->frame.linesize[0];

    }



    if (!(s->avctx->flags & CODEC_FLAG_GRAY))

    {

    src = s->cur_frame->Ubuf;

    dest = s->frame.data[1];

    for (y = 0; y < s->height / 4; y++) {

      memcpy(dest, src, s->cur_frame->uv_w);

      src += s->cur_frame->uv_w;

      dest += s->frame.linesize[1];

    }



    src = s->cur_frame->Vbuf;

    dest = s->frame.data[2];

    for (y = 0; y < s->height / 4; y++) {

      memcpy(dest, src, s->cur_frame->uv_w);

      src += s->cur_frame->uv_w;

      dest += s->frame.linesize[2];

    }

    }



    *data_size=sizeof(AVFrame);

    *(AVFrame*)data= s->frame;



    return buf_size;

}
