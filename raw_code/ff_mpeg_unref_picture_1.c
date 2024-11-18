void ff_mpeg_unref_picture(MpegEncContext *s, Picture *pic)

{

    int off = offsetof(Picture, mb_mean) + sizeof(pic->mb_mean);



    pic->tf.f = &pic->f;

    /* WM Image / Screen codecs allocate internal buffers with different

     * dimensions / colorspaces; ignore user-defined callbacks for these. */

    if (s->codec_id != AV_CODEC_ID_WMV3IMAGE &&

        s->codec_id != AV_CODEC_ID_VC1IMAGE  &&

        s->codec_id != AV_CODEC_ID_MSS2)

        ff_thread_release_buffer(s->avctx, &pic->tf);

    else

        av_frame_unref(&pic->f);



    av_buffer_unref(&pic->hwaccel_priv_buf);



    if (pic->needs_realloc)

        ff_free_picture_tables(pic);



    memset((uint8_t*)pic + off, 0, sizeof(*pic) - off);

}
