int ff_alloc_picture(MpegEncContext *s, Picture *pic, int shared)

{

    int i, ret;



    if (shared) {

        assert(pic->f.data[0]);

        pic->shared = 1;

    } else {

        assert(!pic->f.data[0]);



        if (alloc_frame_buffer(s, pic) < 0)

            return -1;



        s->linesize   = pic->f.linesize[0];

        s->uvlinesize = pic->f.linesize[1];

    }



    if (!pic->qscale_table_buf)

        ret = alloc_picture_tables(s, pic);

    else

        ret = make_tables_writable(pic);

    if (ret < 0)

        goto fail;



    if (s->encoding) {

        pic->mb_var    = (uint16_t*)pic->mb_var_buf->data;

        pic->mc_mb_var = (uint16_t*)pic->mc_mb_var_buf->data;

        pic->mb_mean   = pic->mb_mean_buf->data;

    }



    pic->mbskip_table = pic->mbskip_table_buf->data;

    pic->qscale_table = pic->qscale_table_buf->data + 2 * s->mb_stride + 1;

    pic->mb_type      = (uint32_t*)pic->mb_type_buf->data + 2 * s->mb_stride + 1;



    if (pic->motion_val_buf[0]) {

        for (i = 0; i < 2; i++) {

            pic->motion_val[i] = (int16_t (*)[2])pic->motion_val_buf[i]->data + 4;

            pic->ref_index[i]  = pic->ref_index_buf[i]->data;

        }

    }



    return 0;

fail:

    av_log(s->avctx, AV_LOG_ERROR, "Error allocating a picture.\n");

    ff_mpeg_unref_picture(s, pic);

    free_picture_tables(pic);

    return AVERROR(ENOMEM);

}
