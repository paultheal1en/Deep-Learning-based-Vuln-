static int h264_decode_frame(AVCodecContext *avctx, void *data,

                             int *got_frame, AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size       = avpkt->size;

    H264Context *h     = avctx->priv_data;

    AVFrame *pict      = data;

    int buf_index      = 0;

    H264Picture *out;

    int i, out_idx;

    int ret;



    h->flags = avctx->flags;

    h->setup_finished = 0;



    if (h->backup_width != -1) {

        avctx->width    = h->backup_width;

        h->backup_width = -1;

    }

    if (h->backup_height != -1) {

        avctx->height    = h->backup_height;

        h->backup_height = -1;

    }

    if (h->backup_pix_fmt != AV_PIX_FMT_NONE) {

        avctx->pix_fmt    = h->backup_pix_fmt;

        h->backup_pix_fmt = AV_PIX_FMT_NONE;

    }



    ff_h264_unref_picture(h, &h->last_pic_for_ec);



    /* end of stream, output what is still in the buffers */

    if (buf_size == 0) {

 out:



        h->cur_pic_ptr = NULL;

        h->first_field = 0;



        // FIXME factorize this with the output code below

        out     = h->delayed_pic[0];

        out_idx = 0;

        for (i = 1;

             h->delayed_pic[i] &&

             !h->delayed_pic[i]->f->key_frame &&

             !h->delayed_pic[i]->mmco_reset;

             i++)

            if (h->delayed_pic[i]->poc < out->poc) {

                out     = h->delayed_pic[i];

                out_idx = i;

            }



        for (i = out_idx; h->delayed_pic[i]; i++)

            h->delayed_pic[i] = h->delayed_pic[i + 1];



        if (out) {

            out->reference &= ~DELAYED_PIC_REF;

            ret = output_frame(h, pict, out);

            if (ret < 0)

                return ret;

            *got_frame = 1;

        }



        return buf_index;

    }

    if (h->is_avc && av_packet_get_side_data(avpkt, AV_PKT_DATA_NEW_EXTRADATA, NULL)) {

        int side_size;

        uint8_t *side = av_packet_get_side_data(avpkt, AV_PKT_DATA_NEW_EXTRADATA, &side_size);

        if (is_extra(side, side_size))

            ff_h264_decode_extradata(h, side, side_size);

    }

    if(h->is_avc && buf_size >= 9 && buf[0]==1 && buf[2]==0 && (buf[4]&0xFC)==0xFC && (buf[5]&0x1F) && buf[8]==0x67){

        if (is_extra(buf, buf_size))

            return ff_h264_decode_extradata(h, buf, buf_size);

    }



    buf_index = decode_nal_units(h, buf, buf_size, 0);

    if (buf_index < 0)

        return AVERROR_INVALIDDATA;



    if (!h->cur_pic_ptr && h->nal_unit_type == NAL_END_SEQUENCE) {

        av_assert0(buf_index <= buf_size);

        goto out;

    }



    if (!(avctx->flags2 & AV_CODEC_FLAG2_CHUNKS) && !h->cur_pic_ptr) {

        if (avctx->skip_frame >= AVDISCARD_NONREF ||

            buf_size >= 4 && !memcmp("Q264", buf, 4))

            return buf_size;

        av_log(avctx, AV_LOG_ERROR, "no frame!\n");

        return AVERROR_INVALIDDATA;

    }



    if (!(avctx->flags2 & AV_CODEC_FLAG2_CHUNKS) ||

        (h->mb_y >= h->mb_height && h->mb_height)) {

        if (avctx->flags2 & AV_CODEC_FLAG2_CHUNKS)

            decode_postinit(h, 1);



        if ((ret = ff_h264_field_end(h, &h->slice_ctx[0], 0)) < 0)

            return ret;



        /* Wait for second field. */

        *got_frame = 0;

        if (h->next_output_pic && (

                                   h->next_output_pic->recovered)) {

            if (!h->next_output_pic->recovered)

                h->next_output_pic->f->flags |= AV_FRAME_FLAG_CORRUPT;



            if (!h->avctx->hwaccel &&

                 (h->next_output_pic->field_poc[0] == INT_MAX ||

                  h->next_output_pic->field_poc[1] == INT_MAX)

            ) {

                int p;

                AVFrame *f = h->next_output_pic->f;

                int field = h->next_output_pic->field_poc[0] == INT_MAX;

                uint8_t *dst_data[4];

                int linesizes[4];

                const uint8_t *src_data[4];



                av_log(h->avctx, AV_LOG_DEBUG, "Duplicating field %d to fill missing\n", field);



                for (p = 0; p<4; p++) {

                    dst_data[p] = f->data[p] + (field^1)*f->linesize[p];

                    src_data[p] = f->data[p] +  field   *f->linesize[p];

                    linesizes[p] = 2*f->linesize[p];

                }



                av_image_copy(dst_data, linesizes, src_data, linesizes,

                              f->format, f->width, f->height>>1);

            }



            ret = output_frame(h, pict, h->next_output_pic);

            if (ret < 0)

                return ret;

            *got_frame = 1;

            if (CONFIG_MPEGVIDEO) {

                ff_print_debug_info2(h->avctx, pict, NULL,

                                    h->next_output_pic->mb_type,

                                    h->next_output_pic->qscale_table,

                                    h->next_output_pic->motion_val,

                                    &h->low_delay,

                                    h->mb_width, h->mb_height, h->mb_stride, 1);

            }

        }

    }



    av_assert0(pict->buf[0] || !*got_frame);



    ff_h264_unref_picture(h, &h->last_pic_for_ec);



    return get_consumed_bytes(buf_index, buf_size);

}
