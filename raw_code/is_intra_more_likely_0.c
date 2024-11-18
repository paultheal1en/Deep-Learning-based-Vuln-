static int is_intra_more_likely(ERContext *s)

{

    int is_intra_likely, i, j, undamaged_count, skip_amount, mb_x, mb_y;



    if (!s->last_pic.f || !s->last_pic.f->data[0])

        return 1; // no previous frame available -> use spatial prediction



    undamaged_count = 0;

    for (i = 0; i < s->mb_num; i++) {

        const int mb_xy = s->mb_index2xy[i];

        const int error = s->error_status_table[mb_xy];

        if (!((error & ER_DC_ERROR) && (error & ER_MV_ERROR)))

            undamaged_count++;

    }



    if (s->avctx->codec_id == AV_CODEC_ID_H264 && s->ref_count <= 0)

        return 1;



    if (undamaged_count < 5)

        return 0; // almost all MBs damaged -> use temporal prediction



#if FF_API_XVMC

FF_DISABLE_DEPRECATION_WARNINGS

    // prevent dsp.sad() check, that requires access to the image

    if (CONFIG_MPEG_XVMC_DECODER    &&

        s->avctx->xvmc_acceleration &&

        s->cur_pic.f->pict_type == AV_PICTURE_TYPE_I)

        return 1;

FF_ENABLE_DEPRECATION_WARNINGS

#endif /* FF_API_XVMC */



    skip_amount     = FFMAX(undamaged_count / 50, 1); // check only up to 50 MBs

    is_intra_likely = 0;



    j = 0;

    for (mb_y = 0; mb_y < s->mb_height - 1; mb_y++) {

        for (mb_x = 0; mb_x < s->mb_width; mb_x++) {

            int error;

            const int mb_xy = mb_x + mb_y * s->mb_stride;



            error = s->error_status_table[mb_xy];

            if ((error & ER_DC_ERROR) && (error & ER_MV_ERROR))

                continue; // skip damaged



            j++;

            // skip a few to speed things up

            if ((j % skip_amount) != 0)

                continue;



            if (s->cur_pic.f->pict_type == AV_PICTURE_TYPE_I) {

                int *linesize = s->cur_pic.f->linesize;

                uint8_t *mb_ptr      = s->cur_pic.f->data[0] +

                                       mb_x * 16 + mb_y * 16 * linesize[0];

                uint8_t *last_mb_ptr = s->last_pic.f->data[0] +

                                       mb_x * 16 + mb_y * 16 * linesize[0];



                if (s->avctx->codec_id == AV_CODEC_ID_H264) {

                    // FIXME

                } else {

                    ff_thread_await_progress(s->last_pic.tf, mb_y, 0);

                }

                is_intra_likely += s->mecc->sad[0](NULL, last_mb_ptr, mb_ptr,

                                                   linesize[0], 16);

                is_intra_likely -= s->mecc->sad[0](NULL, last_mb_ptr,

                                                   last_mb_ptr + linesize[0] * 16,

                                                   linesize[0], 16);

            } else {

                if (IS_INTRA(s->cur_pic.mb_type[mb_xy]))

                   is_intra_likely++;

                else

                   is_intra_likely--;

            }

        }

    }

    return is_intra_likely > 0;

}
