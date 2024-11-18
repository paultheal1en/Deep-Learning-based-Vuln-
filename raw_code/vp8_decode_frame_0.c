static int vp8_decode_frame(AVCodecContext *avctx, void *data, int *data_size,

                            AVPacket *avpkt)

{

    VP8Context *s = avctx->priv_data;

    int ret, mb_x, mb_y, i, y, referenced;

    enum AVDiscard skip_thresh;

    AVFrame *curframe;



    if ((ret = decode_frame_header(s, avpkt->data, avpkt->size)) < 0)

        return ret;



    referenced = s->update_last || s->update_golden == VP56_FRAME_CURRENT

                                || s->update_altref == VP56_FRAME_CURRENT;



    skip_thresh = !referenced ? AVDISCARD_NONREF :

                    !s->keyframe ? AVDISCARD_NONKEY : AVDISCARD_ALL;



    if (avctx->skip_frame >= skip_thresh) {

        s->invisible = 1;

        goto skip_decode;

    }



    for (i = 0; i < 4; i++)

        if (&s->frames[i] != s->framep[VP56_FRAME_PREVIOUS] &&

            &s->frames[i] != s->framep[VP56_FRAME_GOLDEN] &&

            &s->frames[i] != s->framep[VP56_FRAME_GOLDEN2]) {

            curframe = s->framep[VP56_FRAME_CURRENT] = &s->frames[i];

            break;

        }

    if (curframe->data[0])

        avctx->release_buffer(avctx, curframe);



    curframe->key_frame = s->keyframe;

    curframe->pict_type = s->keyframe ? FF_I_TYPE : FF_P_TYPE;

    curframe->reference = referenced ? 3 : 0;

    if ((ret = avctx->get_buffer(avctx, curframe))) {

        av_log(avctx, AV_LOG_ERROR, "get_buffer() failed!\n");

        return ret;

    }



    // Given that arithmetic probabilities are updated every frame, it's quite likely

    // that the values we have on a random interframe are complete junk if we didn't

    // start decode on a keyframe. So just don't display anything rather than junk.

    if (!s->keyframe && (!s->framep[VP56_FRAME_PREVIOUS] ||

                         !s->framep[VP56_FRAME_GOLDEN] ||

                         !s->framep[VP56_FRAME_GOLDEN2])) {

        av_log(avctx, AV_LOG_WARNING, "Discarding interframe without a prior keyframe!\n");

        return AVERROR_INVALIDDATA;

    }



    s->linesize   = curframe->linesize[0];

    s->uvlinesize = curframe->linesize[1];



    if (!s->edge_emu_buffer)

        s->edge_emu_buffer = av_malloc(21*s->linesize);



    memset(s->top_nnz, 0, s->mb_width*sizeof(*s->top_nnz));



    // top edge of 127 for intra prediction

    if (!(avctx->flags & CODEC_FLAG_EMU_EDGE)) {

        memset(curframe->data[0] - s->linesize  -1, 127, s->linesize  +1);

        memset(curframe->data[1] - s->uvlinesize-1, 127, s->uvlinesize+1);

        memset(curframe->data[2] - s->uvlinesize-1, 127, s->uvlinesize+1);

    }



    for (mb_y = 0; mb_y < s->mb_height; mb_y++) {

        VP56RangeCoder *c = &s->coeff_partition[mb_y & (s->num_coeff_partitions-1)];

        VP8Macroblock *mb = s->macroblocks + mb_y*s->mb_stride;

        uint8_t *intra4x4 = s->intra4x4_pred_mode + 4*mb_y*s->b4_stride;

        uint8_t *dst[3] = {

            curframe->data[0] + 16*mb_y*s->linesize,

            curframe->data[1] +  8*mb_y*s->uvlinesize,

            curframe->data[2] +  8*mb_y*s->uvlinesize

        };



        memset(s->left_nnz, 0, sizeof(s->left_nnz));



        // left edge of 129 for intra prediction

        if (!(avctx->flags & CODEC_FLAG_EMU_EDGE))

            for (i = 0; i < 3; i++)

                for (y = 0; y < 16>>!!i; y++)

                    dst[i][y*curframe->linesize[i]-1] = 129;



        for (mb_x = 0; mb_x < s->mb_width; mb_x++) {

            decode_mb_mode(s, mb, mb_x, mb_y, intra4x4 + 4*mb_x);



            if (!mb->skip)

                decode_mb_coeffs(s, c, mb, s->top_nnz[mb_x], s->left_nnz);

            else {

                AV_ZERO128(s->non_zero_count_cache);    // luma

                AV_ZERO64(s->non_zero_count_cache[4]);  // chroma

            }



            if (mb->mode <= MODE_I4x4) {

                intra_predict(s, dst, mb, intra4x4 + 4*mb_x, mb_x, mb_y);

                memset(mb->bmv, 0, sizeof(mb->bmv));

            } else {

                inter_predict(s, dst, mb, mb_x, mb_y);

            }



            if (!mb->skip) {

                idct_mb(s, dst[0], dst[1], dst[2], mb);

            } else {

                AV_ZERO64(s->left_nnz);

                AV_WN64(s->top_nnz[mb_x], 0);   // array of 9, so unaligned



                // Reset DC block predictors if they would exist if the mb had coefficients

                if (mb->mode != MODE_I4x4 && mb->mode != VP8_MVMODE_SPLIT) {

                    s->left_nnz[8]      = 0;

                    s->top_nnz[mb_x][8] = 0;

                }

            }



            dst[0] += 16;

            dst[1] += 8;

            dst[2] += 8;

            mb++;

        }

        if (mb_y && s->filter.level && avctx->skip_loop_filter < skip_thresh) {

            if (s->filter.simple)

                filter_mb_row_simple(s, mb_y-1);

            else

                filter_mb_row(s, mb_y-1);

        }

    }

    if (s->filter.level && avctx->skip_loop_filter < skip_thresh) {

        if (s->filter.simple)

            filter_mb_row_simple(s, mb_y-1);

        else

            filter_mb_row(s, mb_y-1);

    }



skip_decode:

    // if future frames don't use the updated probabilities,

    // reset them to the values we saved

    if (!s->update_probabilities)

        s->prob[0] = s->prob[1];



    // check if golden and altref are swapped

    if (s->update_altref == VP56_FRAME_GOLDEN &&

        s->update_golden == VP56_FRAME_GOLDEN2)

        FFSWAP(AVFrame *, s->framep[VP56_FRAME_GOLDEN], s->framep[VP56_FRAME_GOLDEN2]);

    else {

        if (s->update_altref != VP56_FRAME_NONE)

            s->framep[VP56_FRAME_GOLDEN2] = s->framep[s->update_altref];



        if (s->update_golden != VP56_FRAME_NONE)

            s->framep[VP56_FRAME_GOLDEN] = s->framep[s->update_golden];

    }



    if (s->update_last) // move cur->prev

        s->framep[VP56_FRAME_PREVIOUS] = s->framep[VP56_FRAME_CURRENT];



    // release no longer referenced frames

    for (i = 0; i < 4; i++)

        if (s->frames[i].data[0] &&

            &s->frames[i] != s->framep[VP56_FRAME_CURRENT] &&

            &s->frames[i] != s->framep[VP56_FRAME_PREVIOUS] &&

            &s->frames[i] != s->framep[VP56_FRAME_GOLDEN] &&

            &s->frames[i] != s->framep[VP56_FRAME_GOLDEN2])

            avctx->release_buffer(avctx, &s->frames[i]);



    if (!s->invisible) {

        *(AVFrame*)data = *s->framep[VP56_FRAME_CURRENT];

        *data_size = sizeof(AVFrame);

    }



    return avpkt->size;

}
