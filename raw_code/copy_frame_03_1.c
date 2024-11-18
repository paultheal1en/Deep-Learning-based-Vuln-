static inline CopyRet copy_frame(AVCodecContext *avctx,

                                 BC_DTS_PROC_OUT *output,

                                 void *data, int *got_frame)

{

    BC_STATUS ret;

    BC_DTS_STATUS decoder_status = { 0, };

    uint8_t trust_interlaced;

    uint8_t interlaced;



    CHDContext *priv = avctx->priv_data;

    int64_t pkt_pts  = AV_NOPTS_VALUE;

    uint8_t pic_type = 0;



    uint8_t bottom_field = (output->PicInfo.flags & VDEC_FLAG_BOTTOMFIELD) ==

                           VDEC_FLAG_BOTTOMFIELD;

    uint8_t bottom_first = !!(output->PicInfo.flags & VDEC_FLAG_BOTTOM_FIRST);



    int width    = output->PicInfo.width;

    int height   = output->PicInfo.height;

    int bwidth;

    uint8_t *src = output->Ybuff;

    int sStride;

    uint8_t *dst;

    int dStride;



    if (output->PicInfo.timeStamp != 0) {

        OpaqueList *node = opaque_list_pop(priv, output->PicInfo.timeStamp);

        if (node) {

            pkt_pts = node->reordered_opaque;

            pic_type = node->pic_type;

            av_free(node);

        } else {

            /*

             * We will encounter a situation where a timestamp cannot be

             * popped if a second field is being returned. In this case,

             * each field has the same timestamp and the first one will

             * cause it to be popped. To keep subsequent calculations

             * simple, pic_type should be set a FIELD value - doesn't

             * matter which, but I chose BOTTOM.

             */

            pic_type = PICT_BOTTOM_FIELD;

        }

        av_log(avctx, AV_LOG_VERBOSE, "output \"pts\": %"PRIu64"\n",

               output->PicInfo.timeStamp);

        av_log(avctx, AV_LOG_VERBOSE, "output picture type %d\n",

               pic_type);

    }



    ret = DtsGetDriverStatus(priv->dev, &decoder_status);

    if (ret != BC_STS_SUCCESS) {

        av_log(avctx, AV_LOG_ERROR,

               "CrystalHD: GetDriverStatus failed: %u\n", ret);

       return RET_ERROR;

    }



    /*

     * For most content, we can trust the interlaced flag returned

     * by the hardware, but sometimes we can't. These are the

     * conditions under which we can trust the flag:

     *

     * 1) It's not h.264 content

     * 2) The UNKNOWN_SRC flag is not set

     * 3) We know we're expecting a second field

     * 4) The hardware reports this picture and the next picture

     *    have the same picture number.

     *

     * Note that there can still be interlaced content that will

     * fail this check, if the hardware hasn't decoded the next

     * picture or if there is a corruption in the stream. (In either

     * case a 0 will be returned for the next picture number)

     */

    trust_interlaced = avctx->codec->id != AV_CODEC_ID_H264 ||

                       !(output->PicInfo.flags & VDEC_FLAG_UNKNOWN_SRC) ||

                       priv->need_second_field ||

                       (decoder_status.picNumFlags & ~0x40000000) ==

                       output->PicInfo.picture_number;



    /*

     * If we got a false negative for trust_interlaced on the first field,

     * we will realise our mistake here when we see that the picture number is that

     * of the previous picture. We cannot recover the frame and should discard the

     * second field to keep the correct number of output frames.

     */

    if (output->PicInfo.picture_number == priv->last_picture && !priv->need_second_field) {

        av_log(avctx, AV_LOG_WARNING,

               "Incorrectly guessed progressive frame. Discarding second field\n");

        /* Returning without providing a picture. */

        return RET_OK;

    }



    interlaced = (output->PicInfo.flags & VDEC_FLAG_INTERLACED_SRC) &&

                 trust_interlaced;



    if (!trust_interlaced && (decoder_status.picNumFlags & ~0x40000000) == 0) {

        av_log(avctx, AV_LOG_VERBOSE,

               "Next picture number unknown. Assuming progressive frame.\n");

    }



    av_log(avctx, AV_LOG_VERBOSE, "Interlaced state: %d | trust_interlaced %d\n",

           interlaced, trust_interlaced);



    if (priv->pic->data[0] && !priv->need_second_field)

        av_frame_unref(priv->pic);



    priv->need_second_field = interlaced && !priv->need_second_field;



    if (!priv->pic->data[0]) {

        if (ff_get_buffer(avctx, priv->pic, AV_GET_BUFFER_FLAG_REF) < 0)

            return RET_ERROR;

    }



    bwidth = av_image_get_linesize(avctx->pix_fmt, width, 0);

    if (priv->is_70012) {

        int pStride;



        if (width <= 720)

            pStride = 720;

        else if (width <= 1280)

            pStride = 1280;

        else pStride = 1920;

        sStride = av_image_get_linesize(avctx->pix_fmt, pStride, 0);

    } else {

        sStride = bwidth;

    }



    dStride = priv->pic->linesize[0];

    dst     = priv->pic->data[0];



    av_log(priv->avctx, AV_LOG_VERBOSE, "CrystalHD: Copying out frame\n");



    if (interlaced) {

        int dY = 0;

        int sY = 0;



        height /= 2;

        if (bottom_field) {

            av_log(priv->avctx, AV_LOG_VERBOSE, "Interlaced: bottom field\n");

            dY = 1;

        } else {

            av_log(priv->avctx, AV_LOG_VERBOSE, "Interlaced: top field\n");

            dY = 0;

        }



        for (sY = 0; sY < height; dY++, sY++) {

            memcpy(&(dst[dY * dStride]), &(src[sY * sStride]), bwidth);

            dY++;

        }

    } else {

        av_image_copy_plane(dst, dStride, src, sStride, bwidth, height);

    }



    priv->pic->interlaced_frame = interlaced;

    if (interlaced)

        priv->pic->top_field_first = !bottom_first;



    priv->pic->pts = pkt_pts;

#if FF_API_PKT_PTS

FF_DISABLE_DEPRECATION_WARNINGS

    priv->pic->pkt_pts = pkt_pts;

FF_ENABLE_DEPRECATION_WARNINGS

#endif



    if (!priv->need_second_field) {

        *got_frame       = 1;

        if ((ret = av_frame_ref(data, priv->pic)) < 0) {

            return ret;

        }

    }



    /*

     * Two types of PAFF content have been observed. One form causes the

     * hardware to return a field pair and the other individual fields,

     * even though the input is always individual fields. We must skip

     * copying on the next decode() call to maintain pipeline length in

     * the first case.

     */

    if (!interlaced && (output->PicInfo.flags & VDEC_FLAG_UNKNOWN_SRC) &&

        (pic_type == PICT_TOP_FIELD || pic_type == PICT_BOTTOM_FIELD)) {

        av_log(priv->avctx, AV_LOG_VERBOSE, "Fieldpair from two packets.\n");

        return RET_SKIP_NEXT_COPY;

    }



    /*

     * The logic here is purely based on empirical testing with samples.

     * If we need a second field, it could come from a second input packet,

     * or it could come from the same field-pair input packet at the current

     * field. In the first case, we should return and wait for the next time

     * round to get the second field, while in the second case, we should

     * ask the decoder for it immediately.

     *

     * Testing has shown that we are dealing with the fieldpair -> two fields

     * case if the VDEC_FLAG_UNKNOWN_SRC is not set or if the input picture

     * type was PICT_FRAME (in this second case, the flag might still be set)

     */

    return priv->need_second_field &&

           (!(output->PicInfo.flags & VDEC_FLAG_UNKNOWN_SRC) ||

            pic_type == PICT_FRAME) ?

           RET_COPY_NEXT_FIELD : RET_OK;

}
