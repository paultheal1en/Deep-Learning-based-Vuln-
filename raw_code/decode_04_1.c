static int decode(AVCodecContext *avctx, void *data, int *data_size, AVPacket *avpkt)

{

    BC_STATUS ret;

    BC_DTS_STATUS decoder_status;

    CopyRet rec_ret;

    CHDContext *priv   = avctx->priv_data;

    HANDLE dev         = priv->dev;

    int len            = avpkt->size;

    uint8_t pic_type   = 0;



    av_log(avctx, AV_LOG_VERBOSE, "CrystalHD: decode_frame\n");



    if (len) {

        int32_t tx_free = (int32_t)DtsTxFreeSize(dev);



        if (priv->parser) {

            uint8_t *pout;

            int psize = len;

            H264Context *h = priv->parser->priv_data;



            while (psize)

                ret = av_parser_parse2(priv->parser, avctx, &pout, &psize,

                                       avpkt->data, len, avctx->pkt->pts,

                                       avctx->pkt->dts, len - psize);

            av_log(avctx, AV_LOG_VERBOSE,

                   "CrystalHD: parser picture type %d\n",

                   h->s.picture_structure);

            pic_type = h->s.picture_structure;

        }



        if (len < tx_free - 1024) {

            /*

             * Despite being notionally opaque, either libcrystalhd or

             * the hardware itself will mangle pts values that are too

             * small or too large. The docs claim it should be in units

             * of 100ns. Given that we're nominally dealing with a black

             * box on both sides, any transform we do has no guarantee of

             * avoiding mangling so we need to build a mapping to values

             * we know will not be mangled.

             */

            uint64_t pts = opaque_list_push(priv, avctx->pkt->pts, pic_type);

            if (!pts) {

                return AVERROR(ENOMEM);

            }

            av_log(priv->avctx, AV_LOG_VERBOSE,

                   "input \"pts\": %"PRIu64"\n", pts);

            ret = DtsProcInput(dev, avpkt->data, len, pts, 0);

            if (ret == BC_STS_BUSY) {

                av_log(avctx, AV_LOG_WARNING,

                       "CrystalHD: ProcInput returned busy\n");

                usleep(BASE_WAIT);

                return AVERROR(EBUSY);

            } else if (ret != BC_STS_SUCCESS) {

                av_log(avctx, AV_LOG_ERROR,

                       "CrystalHD: ProcInput failed: %u\n", ret);

                return -1;

            }

            avctx->has_b_frames++;

        } else {

            av_log(avctx, AV_LOG_WARNING, "CrystalHD: Input buffer full\n");

            len = 0; // We didn't consume any bytes.

        }

    } else {

        av_log(avctx, AV_LOG_INFO, "CrystalHD: No more input data\n");

    }



    if (priv->skip_next_output) {

        av_log(avctx, AV_LOG_VERBOSE, "CrystalHD: Skipping next output.\n");

        priv->skip_next_output = 0;

        avctx->has_b_frames--;

        return len;

    }



    ret = DtsGetDriverStatus(dev, &decoder_status);

    if (ret != BC_STS_SUCCESS) {

        av_log(avctx, AV_LOG_ERROR, "CrystalHD: GetDriverStatus failed\n");

        return -1;

    }



    /*

     * No frames ready. Don't try to extract.

     *

     * Empirical testing shows that ReadyListCount can be a damn lie,

     * and ProcOut still fails when count > 0. The same testing showed

     * that two more iterations were needed before ProcOutput would

     * succeed.

     */

    if (priv->output_ready < 2) {

        if (decoder_status.ReadyListCount != 0)

            priv->output_ready++;

        usleep(BASE_WAIT);

        av_log(avctx, AV_LOG_INFO, "CrystalHD: Filling pipeline.\n");

        return len;

    } else if (decoder_status.ReadyListCount == 0) {

        /*

         * After the pipeline is established, if we encounter a lack of frames

         * that probably means we're not giving the hardware enough time to

         * decode them, so start increasing the wait time at the end of a

         * decode call.

         */

        usleep(BASE_WAIT);

        priv->decode_wait += WAIT_UNIT;

        av_log(avctx, AV_LOG_INFO, "CrystalHD: No frames ready. Returning\n");

        return len;

    }



    do {

        rec_ret = receive_frame(avctx, data, data_size, 0);

        if (rec_ret == RET_OK && *data_size == 0) {

            /*

             * This case is for when the encoded fields are stored

             * separately and we get a separate avpkt for each one. To keep

             * the pipeline stable, we should return nothing and wait for

             * the next time round to grab the second field.

             * H.264 PAFF is an example of this.

             */

            av_log(avctx, AV_LOG_VERBOSE, "Returning after first field.\n");

            avctx->has_b_frames--;

        } else if (rec_ret == RET_COPY_NEXT_FIELD) {

            /*

             * This case is for when the encoded fields are stored in a

             * single avpkt but the hardware returns then separately. Unless

             * we grab the second field before returning, we'll slip another

             * frame in the pipeline and if that happens a lot, we're sunk.

             * So we have to get that second field now.

             * Interlaced mpeg2 and vc1 are examples of this.

             */

            av_log(avctx, AV_LOG_VERBOSE, "Trying to get second field.\n");

            while (1) {

                usleep(priv->decode_wait);

                ret = DtsGetDriverStatus(dev, &decoder_status);

                if (ret == BC_STS_SUCCESS &&

                    decoder_status.ReadyListCount > 0) {

                    rec_ret = receive_frame(avctx, data, data_size, 1);

                    if ((rec_ret == RET_OK && *data_size > 0) ||

                        rec_ret == RET_ERROR)

                        break;

                }

            }

            av_log(avctx, AV_LOG_VERBOSE, "CrystalHD: Got second field.\n");

        } else if (rec_ret == RET_SKIP_NEXT_COPY) {

            /*

             * Two input packets got turned into a field pair. Gawd.

             */

            av_log(avctx, AV_LOG_VERBOSE,

                   "Don't output on next decode call.\n");

            priv->skip_next_output = 1;

        }

        /*

         * If rec_ret == RET_COPY_AGAIN, that means that either we just handled

         * a FMT_CHANGE event and need to go around again for the actual frame,

         * we got a busy status and need to try again, or we're dealing with

         * packed b-frames, where the hardware strangely returns the packed

         * p-frame twice. We choose to keep the second copy as it carries the

         * valid pts.

         */

    } while (rec_ret == RET_COPY_AGAIN);

    usleep(priv->decode_wait);

    return len;

}
