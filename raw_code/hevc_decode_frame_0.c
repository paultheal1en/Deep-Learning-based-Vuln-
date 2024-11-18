static int hevc_decode_frame(AVCodecContext *avctx, void *data, int *got_output,

                             AVPacket *avpkt)

{

    int ret;

    HEVCContext *s = avctx->priv_data;



    if (!avpkt->size) {

        ret = ff_hevc_output_frame(s, data, 1);

        if (ret < 0)

            return ret;



        *got_output = ret;

        return 0;

    }



    s->ref = NULL;

    ret    = decode_nal_units(s, avpkt->data, avpkt->size);

    if (ret < 0)

        return ret;



    /* verify the SEI checksum */

    if (avctx->err_recognition & AV_EF_CRCCHECK && s->is_decoded &&

        avctx->err_recognition & AV_EF_EXPLODE &&

        s->is_md5) {

        ret = verify_md5(s, s->ref->frame);

        if (ret < 0) {

            ff_hevc_unref_frame(s, s->ref, ~0);

            return ret;

        }

    }

    s->is_md5 = 0;



    if (s->is_decoded) {

        av_log(avctx, AV_LOG_DEBUG, "Decoded frame with POC %d.\n", s->poc);

        s->is_decoded = 0;

    }



    if (s->output_frame->buf[0]) {

        av_frame_move_ref(data, s->output_frame);

        *got_output = 1;

    }



    return avpkt->size;

}
