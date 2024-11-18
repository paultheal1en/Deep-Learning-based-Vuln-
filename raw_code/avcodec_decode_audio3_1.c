int attribute_align_arg avcodec_decode_audio3(AVCodecContext *avctx, int16_t *samples,

                         int *frame_size_ptr,

                         AVPacket *avpkt)

{

    AVFrame frame;

    int ret, got_frame = 0;



    if (avctx->get_buffer != avcodec_default_get_buffer) {

        av_log(avctx, AV_LOG_ERROR, "A custom get_buffer() cannot be used with "

               "avcodec_decode_audio3()\n");

        return AVERROR(EINVAL);

    }



    ret = avcodec_decode_audio4(avctx, &frame, &got_frame, avpkt);



    if (ret >= 0 && got_frame) {

        int ch, plane_size;

        int planar = av_sample_fmt_is_planar(avctx->sample_fmt);

        int data_size = av_samples_get_buffer_size(&plane_size, avctx->channels,

                                                   frame.nb_samples,

                                                   avctx->sample_fmt, 1);

        if (*frame_size_ptr < data_size) {

            av_log(avctx, AV_LOG_ERROR, "output buffer size is too small for "

                   "the current frame (%d < %d)\n", *frame_size_ptr, data_size);

            return AVERROR(EINVAL);

        }



        memcpy(samples, frame.extended_data[0], plane_size);



        if (planar && avctx->channels > 1) {

            uint8_t *out = ((uint8_t *)samples) + plane_size;

            for (ch = 1; ch < avctx->channels; ch++) {

                memcpy(out, frame.extended_data[ch], plane_size);

                out += plane_size;

            }

        }

        *frame_size_ptr = data_size;

    } else {

        *frame_size_ptr = 0;

    }

    return ret;

}
