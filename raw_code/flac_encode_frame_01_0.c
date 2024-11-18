static int flac_encode_frame(AVCodecContext *avctx, uint8_t *frame,

                             int buf_size, void *data)

{

    FlacEncodeContext *s;

    const int16_t *samples = data;

    int frame_bytes, out_bytes;



    s = avctx->priv_data;



    /* when the last block is reached, update the header in extradata */

    if (!data) {

        s->max_framesize = s->max_encoded_framesize;

        av_md5_final(s->md5ctx, s->md5sum);

        write_streaminfo(s, avctx->extradata);

        return 0;

    }



    /* change max_framesize for small final frame */

    if (avctx->frame_size < s->frame.blocksize) {

        s->max_framesize = ff_flac_get_max_frame_size(avctx->frame_size,

                                                      s->channels, 16);

    }



    init_frame(s);



    copy_samples(s, samples);



    channel_decorrelation(s);



    frame_bytes = encode_frame(s);

    if (buf_size < frame_bytes) {

        av_log(avctx, AV_LOG_ERROR, "output buffer too small\n");

        return 0;

    }

    out_bytes = write_frame(s, frame, buf_size);



    /* fallback to verbatim mode if the compressed frame is larger than it

       would be if encoded uncompressed. */

    if (out_bytes > s->max_framesize) {

        s->frame.verbatim_only = 1;

        frame_bytes = encode_frame(s);

        if (buf_size < frame_bytes) {

            av_log(avctx, AV_LOG_ERROR, "output buffer too small\n");

            return 0;

        }

        out_bytes = write_frame(s, frame, buf_size);

    }



    s->frame_count++;

    avctx->coded_frame->pts = s->sample_count;

    s->sample_count += avctx->frame_size;

    update_md5_sum(s, samples);

    if (out_bytes > s->max_encoded_framesize)

        s->max_encoded_framesize = out_bytes;

    if (out_bytes < s->min_framesize)

        s->min_framesize = out_bytes;



    return out_bytes;

}
