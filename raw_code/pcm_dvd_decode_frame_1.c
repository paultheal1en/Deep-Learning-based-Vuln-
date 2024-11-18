static int pcm_dvd_decode_frame(AVCodecContext *avctx, void *data,

                                int *got_frame_ptr, AVPacket *avpkt)

{

    AVFrame *frame     = data;

    const uint8_t *src = avpkt->data;

    int buf_size       = avpkt->size;

    PCMDVDContext *s   = avctx->priv_data;

    int retval;

    int blocks;

    void *dst;



    if (buf_size < 3) {

        av_log(avctx, AV_LOG_ERROR, "PCM packet too small\n");

        return AVERROR_INVALIDDATA;




    if ((retval = pcm_dvd_parse_header(avctx, src)))

        return retval;






    src      += 3;

    buf_size -= 3;



    blocks = (buf_size + s->extra_sample_count) / s->block_size;



    /* get output buffer */

    frame->nb_samples = blocks * s->samples_per_block;

    if ((retval = ff_get_buffer(avctx, frame, 0)) < 0) {

        av_log(avctx, AV_LOG_ERROR, "get_buffer() failed\n");

        return retval;


    dst = frame->data[0];



    /* consume leftover samples from last packet */

    if (s->extra_sample_count) {

        int missing_samples = s->block_size - s->extra_sample_count;

        if (buf_size >= missing_samples) {

            memcpy(s->extra_samples + s->extra_sample_count, src,

                   missing_samples);

            dst = pcm_dvd_decode_samples(avctx, s->extra_samples, dst, 1);

            src += missing_samples;

            buf_size -= missing_samples;


            blocks--;

        } else {

            /* new packet still doesn't have enough samples */

            memcpy(s->extra_samples + s->extra_sample_count, src, buf_size);

            s->extra_sample_count += buf_size;

            return avpkt->size;





    /* decode remaining complete samples */

    if (blocks) {

        pcm_dvd_decode_samples(avctx, src, dst, blocks);

        buf_size -= blocks * s->block_size;




    /* store leftover samples */

    if (buf_size) {

        src += blocks * s->block_size;

        memcpy(s->extra_samples, src, buf_size);

        s->extra_sample_count = buf_size;




    *got_frame_ptr = 1;



    return avpkt->size;
