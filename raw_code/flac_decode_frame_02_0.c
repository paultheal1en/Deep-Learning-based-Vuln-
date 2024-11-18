static int flac_decode_frame(AVCodecContext *avctx, void *data,

                             int *got_frame_ptr, AVPacket *avpkt)

{

    AVFrame *frame     = data;

    ThreadFrame tframe = { .f = data };

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    FLACContext *s = avctx->priv_data;

    int bytes_read = 0;

    int ret;



    *got_frame_ptr = 0;



    if (s->max_framesize == 0) {

        s->max_framesize =

            ff_flac_get_max_frame_size(s->max_blocksize ? s->max_blocksize : FLAC_MAX_BLOCKSIZE,

                                       FLAC_MAX_CHANNELS, 32);

    }



    if (buf_size > 5 && !memcmp(buf, "\177FLAC", 5)) {

        av_log(s->avctx, AV_LOG_DEBUG, "skiping flac header packet 1\n");

        return buf_size;

    }



    if (buf_size > 0 && (*buf & 0x7F) == FLAC_METADATA_TYPE_VORBIS_COMMENT) {

        av_log(s->avctx, AV_LOG_DEBUG, "skiping vorbis comment\n");

        return buf_size;

    }



    /* check that there is at least the smallest decodable amount of data.

       this amount corresponds to the smallest valid FLAC frame possible.

       FF F8 69 02 00 00 9A 00 00 34 46 */

    if (buf_size < FLAC_MIN_FRAME_SIZE)

        return buf_size;



    /* check for inline header */

    if (AV_RB32(buf) == MKBETAG('f','L','a','C')) {

        if (!s->got_streaminfo && (ret = parse_streaminfo(s, buf, buf_size))) {

            av_log(s->avctx, AV_LOG_ERROR, "invalid header\n");

            return ret;

        }

        return get_metadata_size(buf, buf_size);

    }



    /* decode frame */

    if ((ret = init_get_bits8(&s->gb, buf, buf_size)) < 0)

        return ret;

    if ((ret = decode_frame(s)) < 0) {

        av_log(s->avctx, AV_LOG_ERROR, "decode_frame() failed\n");

        return ret;

    }

    bytes_read = get_bits_count(&s->gb)/8;



    if ((s->avctx->err_recognition & AV_EF_CRCCHECK) &&

        av_crc(av_crc_get_table(AV_CRC_16_ANSI),

               0, buf, bytes_read)) {

        av_log(s->avctx, AV_LOG_ERROR, "CRC error at PTS %"PRId64"\n", avpkt->pts);

        if (s->avctx->err_recognition & AV_EF_EXPLODE)

            return AVERROR_INVALIDDATA;

    }



    /* get output buffer */

    frame->nb_samples = s->blocksize;

    if ((ret = ff_thread_get_buffer(avctx, &tframe, 0)) < 0)

        return ret;



    s->dsp.decorrelate[s->ch_mode](frame->data, s->decoded, s->channels,

                                   s->blocksize, s->sample_shift);



    if (bytes_read > buf_size) {

        av_log(s->avctx, AV_LOG_ERROR, "overread: %d\n", bytes_read - buf_size);

        return AVERROR_INVALIDDATA;

    }

    if (bytes_read < buf_size) {

        av_log(s->avctx, AV_LOG_DEBUG, "underread: %d orig size: %d\n",

               buf_size - bytes_read, buf_size);

    }



    *got_frame_ptr = 1;



    return bytes_read;

}
