static int pcm_encode_frame(AVCodecContext *avctx, AVPacket *avpkt,

                            const AVFrame *frame, int *got_packet_ptr)

{

    int n, c, sample_size, v, ret;

    const short *samples;

    unsigned char *dst;

    const uint8_t *samples_uint8_t;

    const int16_t *samples_int16_t;

    const int32_t *samples_int32_t;

    const int64_t *samples_int64_t;

    const uint16_t *samples_uint16_t;

    const uint32_t *samples_uint32_t;



    sample_size = av_get_bits_per_sample(avctx->codec->id) / 8;

    n           = frame->nb_samples * avctx->channels;

    samples     = (const short *)frame->data[0];



    if ((ret = ff_alloc_packet2(avctx, avpkt, n * sample_size)))

        return ret;

    dst = avpkt->data;



    switch (avctx->codec->id) {

    case AV_CODEC_ID_PCM_U32LE:

        ENCODE(uint32_t, le32, samples, dst, n, 0, 0x80000000)

        break;

    case AV_CODEC_ID_PCM_U32BE:

        ENCODE(uint32_t, be32, samples, dst, n, 0, 0x80000000)

        break;

    case AV_CODEC_ID_PCM_S24LE:

        ENCODE(int32_t, le24, samples, dst, n, 8, 0)

        break;

    case AV_CODEC_ID_PCM_S24LE_PLANAR:

        ENCODE_PLANAR(int32_t, le24, dst, n, 8, 0)

        break;

    case AV_CODEC_ID_PCM_S24BE:

        ENCODE(int32_t, be24, samples, dst, n, 8, 0)

        break;

    case AV_CODEC_ID_PCM_U24LE:

        ENCODE(uint32_t, le24, samples, dst, n, 8, 0x800000)

        break;

    case AV_CODEC_ID_PCM_U24BE:

        ENCODE(uint32_t, be24, samples, dst, n, 8, 0x800000)

        break;

    case AV_CODEC_ID_PCM_S24DAUD:

        for (; n > 0; n--) {

            uint32_t tmp = ff_reverse[(*samples >> 8) & 0xff] +

                           (ff_reverse[*samples & 0xff] << 8);

            tmp <<= 4; // sync flags would go here

            bytestream_put_be24(&dst, tmp);

            samples++;

        }

        break;

    case AV_CODEC_ID_PCM_U16LE:

        ENCODE(uint16_t, le16, samples, dst, n, 0, 0x8000)

        break;

    case AV_CODEC_ID_PCM_U16BE:

        ENCODE(uint16_t, be16, samples, dst, n, 0, 0x8000)

        break;

    case AV_CODEC_ID_PCM_S8:

        ENCODE(uint8_t, byte, samples, dst, n, 0, -128)

        break;

    case AV_CODEC_ID_PCM_S8_PLANAR:

        ENCODE_PLANAR(uint8_t, byte, dst, n, 0, -128)

        break;

#if HAVE_BIGENDIAN

    case AV_CODEC_ID_PCM_F64LE:

        ENCODE(int64_t, le64, samples, dst, n, 0, 0)

        break;

    case AV_CODEC_ID_PCM_S32LE:

    case AV_CODEC_ID_PCM_F32LE:

        ENCODE(int32_t, le32, samples, dst, n, 0, 0)

        break;

    case AV_CODEC_ID_PCM_S32LE_PLANAR:

        ENCODE_PLANAR(int32_t, le32, dst, n, 0, 0)

        break;

    case AV_CODEC_ID_PCM_S16LE:

        ENCODE(int16_t, le16, samples, dst, n, 0, 0)

        break;

    case AV_CODEC_ID_PCM_S16LE_PLANAR:

        ENCODE_PLANAR(int16_t, le16, dst, n, 0, 0)

        break;

    case AV_CODEC_ID_PCM_F64BE:

    case AV_CODEC_ID_PCM_F32BE:

    case AV_CODEC_ID_PCM_S32BE:

    case AV_CODEC_ID_PCM_S16BE:

#else

    case AV_CODEC_ID_PCM_F64BE:

        ENCODE(int64_t, be64, samples, dst, n, 0, 0)

        break;

    case AV_CODEC_ID_PCM_F32BE:

    case AV_CODEC_ID_PCM_S32BE:

        ENCODE(int32_t, be32, samples, dst, n, 0, 0)

        break;

    case AV_CODEC_ID_PCM_S16BE:

        ENCODE(int16_t, be16, samples, dst, n, 0, 0)

        break;

    case AV_CODEC_ID_PCM_S16BE_PLANAR:

        ENCODE_PLANAR(int16_t, be16, dst, n, 0, 0)

        break;

    case AV_CODEC_ID_PCM_F64LE:

    case AV_CODEC_ID_PCM_F32LE:

    case AV_CODEC_ID_PCM_S32LE:

    case AV_CODEC_ID_PCM_S16LE:

#endif /* HAVE_BIGENDIAN */

    case AV_CODEC_ID_PCM_U8:

        memcpy(dst, samples, n * sample_size);

        break;

#if HAVE_BIGENDIAN

    case AV_CODEC_ID_PCM_S16BE_PLANAR:

#else

    case AV_CODEC_ID_PCM_S16LE_PLANAR:

    case AV_CODEC_ID_PCM_S32LE_PLANAR:

#endif /* HAVE_BIGENDIAN */

        n /= avctx->channels;

        for (c = 0; c < avctx->channels; c++) {

            const uint8_t *src = frame->extended_data[c];

            bytestream_put_buffer(&dst, src, n * sample_size);

        }

        break;

    case AV_CODEC_ID_PCM_ALAW:

        for (; n > 0; n--) {

            v      = *samples++;

            *dst++ = linear_to_alaw[(v + 32768) >> 2];

        }

        break;

    case AV_CODEC_ID_PCM_MULAW:

        for (; n > 0; n--) {

            v      = *samples++;

            *dst++ = linear_to_ulaw[(v + 32768) >> 2];

        }

        break;

    default:

        return -1;

    }



    *got_packet_ptr = 1;

    return 0;

}
