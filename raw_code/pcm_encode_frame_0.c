static int pcm_encode_frame(AVCodecContext *avctx,

                            unsigned char *frame, int buf_size, void *data)

{

    int n, sample_size, v;

    const short *samples;

    unsigned char *dst;

    const uint8_t *srcu8;

    const int16_t *samples_int16_t;

    const int32_t *samples_int32_t;

    const int64_t *samples_int64_t;

    const uint16_t *samples_uint16_t;

    const uint32_t *samples_uint32_t;



    sample_size = av_get_bits_per_sample(avctx->codec->id)/8;

    n = buf_size / sample_size;

    samples = data;

    dst = frame;



    if (avctx->sample_fmt!=avctx->codec->sample_fmts[0]) {

        av_log(avctx, AV_LOG_ERROR, "invalid sample_fmt\n");

        return -1;

    }



    switch(avctx->codec->id) {

    case CODEC_ID_PCM_U32LE:

        ENCODE(uint32_t, le32, samples, dst, n, 0, 0x80000000)

        break;

    case CODEC_ID_PCM_U32BE:

        ENCODE(uint32_t, be32, samples, dst, n, 0, 0x80000000)

        break;

    case CODEC_ID_PCM_S24LE:

        ENCODE(int32_t, le24, samples, dst, n, 8, 0)

        break;

    case CODEC_ID_PCM_S24BE:

        ENCODE(int32_t, be24, samples, dst, n, 8, 0)

        break;

    case CODEC_ID_PCM_U24LE:

        ENCODE(uint32_t, le24, samples, dst, n, 8, 0x800000)

        break;

    case CODEC_ID_PCM_U24BE:

        ENCODE(uint32_t, be24, samples, dst, n, 8, 0x800000)

        break;

    case CODEC_ID_PCM_S24DAUD:

        for(;n>0;n--) {

            uint32_t tmp = av_reverse[(*samples >> 8) & 0xff] +

                           (av_reverse[*samples & 0xff] << 8);

            tmp <<= 4; // sync flags would go here

            bytestream_put_be24(&dst, tmp);

            samples++;

        }

        break;

    case CODEC_ID_PCM_U16LE:

        ENCODE(uint16_t, le16, samples, dst, n, 0, 0x8000)

        break;

    case CODEC_ID_PCM_U16BE:

        ENCODE(uint16_t, be16, samples, dst, n, 0, 0x8000)

        break;

    case CODEC_ID_PCM_S8:

        srcu8= data;

        for(;n>0;n--) {

            v = *srcu8++;

            *dst++ = v - 128;

        }

        break;

#if HAVE_BIGENDIAN

    case CODEC_ID_PCM_F64LE:

        ENCODE(int64_t, le64, samples, dst, n, 0, 0)

        break;

    case CODEC_ID_PCM_S32LE:

    case CODEC_ID_PCM_F32LE:

        ENCODE(int32_t, le32, samples, dst, n, 0, 0)

        break;

    case CODEC_ID_PCM_S16LE:

        ENCODE(int16_t, le16, samples, dst, n, 0, 0)

        break;

    case CODEC_ID_PCM_F64BE:

    case CODEC_ID_PCM_F32BE:

    case CODEC_ID_PCM_S32BE:

    case CODEC_ID_PCM_S16BE:

#else

    case CODEC_ID_PCM_F64BE:

        ENCODE(int64_t, be64, samples, dst, n, 0, 0)

        break;

    case CODEC_ID_PCM_F32BE:

    case CODEC_ID_PCM_S32BE:

        ENCODE(int32_t, be32, samples, dst, n, 0, 0)

        break;

    case CODEC_ID_PCM_S16BE:

        ENCODE(int16_t, be16, samples, dst, n, 0, 0)

        break;

    case CODEC_ID_PCM_F64LE:

    case CODEC_ID_PCM_F32LE:

    case CODEC_ID_PCM_S32LE:

    case CODEC_ID_PCM_S16LE:

#endif /* HAVE_BIGENDIAN */

    case CODEC_ID_PCM_U8:

        memcpy(dst, samples, n*sample_size);

        dst += n*sample_size;

        break;

    case CODEC_ID_PCM_ZORK:

        for(;n>0;n--) {

            v= *samples++ >> 8;

            if(v<0)   v = -v;

            else      v+= 128;

            *dst++ = v;

        }

        break;

    case CODEC_ID_PCM_ALAW:

        for(;n>0;n--) {

            v = *samples++;

            *dst++ = linear_to_alaw[(v + 32768) >> 2];

        }

        break;

    case CODEC_ID_PCM_MULAW:

        for(;n>0;n--) {

            v = *samples++;

            *dst++ = linear_to_ulaw[(v + 32768) >> 2];

        }

        break;

    default:

        return -1;

    }

    //avctx->frame_size = (dst - frame) / (sample_size * avctx->channels);



    return dst - frame;

}
