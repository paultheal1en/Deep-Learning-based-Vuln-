static int pcm_decode_frame(AVCodecContext *avctx,

                            void *data, int *data_size,

                            uint8_t *buf, int buf_size)

{

    PCMDecode *s = avctx->priv_data;

    int n;

    short *samples;

    uint8_t *src;



    samples = data;

    src = buf;



    if(buf_size > AVCODEC_MAX_AUDIO_FRAME_SIZE/2)

        buf_size = AVCODEC_MAX_AUDIO_FRAME_SIZE/2;



    switch(avctx->codec->id) {

    case CODEC_ID_PCM_S32LE:

        decode_to16(4, 1, 0, &src, &samples, buf_size);

        break;

    case CODEC_ID_PCM_S32BE:

        decode_to16(4, 0, 0, &src, &samples, buf_size);

        break;

    case CODEC_ID_PCM_U32LE:

        decode_to16(4, 1, 1, &src, &samples, buf_size);

        break;

    case CODEC_ID_PCM_U32BE:

        decode_to16(4, 0, 1, &src, &samples, buf_size);

        break;

    case CODEC_ID_PCM_S24LE:

        decode_to16(3, 1, 0, &src, &samples, buf_size);

        break;

    case CODEC_ID_PCM_S24BE:

        decode_to16(3, 0, 0, &src, &samples, buf_size);

        break;

    case CODEC_ID_PCM_U24LE:

        decode_to16(3, 1, 1, &src, &samples, buf_size);

        break;

    case CODEC_ID_PCM_U24BE:

        decode_to16(3, 0, 1, &src, &samples, buf_size);

        break;

    case CODEC_ID_PCM_S24DAUD:

        n = buf_size / 3;

        for(;n>0;n--) {

          uint32_t v = src[0] << 16 | src[1] << 8 | src[2];

          v >>= 4; // sync flags are here

          *samples++ = ff_reverse[(v >> 8) & 0xff] +

                       (ff_reverse[v & 0xff] << 8);

          src += 3;

        }

        break;

    case CODEC_ID_PCM_S16LE:

        n = buf_size >> 1;

        for(;n>0;n--) {

            *samples++ = src[0] | (src[1] << 8);

            src += 2;

        }

        break;

    case CODEC_ID_PCM_S16BE:

        n = buf_size >> 1;

        for(;n>0;n--) {

            *samples++ = (src[0] << 8) | src[1];

            src += 2;

        }

        break;

    case CODEC_ID_PCM_U16LE:

        n = buf_size >> 1;

        for(;n>0;n--) {

            *samples++ = (src[0] | (src[1] << 8)) - 0x8000;

            src += 2;

        }

        break;

    case CODEC_ID_PCM_U16BE:

        n = buf_size >> 1;

        for(;n>0;n--) {

            *samples++ = ((src[0] << 8) | src[1]) - 0x8000;

            src += 2;

        }

        break;

    case CODEC_ID_PCM_S8:

        n = buf_size;

        for(;n>0;n--) {

            *samples++ = src[0] << 8;

            src++;

        }

        break;

    case CODEC_ID_PCM_U8:

        n = buf_size;

        for(;n>0;n--) {

            *samples++ = ((int)src[0] - 128) << 8;

            src++;

        }

        break;

    case CODEC_ID_PCM_ALAW:

    case CODEC_ID_PCM_MULAW:

        n = buf_size;

        for(;n>0;n--) {

            *samples++ = s->table[src[0]];

            src++;

        }

        break;

    default:

        return -1;

    }

    *data_size = (uint8_t *)samples - (uint8_t *)data;

    return src - buf;

}
