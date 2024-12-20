static int dpcm_decode_frame(AVCodecContext *avctx,
                             void *data, int *data_size,
                             uint8_t *buf, int buf_size)
{
    DPCMContext *s = avctx->priv_data;
    int in, out = 0;
    int predictor[2];
    int channel_number = 0;
    short *output_samples = data;
    int shift[2];
    unsigned char byte;
    short diff;
    if (!buf_size)
        return 0;
    switch(avctx->codec->id) {
    case CODEC_ID_ROQ_DPCM:
        if (s->channels == 1)
            predictor[0] = AV_RL16(&buf[6]);
        else {
            predictor[0] = buf[7] << 8;
            predictor[1] = buf[6] << 8;
        }
        SE_16BIT(predictor[0]);
        SE_16BIT(predictor[1]);
        /* decode the samples */
        for (in = 8, out = 0; in < buf_size; in++, out++) {
            predictor[channel_number] += s->roq_square_array[buf[in]];
            predictor[channel_number] = av_clip_int16(predictor[channel_number]);
            output_samples[out] = predictor[channel_number];
            /* toggle channel */
            channel_number ^= s->channels - 1;
        }
        break;
    case CODEC_ID_INTERPLAY_DPCM:
        in = 6;  /* skip over the stream mask and stream length */
        predictor[0] = AV_RL16(&buf[in]);
        in += 2;
        SE_16BIT(predictor[0])
        output_samples[out++] = predictor[0];
        if (s->channels == 2) {
            predictor[1] = AV_RL16(&buf[in]);
            in += 2;
            SE_16BIT(predictor[1])
            output_samples[out++] = predictor[1];
        }
        while (in < buf_size) {
            predictor[channel_number] += interplay_delta_table[buf[in++]];
            predictor[channel_number] = av_clip_int16(predictor[channel_number]);
            output_samples[out++] = predictor[channel_number];
            /* toggle channel */
            channel_number ^= s->channels - 1;
        }
        break;
    case CODEC_ID_XAN_DPCM:
        in = 0;
        shift[0] = shift[1] = 4;
        predictor[0] = AV_RL16(&buf[in]);
        in += 2;
        SE_16BIT(predictor[0]);
        if (s->channels == 2) {
            predictor[1] = AV_RL16(&buf[in]);
            in += 2;
            SE_16BIT(predictor[1]);
        }
        while (in < buf_size) {
            byte = buf[in++];
            diff = (byte & 0xFC) << 8;
            if ((byte & 0x03) == 3)
                shift[channel_number]++;
            else
                shift[channel_number] -= (2 * (byte & 3));
            /* saturate the shifter to a lower limit of 0 */
            if (shift[channel_number] < 0)
                shift[channel_number] = 0;
            diff >>= shift[channel_number];
            predictor[channel_number] += diff;
            predictor[channel_number] = av_clip_int16(predictor[channel_number]);
            output_samples[out++] = predictor[channel_number];
            /* toggle channel */
            channel_number ^= s->channels - 1;
        }
        break;
    case CODEC_ID_SOL_DPCM:
        in = 0;
        if (avctx->codec_tag != 3) {
            if(*data_size/4 < buf_size)
            while (in < buf_size) {
                int n1, n2;
                n1 = (buf[in] >> 4) & 0xF;
                n2 = buf[in++] & 0xF;
                s->sample[0] += s->sol_table[n1];
                 if (s->sample[0] < 0) s->sample[0] = 0;
                if (s->sample[0] > 255) s->sample[0] = 255;
                output_samples[out++] = (s->sample[0] - 128) << 8;
                s->sample[s->channels - 1] += s->sol_table[n2];
                if (s->sample[s->channels - 1] < 0) s->sample[s->channels - 1] = 0;
                if (s->sample[s->channels - 1] > 255) s->sample[s->channels - 1] = 255;
                output_samples[out++] = (s->sample[s->channels - 1] - 128) << 8;
            }
        } else {
            while (in < buf_size) {
                int n;
                n = buf[in++];
                if (n & 0x80) s->sample[channel_number] -= s->sol_table[n & 0x7F];
                else s->sample[channel_number] += s->sol_table[n & 0x7F];
                s->sample[channel_number] = av_clip_int16(s->sample[channel_number]);
                output_samples[out++] = s->sample[channel_number];
                /* toggle channel */
                channel_number ^= s->channels - 1;
            }
        }
        break;
    }
    *data_size = out * sizeof(short);
    return buf_size;
}