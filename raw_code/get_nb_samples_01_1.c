static int get_nb_samples(AVCodecContext *avctx, const uint8_t *buf,
                          int buf_size, int *coded_samples)
{
    ADPCMDecodeContext *s = avctx->priv_data;
    int nb_samples        = 0;
    int ch                = avctx->channels;
    int has_coded_samples = 0;
    int header_size;
    *coded_samples = 0;
    switch (avctx->codec->id) {
    /* constant, only check buf_size */
    case CODEC_ID_ADPCM_EA_XAS:
        if (buf_size < 76 * ch)
        nb_samples = 128;
        break;
    case CODEC_ID_ADPCM_IMA_QT:
        if (buf_size < 34 * ch)
        nb_samples = 64;
        break;
    /* simple 4-bit adpcm */
    case CODEC_ID_ADPCM_CT:
    case CODEC_ID_ADPCM_IMA_EA_SEAD:
    case CODEC_ID_ADPCM_IMA_WS:
    case CODEC_ID_ADPCM_YAMAHA:
        nb_samples = buf_size * 2 / ch;
        break;
    }
    if (nb_samples)
        return nb_samples;
    /* simple 4-bit adpcm, with header */
    header_size = 0;
    switch (avctx->codec->id) {
        case CODEC_ID_ADPCM_4XM:
        case CODEC_ID_ADPCM_IMA_ISS:     header_size = 4 * ch;      break;
        case CODEC_ID_ADPCM_IMA_AMV:     header_size = 8;           break;
        case CODEC_ID_ADPCM_IMA_SMJPEG:  header_size = 4;           break;
    }
    if (header_size > 0)
        return (buf_size - header_size) * 2 / ch;
    /* more complex formats */
    switch (avctx->codec->id) {
    case CODEC_ID_ADPCM_EA:
        has_coded_samples = 1;
        if (buf_size < 4)
        *coded_samples  = AV_RL32(buf);
        *coded_samples -= *coded_samples % 28;
        nb_samples      = (buf_size - 12) / 30 * 28;
        break;
    case CODEC_ID_ADPCM_IMA_EA_EACS:
        has_coded_samples = 1;
        if (buf_size < 4)
        *coded_samples = AV_RL32(buf);
        nb_samples     = (buf_size - (4 + 8 * ch)) * 2 / ch;
        break;
    case CODEC_ID_ADPCM_EA_MAXIS_XA:
        nb_samples = ((buf_size - ch) / (2 * ch)) * 2 * ch;
        break;
    case CODEC_ID_ADPCM_EA_R1:
    case CODEC_ID_ADPCM_EA_R2:
    case CODEC_ID_ADPCM_EA_R3:
        /* maximum number of samples */
        /* has internal offsets and a per-frame switch to signal raw 16-bit */
        has_coded_samples = 1;
        if (buf_size < 4)
        switch (avctx->codec->id) {
        case CODEC_ID_ADPCM_EA_R1:
            header_size    = 4 + 9 * ch;
            *coded_samples = AV_RL32(buf);
            break;
        case CODEC_ID_ADPCM_EA_R2:
            header_size    = 4 + 5 * ch;
            *coded_samples = AV_RL32(buf);
            break;
        case CODEC_ID_ADPCM_EA_R3:
            header_size    = 4 + 5 * ch;
            *coded_samples = AV_RB32(buf);
            break;
        }
        *coded_samples -= *coded_samples % 28;
        nb_samples      = (buf_size - header_size) * 2 / ch;
        nb_samples     -= nb_samples % 28;
        break;
    case CODEC_ID_ADPCM_IMA_DK3:
        if (avctx->block_align > 0)
            buf_size = FFMIN(buf_size, avctx->block_align);
        nb_samples = ((buf_size - 16) * 8 / 3) / ch;
        break;
    case CODEC_ID_ADPCM_IMA_DK4:
        nb_samples = 1 + (buf_size - 4 * ch) * 2 / ch;
        break;
    case CODEC_ID_ADPCM_IMA_WAV:
        if (avctx->block_align > 0)
            buf_size = FFMIN(buf_size, avctx->block_align);
        nb_samples = 1 + (buf_size - 4 * ch) / (4 * ch) * 8;
        break;
    case CODEC_ID_ADPCM_MS:
        if (avctx->block_align > 0)
            buf_size = FFMIN(buf_size, avctx->block_align);
        nb_samples = 2 + (buf_size - 7 * ch) * 2 / ch;
        break;
    case CODEC_ID_ADPCM_SBPRO_2:
    case CODEC_ID_ADPCM_SBPRO_3:
    case CODEC_ID_ADPCM_SBPRO_4:
    {
        int samples_per_byte;
        switch (avctx->codec->id) {
        case CODEC_ID_ADPCM_SBPRO_2: samples_per_byte = 4; break;
        case CODEC_ID_ADPCM_SBPRO_3: samples_per_byte = 3; break;
        case CODEC_ID_ADPCM_SBPRO_4: samples_per_byte = 2; break;
        }
        if (!s->status[0].step_index) {
            nb_samples++;
            buf_size -= ch;
        }
        nb_samples += buf_size * samples_per_byte / ch;
        break;
    }
    case CODEC_ID_ADPCM_SWF:
    {
        int buf_bits       = buf_size * 8 - 2;
        int nbits          = (buf[0] >> 6) + 2;
        int block_hdr_size = 22 * ch;
        int block_size     = block_hdr_size + nbits * ch * 4095;
        int nblocks        = buf_bits / block_size;
        int bits_left      = buf_bits - nblocks * block_size;
        nb_samples         = nblocks * 4096;
        if (bits_left >= block_hdr_size)
            nb_samples += 1 + (bits_left - block_hdr_size) / (nbits * ch);
        break;
    }
    case CODEC_ID_ADPCM_THP:
        has_coded_samples = 1;
        if (buf_size < 8)
        *coded_samples  = AV_RB32(&buf[4]);
        *coded_samples -= *coded_samples % 14;
        nb_samples      = (buf_size - 80) / (8 * ch) * 14;
        break;
    case CODEC_ID_ADPCM_XA:
        nb_samples = (buf_size / 128) * 224 / ch;
        break;
    }
    /* validate coded sample count */
    if (has_coded_samples && (*coded_samples <= 0 || *coded_samples > nb_samples))
        return AVERROR_INVALIDDATA;
    return nb_samples;
}