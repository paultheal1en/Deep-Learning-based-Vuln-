static int adpcm_decode_frame(AVCodecContext *avctx, void *data,
                              int *got_frame_ptr, AVPacket *avpkt)
{
    const uint8_t *buf = avpkt->data;
    int buf_size = avpkt->size;
    ADPCMDecodeContext *c = avctx->priv_data;
    ADPCMChannelStatus *cs;
    int n, m, channel, i;
    short *samples;
    const uint8_t *src;
    int st; /* stereo */
    int count1, count2;
    int nb_samples, coded_samples, ret;
    nb_samples = get_nb_samples(avctx, buf, buf_size, &coded_samples);
    if (nb_samples <= 0) {
        av_log(avctx, AV_LOG_ERROR, "invalid number of samples in packet\n");
    }
    /* get output buffer */
    c->frame.nb_samples = nb_samples;
    if ((ret = avctx->get_buffer(avctx, &c->frame)) < 0) {
        av_log(avctx, AV_LOG_ERROR, "get_buffer() failed\n");
        return ret;
    }
    samples = (short *)c->frame.data[0];
    /* use coded_samples when applicable */
    /* it is always <= nb_samples, so the output buffer will be large enough */
    if (coded_samples) {
        if (coded_samples != nb_samples)
            av_log(avctx, AV_LOG_WARNING, "mismatch in coded sample count\n");
        c->frame.nb_samples = nb_samples = coded_samples;
    }
    src = buf;
    st = avctx->channels == 2 ? 1 : 0;
    switch(avctx->codec->id) {
    case CODEC_ID_ADPCM_IMA_QT:
        /* In QuickTime, IMA is encoded by chunks of 34 bytes (=64 samples).
           Channel data is interleaved per-chunk. */
        for (channel = 0; channel < avctx->channels; channel++) {
            int16_t predictor;
            int step_index;
            cs = &(c->status[channel]);
            /* (pppppp) (piiiiiii) */
            /* Bits 15-7 are the _top_ 9 bits of the 16-bit initial predictor value */
            predictor = AV_RB16(src);
            step_index = predictor & 0x7F;
            predictor &= 0xFF80;
            src += 2;
            if (cs->step_index == step_index) {
                int diff = (int)predictor - cs->predictor;
                if (diff < 0)
                    diff = - diff;
                if (diff > 0x7f)
                    goto update;
            } else {
            update:
                cs->step_index = step_index;
                cs->predictor = predictor;
            }
            if (cs->step_index > 88){
                av_log(avctx, AV_LOG_ERROR, "ERROR: step_index = %i\n", cs->step_index);
                cs->step_index = 88;
            }
            samples = (short *)c->frame.data[0] + channel;
            for (m = 0; m < 32; m++) {
                *samples = adpcm_ima_qt_expand_nibble(cs, src[0] & 0x0F, 3);
                samples += avctx->channels;
                *samples = adpcm_ima_qt_expand_nibble(cs, src[0] >> 4  , 3);
                samples += avctx->channels;
                src ++;
            }
        }
        break;
    case CODEC_ID_ADPCM_IMA_WAV:
        if (avctx->block_align != 0 && buf_size > avctx->block_align)
            buf_size = avctx->block_align;
        for(i=0; i<avctx->channels; i++){
            cs = &(c->status[i]);
            cs->predictor = *samples++ = (int16_t)bytestream_get_le16(&src);
            cs->step_index = *src++;
            if (cs->step_index > 88){
                av_log(avctx, AV_LOG_ERROR, "ERROR: step_index = %i\n", cs->step_index);
                cs->step_index = 88;
            }
            if (*src++) av_log(avctx, AV_LOG_ERROR, "unused byte should be null but is %d!!\n", src[-1]); /* unused */
        }
        for (n = (nb_samples - 1) / 8; n > 0; n--) {
            for (i = 0; i < avctx->channels; i++) {
                cs = &c->status[i];
                for (m = 0; m < 4; m++) {
                    uint8_t v = *src++;
                    *samples = adpcm_ima_expand_nibble(cs, v & 0x0F, 3);
                    samples += avctx->channels;
                    *samples = adpcm_ima_expand_nibble(cs, v >> 4  , 3);
                    samples += avctx->channels;
                }
                samples -= 8 * avctx->channels - 1;
            }
            samples += 7 * avctx->channels;
        }
        break;
    case CODEC_ID_ADPCM_4XM:
        for (i = 0; i < avctx->channels; i++)
            c->status[i].predictor= (int16_t)bytestream_get_le16(&src);
        for (i = 0; i < avctx->channels; i++) {
            c->status[i].step_index= (int16_t)bytestream_get_le16(&src);
            c->status[i].step_index = av_clip(c->status[i].step_index, 0, 88);
        }
        for (i = 0; i < avctx->channels; i++) {
            samples = (short *)c->frame.data[0] + i;
            cs = &c->status[i];
            for (n = nb_samples >> 1; n > 0; n--, src++) {
                uint8_t v = *src;
                *samples = adpcm_ima_expand_nibble(cs, v & 0x0F, 4);
                samples += avctx->channels;
                *samples = adpcm_ima_expand_nibble(cs, v >> 4  , 4);
                samples += avctx->channels;
            }
        }
        break;
    case CODEC_ID_ADPCM_MS:
    {
        int block_predictor;
        if (avctx->block_align != 0 && buf_size > avctx->block_align)
            buf_size = avctx->block_align;
        block_predictor = av_clip(*src++, 0, 6);
        c->status[0].coeff1 = ff_adpcm_AdaptCoeff1[block_predictor];
        c->status[0].coeff2 = ff_adpcm_AdaptCoeff2[block_predictor];
        if (st) {
            block_predictor = av_clip(*src++, 0, 6);
            c->status[1].coeff1 = ff_adpcm_AdaptCoeff1[block_predictor];
            c->status[1].coeff2 = ff_adpcm_AdaptCoeff2[block_predictor];
        }
        c->status[0].idelta = (int16_t)bytestream_get_le16(&src);
        if (st){
            c->status[1].idelta = (int16_t)bytestream_get_le16(&src);
        }
        c->status[0].sample1 = bytestream_get_le16(&src);
        if (st) c->status[1].sample1 = bytestream_get_le16(&src);
        c->status[0].sample2 = bytestream_get_le16(&src);
        if (st) c->status[1].sample2 = bytestream_get_le16(&src);
        *samples++ = c->status[0].sample2;
        if (st) *samples++ = c->status[1].sample2;
        *samples++ = c->status[0].sample1;
        if (st) *samples++ = c->status[1].sample1;
        for(n = (nb_samples - 2) >> (1 - st); n > 0; n--, src++) {
            *samples++ = adpcm_ms_expand_nibble(&c->status[0 ], src[0] >> 4  );
            *samples++ = adpcm_ms_expand_nibble(&c->status[st], src[0] & 0x0F);
        }
        break;
    }
    case CODEC_ID_ADPCM_IMA_DK4:
        if (avctx->block_align != 0 && buf_size > avctx->block_align)
            buf_size = avctx->block_align;
        for (channel = 0; channel < avctx->channels; channel++) {
            cs = &c->status[channel];
            cs->predictor  = (int16_t)bytestream_get_le16(&src);
            cs->step_index = *src++;
            src++;
            *samples++ = cs->predictor;
        }
        for (n = nb_samples >> (1 - st); n > 0; n--, src++) {
            uint8_t v = *src;
            *samples++ = adpcm_ima_expand_nibble(&c->status[0 ], v >> 4  , 3);
            *samples++ = adpcm_ima_expand_nibble(&c->status[st], v & 0x0F, 3);
        }
        break;
    case CODEC_ID_ADPCM_IMA_DK3:
    {
        unsigned char last_byte = 0;
        unsigned char nibble;
        int decode_top_nibble_next = 0;
        int end_of_packet = 0;
        int diff_channel;
        if (avctx->block_align != 0 && buf_size > avctx->block_align)
            buf_size = avctx->block_align;
        c->status[0].predictor  = (int16_t)AV_RL16(src + 10);
        c->status[1].predictor  = (int16_t)AV_RL16(src + 12);
        c->status[0].step_index = src[14];
        c->status[1].step_index = src[15];
        /* sign extend the predictors */
        src += 16;
        diff_channel = c->status[1].predictor;
        /* the DK3_GET_NEXT_NIBBLE macro issues the break statement when
         * the buffer is consumed */
        while (1) {
            /* for this algorithm, c->status[0] is the sum channel and
             * c->status[1] is the diff channel */
            /* process the first predictor of the sum channel */
            DK3_GET_NEXT_NIBBLE();
            adpcm_ima_expand_nibble(&c->status[0], nibble, 3);
            /* process the diff channel predictor */
            DK3_GET_NEXT_NIBBLE();
            adpcm_ima_expand_nibble(&c->status[1], nibble, 3);
            /* process the first pair of stereo PCM samples */
            diff_channel = (diff_channel + c->status[1].predictor) / 2;
            *samples++ = c->status[0].predictor + c->status[1].predictor;
            *samples++ = c->status[0].predictor - c->status[1].predictor;
            /* process the second predictor of the sum channel */
            DK3_GET_NEXT_NIBBLE();
            adpcm_ima_expand_nibble(&c->status[0], nibble, 3);
            /* process the second pair of stereo PCM samples */
            diff_channel = (diff_channel + c->status[1].predictor) / 2;
            *samples++ = c->status[0].predictor + c->status[1].predictor;
            *samples++ = c->status[0].predictor - c->status[1].predictor;
        }
        break;
    }
    case CODEC_ID_ADPCM_IMA_ISS:
        for (channel = 0; channel < avctx->channels; channel++) {
            cs = &c->status[channel];
            cs->predictor  = (int16_t)bytestream_get_le16(&src);
            cs->step_index = *src++;
            src++;
        }
        for (n = nb_samples >> (1 - st); n > 0; n--, src++) {
            uint8_t v1, v2;
            uint8_t v = *src;
            /* nibbles are swapped for mono */
            if (st) {
                v1 = v >> 4;
                v2 = v & 0x0F;
            } else {
                v2 = v >> 4;
                v1 = v & 0x0F;
            }
            *samples++ = adpcm_ima_expand_nibble(&c->status[0 ], v1, 3);
            *samples++ = adpcm_ima_expand_nibble(&c->status[st], v2, 3);
        }
        break;
    case CODEC_ID_ADPCM_IMA_WS:
        while (src < buf + buf_size) {
            uint8_t v = *src++;
            *samples++ = adpcm_ima_expand_nibble(&c->status[0],  v >> 4  , 3);
            *samples++ = adpcm_ima_expand_nibble(&c->status[st], v & 0x0F, 3);
        }
        break;
    case CODEC_ID_ADPCM_XA:
        while (buf_size >= 128) {
            xa_decode(samples, src, &c->status[0], &c->status[1],
                avctx->channels);
            src += 128;
            samples += 28 * 8;
            buf_size -= 128;
        }
        break;
    case CODEC_ID_ADPCM_IMA_EA_EACS:
        src += 4; // skip sample count (already read)
        for (i=0; i<=st; i++)
            c->status[i].step_index = bytestream_get_le32(&src);
        for (i=0; i<=st; i++)
            c->status[i].predictor  = bytestream_get_le32(&src);
        for (n = nb_samples >> (1 - st); n > 0; n--, src++) {
            *samples++ = adpcm_ima_expand_nibble(&c->status[0],  *src>>4,   3);
            *samples++ = adpcm_ima_expand_nibble(&c->status[st], *src&0x0F, 3);
        }
        break;
    case CODEC_ID_ADPCM_IMA_EA_SEAD:
        for (n = nb_samples >> (1 - st); n > 0; n--, src++) {
            *samples++ = adpcm_ima_expand_nibble(&c->status[0], src[0] >> 4, 6);
            *samples++ = adpcm_ima_expand_nibble(&c->status[st],src[0]&0x0F, 6);
        }
        break;
    case CODEC_ID_ADPCM_EA:
    {
        int32_t previous_left_sample, previous_right_sample;
        int32_t current_left_sample, current_right_sample;
        int32_t next_left_sample, next_right_sample;
        int32_t coeff1l, coeff2l, coeff1r, coeff2r;
        uint8_t shift_left, shift_right;
        /* Each EA ADPCM frame has a 12-byte header followed by 30-byte pieces,
           each coding 28 stereo samples. */
        src += 4; // skip sample count (already read)
        current_left_sample   = (int16_t)bytestream_get_le16(&src);
        previous_left_sample  = (int16_t)bytestream_get_le16(&src);
        current_right_sample  = (int16_t)bytestream_get_le16(&src);
        previous_right_sample = (int16_t)bytestream_get_le16(&src);
        for (count1 = 0; count1 < nb_samples / 28; count1++) {
            coeff1l = ea_adpcm_table[ *src >> 4       ];
            coeff2l = ea_adpcm_table[(*src >> 4  ) + 4];
            coeff1r = ea_adpcm_table[*src & 0x0F];
            coeff2r = ea_adpcm_table[(*src & 0x0F) + 4];
            src++;
            shift_left  = 20 - (*src >> 4);
            shift_right = 20 - (*src & 0x0F);
            src++;
            for (count2 = 0; count2 < 28; count2++) {
                next_left_sample  = sign_extend(*src >> 4, 4) << shift_left;
                next_right_sample = sign_extend(*src,      4) << shift_right;
                src++;
                next_left_sample = (next_left_sample +
                    (current_left_sample * coeff1l) +
                    (previous_left_sample * coeff2l) + 0x80) >> 8;
                next_right_sample = (next_right_sample +
                    (current_right_sample * coeff1r) +
                    (previous_right_sample * coeff2r) + 0x80) >> 8;
                previous_left_sample = current_left_sample;
                current_left_sample = av_clip_int16(next_left_sample);
                previous_right_sample = current_right_sample;
                current_right_sample = av_clip_int16(next_right_sample);
                *samples++ = (unsigned short)current_left_sample;
                *samples++ = (unsigned short)current_right_sample;
            }
        }
        if (src - buf == buf_size - 2)
            src += 2; // Skip terminating 0x0000
        break;
    }
    case CODEC_ID_ADPCM_EA_MAXIS_XA:
    {
        int coeff[2][2], shift[2];
        for(channel = 0; channel < avctx->channels; channel++) {
            for (i=0; i<2; i++)
                coeff[channel][i] = ea_adpcm_table[(*src >> 4) + 4*i];
            shift[channel] = 20 - (*src & 0x0F);
            src++;
        }
        for (count1 = 0; count1 < nb_samples / 2; count1++) {
            for(i = 4; i >= 0; i-=4) { /* Pairwise samples LL RR (st) or LL LL (mono) */
                for(channel = 0; channel < avctx->channels; channel++) {
                    int32_t sample = sign_extend(src[channel] >> i, 4) << shift[channel];
                    sample = (sample +
                             c->status[channel].sample1 * coeff[channel][0] +
                             c->status[channel].sample2 * coeff[channel][1] + 0x80) >> 8;
                    c->status[channel].sample2 = c->status[channel].sample1;
                    c->status[channel].sample1 = av_clip_int16(sample);
                    *samples++ = c->status[channel].sample1;
                }
            }
            src+=avctx->channels;
        }
        /* consume whole packet */
        src = buf + buf_size;
        break;
    }
    case CODEC_ID_ADPCM_EA_R1:
    case CODEC_ID_ADPCM_EA_R2:
    case CODEC_ID_ADPCM_EA_R3: {
        /* channel numbering
           2chan: 0=fl, 1=fr
           4chan: 0=fl, 1=rl, 2=fr, 3=rr
           6chan: 0=fl, 1=c,  2=fr, 3=rl,  4=rr, 5=sub */
        const int big_endian = avctx->codec->id == CODEC_ID_ADPCM_EA_R3;
        int32_t previous_sample, current_sample, next_sample;
        int32_t coeff1, coeff2;
        uint8_t shift;
        unsigned int channel;
        uint16_t *samplesC;
        const uint8_t *srcC;
        const uint8_t *src_end = buf + buf_size;
        int count = 0;
        src += 4; // skip sample count (already read)
        for (channel=0; channel<avctx->channels; channel++) {
            int32_t offset = (big_endian ? bytestream_get_be32(&src)
                                         : bytestream_get_le32(&src))
                           + (avctx->channels-channel-1) * 4;
            if ((offset < 0) || (offset >= src_end - src - 4)) break;
            srcC  = src + offset;
            samplesC = samples + channel;
            if (avctx->codec->id == CODEC_ID_ADPCM_EA_R1) {
                current_sample  = (int16_t)bytestream_get_le16(&srcC);
                previous_sample = (int16_t)bytestream_get_le16(&srcC);
            } else {
                current_sample  = c->status[channel].predictor;
                previous_sample = c->status[channel].prev_sample;
            }
            for (count1 = 0; count1 < nb_samples / 28; count1++) {
                if (*srcC == 0xEE) {  /* only seen in R2 and R3 */
                    srcC++;
                    if (srcC > src_end - 30*2) break;
                    current_sample  = (int16_t)bytestream_get_be16(&srcC);
                    previous_sample = (int16_t)bytestream_get_be16(&srcC);
                    for (count2=0; count2<28; count2++) {
                        *samplesC = (int16_t)bytestream_get_be16(&srcC);
                        samplesC += avctx->channels;
                    }
                } else {
                    coeff1 = ea_adpcm_table[ *srcC>>4     ];
                    coeff2 = ea_adpcm_table[(*srcC>>4) + 4];
                    shift = 20 - (*srcC++ & 0x0F);
                    if (srcC > src_end - 14) break;
                    for (count2=0; count2<28; count2++) {
                        if (count2 & 1)
                            next_sample = sign_extend(*srcC++,    4) << shift;
                        else
                            next_sample = sign_extend(*srcC >> 4, 4) << shift;
                        next_sample += (current_sample  * coeff1) +
                                       (previous_sample * coeff2);
                        next_sample = av_clip_int16(next_sample >> 8);
                        previous_sample = current_sample;
                        current_sample  = next_sample;
                        *samplesC = current_sample;
                        samplesC += avctx->channels;
                    }
                }
            }
            if (!count) {
                count = count1;
            } else if (count != count1) {
                av_log(avctx, AV_LOG_WARNING, "per-channel sample count mismatch\n");
                count = FFMAX(count, count1);
            }
            if (avctx->codec->id != CODEC_ID_ADPCM_EA_R1) {
                c->status[channel].predictor   = current_sample;
                c->status[channel].prev_sample = previous_sample;
            }
        }
        c->frame.nb_samples = count * 28;
        src = src_end;
        break;
    }
    case CODEC_ID_ADPCM_EA_XAS:
        for (channel=0; channel<avctx->channels; channel++) {
            int coeff[2][4], shift[4];
            short *s2, *s = &samples[channel];
            for (n=0; n<4; n++, s+=32*avctx->channels) {
                for (i=0; i<2; i++)
                    coeff[i][n] = ea_adpcm_table[(src[0]&0x0F)+4*i];
                shift[n] = 20 - (src[2] & 0x0F);
                for (s2=s, i=0; i<2; i++, src+=2, s2+=avctx->channels)
                    s2[0] = (src[0]&0xF0) + (src[1]<<8);
            }
            for (m=2; m<32; m+=2) {
                s = &samples[m*avctx->channels + channel];
                for (n=0; n<4; n++, src++, s+=32*avctx->channels) {
                    for (s2=s, i=0; i<8; i+=4, s2+=avctx->channels) {
                        int level = sign_extend(*src >> (4 - i), 4) << shift[n];
                        int pred  = s2[-1*avctx->channels] * coeff[0][n]
                                  + s2[-2*avctx->channels] * coeff[1][n];
                        s2[0] = av_clip_int16((level + pred + 0x80) >> 8);
                    }
                }
            }
        }
        break;
    case CODEC_ID_ADPCM_IMA_AMV:
    case CODEC_ID_ADPCM_IMA_SMJPEG:
        c->status[0].predictor = (int16_t)bytestream_get_le16(&src);
        c->status[0].step_index = bytestream_get_le16(&src);
        if (avctx->codec->id == CODEC_ID_ADPCM_IMA_AMV)
            src+=4;
        for (n = nb_samples >> (1 - st); n > 0; n--, src++) {
            char hi, lo;
            lo = *src & 0x0F;
            hi = *src >> 4;
            if (avctx->codec->id == CODEC_ID_ADPCM_IMA_AMV)
                FFSWAP(char, hi, lo);
            *samples++ = adpcm_ima_expand_nibble(&c->status[0],
                lo, 3);
            *samples++ = adpcm_ima_expand_nibble(&c->status[0],
                hi, 3);
        }
        break;
    case CODEC_ID_ADPCM_CT:
        for (n = nb_samples >> (1 - st); n > 0; n--, src++) {
            uint8_t v = *src;
            *samples++ = adpcm_ct_expand_nibble(&c->status[0 ], v >> 4  );
            *samples++ = adpcm_ct_expand_nibble(&c->status[st], v & 0x0F);
        }
        break;
    case CODEC_ID_ADPCM_SBPRO_4:
    case CODEC_ID_ADPCM_SBPRO_3:
    case CODEC_ID_ADPCM_SBPRO_2:
        if (!c->status[0].step_index) {
            /* the first byte is a raw sample */
            *samples++ = 128 * (*src++ - 0x80);
            if (st)
              *samples++ = 128 * (*src++ - 0x80);
            c->status[0].step_index = 1;
            nb_samples--;
        }
        if (avctx->codec->id == CODEC_ID_ADPCM_SBPRO_4) {
            for (n = nb_samples >> (1 - st); n > 0; n--, src++) {
                *samples++ = adpcm_sbpro_expand_nibble(&c->status[0],
                    src[0] >> 4, 4, 0);
                *samples++ = adpcm_sbpro_expand_nibble(&c->status[st],
                    src[0] & 0x0F, 4, 0);
            }
        } else if (avctx->codec->id == CODEC_ID_ADPCM_SBPRO_3) {
            for (n = nb_samples / 3; n > 0; n--, src++) {
                *samples++ = adpcm_sbpro_expand_nibble(&c->status[0],
                     src[0] >> 5        , 3, 0);
                *samples++ = adpcm_sbpro_expand_nibble(&c->status[0],
                    (src[0] >> 2) & 0x07, 3, 0);
                *samples++ = adpcm_sbpro_expand_nibble(&c->status[0],
                    src[0] & 0x03, 2, 0);
            }
        } else {
            for (n = nb_samples >> (2 - st); n > 0; n--, src++) {
                *samples++ = adpcm_sbpro_expand_nibble(&c->status[0],
                     src[0] >> 6        , 2, 2);
                *samples++ = adpcm_sbpro_expand_nibble(&c->status[st],
                    (src[0] >> 4) & 0x03, 2, 2);
                *samples++ = adpcm_sbpro_expand_nibble(&c->status[0],
                    (src[0] >> 2) & 0x03, 2, 2);
                *samples++ = adpcm_sbpro_expand_nibble(&c->status[st],
                    src[0] & 0x03, 2, 2);
            }
        }
        break;
    case CODEC_ID_ADPCM_SWF:
    {
        GetBitContext gb;
        const int *table;
        int k0, signmask, nb_bits, count;
        int size = buf_size*8;
        init_get_bits(&gb, buf, size);
        //read bits & initial values
        nb_bits = get_bits(&gb, 2)+2;
        //av_log(NULL,AV_LOG_INFO,"nb_bits: %d\n", nb_bits);
        table = swf_index_tables[nb_bits-2];
        k0 = 1 << (nb_bits-2);
        signmask = 1 << (nb_bits-1);
        while (get_bits_count(&gb) <= size - 22*avctx->channels) {
            for (i = 0; i < avctx->channels; i++) {
                *samples++ = c->status[i].predictor = get_sbits(&gb, 16);
                c->status[i].step_index = get_bits(&gb, 6);
            }
            for (count = 0; get_bits_count(&gb) <= size - nb_bits*avctx->channels && count < 4095; count++) {
                int i;
                for (i = 0; i < avctx->channels; i++) {
                    // similar to IMA adpcm
                    int delta = get_bits(&gb, nb_bits);
                    int step = ff_adpcm_step_table[c->status[i].step_index];
                    long vpdiff = 0; // vpdiff = (delta+0.5)*step/4
                    int k = k0;
                    do {
                        if (delta & k)
                            vpdiff += step;
                        step >>= 1;
                        k >>= 1;
                    } while(k);
                    vpdiff += step;
                    if (delta & signmask)
                        c->status[i].predictor -= vpdiff;
                    else
                        c->status[i].predictor += vpdiff;
                    c->status[i].step_index += table[delta & (~signmask)];
                    c->status[i].step_index = av_clip(c->status[i].step_index, 0, 88);
                    c->status[i].predictor = av_clip_int16(c->status[i].predictor);
                    *samples++ = c->status[i].predictor;
                }
            }
        }
        src += buf_size;
        break;
    }
    case CODEC_ID_ADPCM_YAMAHA:
        for (n = nb_samples >> (1 - st); n > 0; n--, src++) {
            uint8_t v = *src;
            *samples++ = adpcm_yamaha_expand_nibble(&c->status[0 ], v & 0x0F);
            *samples++ = adpcm_yamaha_expand_nibble(&c->status[st], v >> 4  );
        }
        break;
    case CODEC_ID_ADPCM_THP:
    {
        int table[2][16];
        int prev[2][2];
        int ch;
        src += 4; // skip channel size
        src += 4; // skip number of samples (already read)
        for (i = 0; i < 32; i++)
            table[0][i] = (int16_t)bytestream_get_be16(&src);
        /* Initialize the previous sample.  */
        for (i = 0; i < 4; i++)
            prev[0][i] = (int16_t)bytestream_get_be16(&src);
        for (ch = 0; ch <= st; ch++) {
            samples = (short *)c->frame.data[0] + ch;
            /* Read in every sample for this channel.  */
            for (i = 0; i < nb_samples / 14; i++) {
                int index = (*src >> 4) & 7;
                unsigned int exp = *src++ & 15;
                int factor1 = table[ch][index * 2];
                int factor2 = table[ch][index * 2 + 1];
                /* Decode 14 samples.  */
                for (n = 0; n < 14; n++) {
                    int32_t sampledat;
                    if(n&1) sampledat = sign_extend(*src++, 4);
                    else    sampledat = sign_extend(*src >> 4, 4);
                    sampledat = ((prev[ch][0]*factor1
                                + prev[ch][1]*factor2) >> 11) + (sampledat << exp);
                    *samples = av_clip_int16(sampledat);
                    prev[ch][1] = prev[ch][0];
                    prev[ch][0] = *samples++;
                    /* In case of stereo, skip one sample, this sample
                       is for the other channel.  */
                    samples += st;
                }
            }
        }
        break;
    }
    default:
        return -1;
    }
    *got_frame_ptr   = 1;
    *(AVFrame *)data = c->frame;
    return src - buf;
}