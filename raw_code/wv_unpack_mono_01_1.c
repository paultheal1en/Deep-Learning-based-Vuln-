static inline int wv_unpack_mono(WavpackFrameContext *s, GetBitContext *gb, void *dst, const int type)

{

    int i, j, count = 0;

    int last, t;

    int A, S, T;

    int pos = s->pos;

    uint32_t crc = s->sc.crc;

    uint32_t crc_extra_bits = s->extra_sc.crc;

    int16_t *dst16 = dst;

    int32_t *dst32 = dst;

    float   *dstfl = dst;

    const int channel_stride = s->avctx->channels;



    if(s->samples_left == s->samples)

        s->one = s->zero = s->zeroes = 0;

    do{

        T = wv_get_value(s, gb, 0, &last);

        S = 0;

        if(last) break;

        for(i = 0; i < s->terms; i++){

            t = s->decorr[i].value;

            if(t > 8){

                if(t & 1)

                    A = 2 * s->decorr[i].samplesA[0] - s->decorr[i].samplesA[1];

                else

                    A = (3 * s->decorr[i].samplesA[0] - s->decorr[i].samplesA[1]) >> 1;

                s->decorr[i].samplesA[1] = s->decorr[i].samplesA[0];

                j = 0;

            }else{

                A = s->decorr[i].samplesA[pos];

                j = (pos + t) & 7;

            }

            if(type != AV_SAMPLE_FMT_S16)

                S = T + ((s->decorr[i].weightA * (int64_t)A + 512) >> 10);

            else

                S = T + ((s->decorr[i].weightA * A + 512) >> 10);

            if(A && T) s->decorr[i].weightA -= ((((T ^ A) >> 30) & 2) - 1) * s->decorr[i].delta;

            s->decorr[i].samplesA[j] = T = S;

        }

        pos = (pos + 1) & 7;

        crc = crc * 3 + S;



        if(type == AV_SAMPLE_FMT_FLT){

            *dstfl = wv_get_value_float(s, &crc_extra_bits, S);

            dstfl += channel_stride;

        }else if(type == AV_SAMPLE_FMT_S32){

            *dst32 = wv_get_value_integer(s, &crc_extra_bits, S);

            dst32 += channel_stride;

        }else{

            *dst16 = wv_get_value_integer(s, &crc_extra_bits, S);

            dst16 += channel_stride;

        }

        count++;

    }while(!last && count < s->max_samples);



    s->samples_left -= count;

    if(!s->samples_left){

        wv_reset_saved_context(s);

        if(crc != s->CRC){

            av_log(s->avctx, AV_LOG_ERROR, "CRC error\n");

            return -1;

        }

        if(s->got_extra_bits && crc_extra_bits != s->crc_extra_bits){

            av_log(s->avctx, AV_LOG_ERROR, "Extra bits CRC error\n");

            return -1;

        }

    }else{

        s->pos = pos;

        s->sc.crc = crc;

        s->sc.bits_used = get_bits_count(&s->gb);

        if(s->got_extra_bits){

            s->extra_sc.crc = crc_extra_bits;

            s->extra_sc.bits_used = get_bits_count(&s->gb_extra_bits);

        }

    }

    return count;

}
