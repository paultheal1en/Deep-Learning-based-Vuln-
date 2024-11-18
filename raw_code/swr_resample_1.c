int swr_resample(AVResampleContext *c, short *dst, const short *src, int *consumed, int src_size, int dst_size, int update_ctx){

    int dst_index, i;

    int index= c->index;

    int frac= c->frac;

    int dst_incr_frac= c->dst_incr % c->src_incr;

    int dst_incr=      c->dst_incr / c->src_incr;

    int compensation_distance= c->compensation_distance;



    if(compensation_distance == 0 && c->filter_length == 1 && c->phase_shift==0){

        int64_t index2= ((int64_t)index)<<32;

        int64_t incr= (1LL<<32) * c->dst_incr / c->src_incr;

        dst_size= FFMIN(dst_size, (src_size-1-index) * (int64_t)c->src_incr / c->dst_incr);



        for(dst_index=0; dst_index < dst_size; dst_index++){

            dst[dst_index] = src[index2>>32];

            index2 += incr;

        }

        frac += dst_index * dst_incr_frac;

        index += dst_index * dst_incr;

        index += frac / c->src_incr;

        frac %= c->src_incr;

    }else{

        for(dst_index=0; dst_index < dst_size; dst_index++){

            FELEM *filter= c->filter_bank + c->filter_length*(index & c->phase_mask);

            int sample_index= index >> c->phase_shift;

            FELEM2 val=0;



            if(sample_index < 0){

                for(i=0; i<c->filter_length; i++)

                    val += src[FFABS(sample_index + i) % src_size] * filter[i];

            }else if(sample_index + c->filter_length > src_size){

                break;

            }else if(c->linear){

                FELEM2 v2=0;

                for(i=0; i<c->filter_length; i++){

                    val += src[sample_index + i] * (FELEM2)filter[i];

                    v2  += src[sample_index + i] * (FELEM2)filter[i + c->filter_length];

                }

                val+=(v2-val)*(FELEML)frac / c->src_incr;

            }else{

                for(i=0; i<c->filter_length; i++){

                    val += src[sample_index + i] * (FELEM2)filter[i];

                }

            }



#ifdef CONFIG_RESAMPLE_AUDIOPHILE_KIDDY_MODE

            dst[dst_index] = av_clip_int16(lrintf(val));

#else

            val = (val + (1<<(FILTER_SHIFT-1)))>>FILTER_SHIFT;

            dst[dst_index] = (unsigned)(val + 32768) > 65535 ? (val>>31) ^ 32767 : val;

#endif



            frac += dst_incr_frac;

            index += dst_incr;

            if(frac >= c->src_incr){

                frac -= c->src_incr;

                index++;

            }



            if(dst_index + 1 == compensation_distance){

                compensation_distance= 0;

                dst_incr_frac= c->ideal_dst_incr % c->src_incr;

                dst_incr=      c->ideal_dst_incr / c->src_incr;

            }

        }

    }

    *consumed= FFMAX(index, 0) >> c->phase_shift;

    if(index>=0) index &= c->phase_mask;



    if(compensation_distance){

        compensation_distance -= dst_index;

        assert(compensation_distance > 0);

    }

    if(update_ctx){

        c->frac= frac;

        c->index= index;

        c->dst_incr= dst_incr_frac + c->src_incr*dst_incr;

        c->compensation_distance= compensation_distance;

    }

#if 0

    if(update_ctx && !c->compensation_distance){

#undef rand

        av_resample_compensate(c, rand() % (8000*2) - 8000, 8000*2);

av_log(NULL, AV_LOG_DEBUG, "%d %d %d\n", c->dst_incr, c->ideal_dst_incr, c->compensation_distance);

    }

#endif



    return dst_index;

}
