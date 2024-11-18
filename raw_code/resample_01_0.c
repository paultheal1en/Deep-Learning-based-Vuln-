static int resample(SwrContext *s, AudioData *out_param, int out_count,

                             const AudioData * in_param, int in_count){

    AudioData in, out, tmp;

    int ret_sum=0;

    int border=0;



    av_assert1(s->in_buffer.ch_count == in_param->ch_count);

    av_assert1(s->in_buffer.planar   == in_param->planar);

    av_assert1(s->in_buffer.fmt      == in_param->fmt);



    tmp=out=*out_param;

    in =  *in_param;



    do{

        int ret, size, consumed;

        if(!s->resample_in_constraint && s->in_buffer_count){

            buf_set(&tmp, &s->in_buffer, s->in_buffer_index);

            ret= s->resampler->multiple_resample(s->resample, &out, out_count, &tmp, s->in_buffer_count, &consumed);

            out_count -= ret;

            ret_sum += ret;

            buf_set(&out, &out, ret);

            s->in_buffer_count -= consumed;

            s->in_buffer_index += consumed;



            if(!in_count)

                break;

            if(s->in_buffer_count <= border){

                buf_set(&in, &in, -s->in_buffer_count);

                in_count += s->in_buffer_count;

                s->in_buffer_count=0;

                s->in_buffer_index=0;

                border = 0;

            }

        }



        if((s->flushed || in_count) && !s->in_buffer_count){

            s->in_buffer_index=0;

            ret= s->resampler->multiple_resample(s->resample, &out, out_count, &in, in_count, &consumed);

            out_count -= ret;

            ret_sum += ret;

            buf_set(&out, &out, ret);

            in_count -= consumed;

            buf_set(&in, &in, consumed);

        }



        //TODO is this check sane considering the advanced copy avoidance below

        size= s->in_buffer_index + s->in_buffer_count + in_count;

        if(   size > s->in_buffer.count

           && s->in_buffer_count + in_count <= s->in_buffer_index){

            buf_set(&tmp, &s->in_buffer, s->in_buffer_index);

            copy(&s->in_buffer, &tmp, s->in_buffer_count);

            s->in_buffer_index=0;

        }else

            if((ret=swri_realloc_audio(&s->in_buffer, size)) < 0)

                return ret;



        if(in_count){

            int count= in_count;

            if(s->in_buffer_count && s->in_buffer_count+2 < count && out_count) count= s->in_buffer_count+2;



            buf_set(&tmp, &s->in_buffer, s->in_buffer_index + s->in_buffer_count);

            copy(&tmp, &in, /*in_*/count);

            s->in_buffer_count += count;

            in_count -= count;

            border += count;

            buf_set(&in, &in, count);

            s->resample_in_constraint= 0;

            if(s->in_buffer_count != count || in_count)

                continue;

        }

        break;

    }while(1);



    s->resample_in_constraint= !!out_count;



    return ret_sum;

}
