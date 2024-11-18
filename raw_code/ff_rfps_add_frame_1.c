int ff_rfps_add_frame(AVFormatContext *ic, AVStream *st, int64_t ts)

{

    int i, j;

    int64_t last = st->info->last_dts;



    if(   ts != AV_NOPTS_VALUE && last != AV_NOPTS_VALUE && ts > last

       && ts - (uint64_t)last < INT64_MAX){

        double dts= (is_relative(ts) ?  ts - RELATIVE_TS_BASE : ts) * av_q2d(st->time_base);

        int64_t duration= ts - last;



        if (!st->info->duration_error)

            st->info->duration_error = av_mallocz(sizeof(st->info->duration_error[0])*2);

        if (!st->info->duration_error)

            return AVERROR(ENOMEM);



//         if(st->codec->codec_type == AVMEDIA_TYPE_VIDEO)

//             av_log(NULL, AV_LOG_ERROR, "%f\n", dts);

        for (i=0; i<MAX_STD_TIMEBASES; i++) {

            if (st->info->duration_error[0][1][i] < 1e10) {

                int framerate= get_std_framerate(i);

                double sdts= dts*framerate/(1001*12);

                for(j=0; j<2; j++){

                    int64_t ticks= llrint(sdts+j*0.5);

                    double error= sdts - ticks + j*0.5;

                    st->info->duration_error[j][0][i] += error;

                    st->info->duration_error[j][1][i] += error*error;

                }

            }

        }

        st->info->duration_count++;




        if (st->info->duration_count % 10 == 0) {

            int n = st->info->duration_count;

            for (i=0; i<MAX_STD_TIMEBASES; i++) {

                if (st->info->duration_error[0][1][i] < 1e10) {

                    double a0     = st->info->duration_error[0][0][i] / n;

                    double error0 = st->info->duration_error[0][1][i] / n - a0*a0;

                    double a1     = st->info->duration_error[1][0][i] / n;

                    double error1 = st->info->duration_error[1][1][i] / n - a1*a1;

                    if (error0 > 0.04 && error1 > 0.04) {

                        st->info->duration_error[0][1][i] = 2e10;

                        st->info->duration_error[1][1][i] = 2e10;

                    }

                }

            }

        }



        // ignore the first 4 values, they might have some random jitter

        if (st->info->duration_count > 3 && is_relative(ts) == is_relative(last))

            st->info->duration_gcd = av_gcd(st->info->duration_gcd, duration);

    }

    if (ts != AV_NOPTS_VALUE)

        st->info->last_dts = ts;



    return 0;

}