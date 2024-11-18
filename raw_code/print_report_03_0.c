static void print_report(OutputFile *output_files,

                         OutputStream *ost_table, int nb_ostreams,

                         int is_last_report, int64_t timer_start)

{

    char buf[1024];

    OutputStream *ost;

    AVFormatContext *oc;

    int64_t total_size;

    AVCodecContext *enc;

    int frame_number, vid, i;

    double bitrate;

    int64_t pts = INT64_MAX;

    static int64_t last_time = -1;

    static int qp_histogram[52];

    int hours, mins, secs, us;



    if (!is_last_report) {

        int64_t cur_time;

        /* display the report every 0.5 seconds */

        cur_time = av_gettime();

        if (last_time == -1) {

            last_time = cur_time;

            return;

        }

        if ((cur_time - last_time) < 500000)

            return;

        last_time = cur_time;

    }





    oc = output_files[0].ctx;



    total_size = avio_size(oc->pb);

    if(total_size<0) // FIXME improve avio_size() so it works with non seekable output too

        total_size= avio_tell(oc->pb);



    buf[0] = '\0';

    vid = 0;

    for(i=0;i<nb_ostreams;i++) {

        float q = -1;

        ost = &ost_table[i];

        enc = ost->st->codec;

        if (!ost->st->stream_copy && enc->coded_frame)

            q = enc->coded_frame->quality/(float)FF_QP2LAMBDA;

        if (vid && enc->codec_type == AVMEDIA_TYPE_VIDEO) {

            snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), "q=%2.1f ", q);

        }

        if (!vid && enc->codec_type == AVMEDIA_TYPE_VIDEO) {

            float t = (av_gettime()-timer_start) / 1000000.0;



            frame_number = ost->frame_number;

            snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), "frame=%5d fps=%3d q=%3.1f ",

                     frame_number, (t>1)?(int)(frame_number/t+0.5) : 0, q);

            if(is_last_report)

                snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), "L");

            if(qp_hist){

                int j;

                int qp = lrintf(q);

                if(qp>=0 && qp<FF_ARRAY_ELEMS(qp_histogram))

                    qp_histogram[qp]++;

                for(j=0; j<32; j++)

                    snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), "%X", (int)lrintf(log(qp_histogram[j]+1)/log(2)));

            }

            if (enc->flags&CODEC_FLAG_PSNR){

                int j;

                double error, error_sum=0;

                double scale, scale_sum=0;

                char type[3]= {'Y','U','V'};

                snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), "PSNR=");

                for(j=0; j<3; j++){

                    if(is_last_report){

                        error= enc->error[j];

                        scale= enc->width*enc->height*255.0*255.0*frame_number;

                    }else{

                        error= enc->coded_frame->error[j];

                        scale= enc->width*enc->height*255.0*255.0;

                    }

                    if(j) scale/=4;

                    error_sum += error;

                    scale_sum += scale;

                    snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), "%c:%2.2f ", type[j], psnr(error/scale));

                }

                snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), "*:%2.2f ", psnr(error_sum/scale_sum));

            }

            vid = 1;

        }

        /* compute min output value */

        pts = FFMIN(pts, av_rescale_q(ost->st->pts.val,

                                      ost->st->time_base, AV_TIME_BASE_Q));

    }



    secs = pts / AV_TIME_BASE;

    us = pts % AV_TIME_BASE;

    mins = secs / 60;

    secs %= 60;

    hours = mins / 60;

    mins %= 60;



    bitrate = pts ? total_size * 8 / (pts / 1000.0) : 0;



    snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf),

             "size=%8.0fkB time=", total_size / 1024.0);

    snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf),

             "%02d:%02d:%02d.%02d ", hours, mins, secs,

             (100 * us) / AV_TIME_BASE);

    snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf),

             "bitrate=%6.1fkbits/s", bitrate);



    if (nb_frames_dup || nb_frames_drop)

        snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), " dup=%d drop=%d",

                nb_frames_dup, nb_frames_drop);



    av_log(NULL, is_last_report ? AV_LOG_WARNING : AV_LOG_INFO, "%s    \r", buf);



    fflush(stderr);



    if (is_last_report) {

        int64_t raw= audio_size + video_size + extra_size;

        av_log(NULL, AV_LOG_INFO, "\n");

        av_log(NULL, AV_LOG_INFO, "video:%1.0fkB audio:%1.0fkB global headers:%1.0fkB muxing overhead %f%%\n",

               video_size/1024.0,

               audio_size/1024.0,

               extra_size/1024.0,

               100.0*(total_size - raw)/raw

        );

    }

}
