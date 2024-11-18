static void do_video_stats(OutputStream *ost, int frame_size)

{

    AVCodecContext *enc;

    int frame_number;

    double ti1, bitrate, avg_bitrate;



    /* this is executed just the first time do_video_stats is called */

    if (!vstats_file) {

        vstats_file = fopen(vstats_filename, "w");

        if (!vstats_file) {

            perror("fopen");

            exit(1);

        }

    }



    enc = ost->st->codec;

    if (enc->codec_type == AVMEDIA_TYPE_VIDEO) {

        frame_number = ost->frame_number;

        fprintf(vstats_file, "frame= %5d q= %2.1f ", frame_number, enc->coded_frame->quality / (float)FF_QP2LAMBDA);

        if (enc->flags&CODEC_FLAG_PSNR)

            fprintf(vstats_file, "PSNR= %6.2f ", psnr(enc->coded_frame->error[0] / (enc->width * enc->height * 255.0 * 255.0)));



        fprintf(vstats_file,"f_size= %6d ", frame_size);

        /* compute pts value */

        ti1 = ost->sync_opts * av_q2d(enc->time_base);

        if (ti1 < 0.01)

            ti1 = 0.01;



        bitrate     = (frame_size * 8) / av_q2d(enc->time_base) / 1000.0;

        avg_bitrate = (double)(video_size * 8) / ti1 / 1000.0;

        fprintf(vstats_file, "s_size= %8.0fkB time= %0.3f br= %7.1fkbits/s avg_br= %7.1fkbits/s ",

               (double)video_size / 1024, ti1, bitrate, avg_bitrate);

        fprintf(vstats_file, "type= %c\n", av_get_picture_type_char(enc->coded_frame->pict_type));

    }

}
