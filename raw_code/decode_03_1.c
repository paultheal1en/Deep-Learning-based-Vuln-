static void decode(AVCodecContext *dec_ctx, AVFrame *frame, AVPacket *pkt,

                   const char *filename)

{

    char buf[1024];

    int ret;



    ret = avcodec_send_packet(dec_ctx, pkt);

    if (ret < 0) {

        fprintf(stderr, "Error sending a packet for decoding\n");

        exit(1);

    }



    while (ret >= 0) {

        ret = avcodec_receive_frame(dec_ctx, frame);

        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)

            return;

        else if (ret < 0) {

            fprintf(stderr, "Error during decoding\n");

            exit(1);

        }



        printf("saving frame %3d\n", dec_ctx->frame_number);

        fflush(stdout);



        /* the picture is allocated by the decoder. no need to

           free it */

        snprintf(buf, sizeof(buf), filename, dec_ctx->frame_number);

        pgm_save(frame->data[0], frame->linesize[0],

                 frame->width, frame->height, buf);

    }

}
