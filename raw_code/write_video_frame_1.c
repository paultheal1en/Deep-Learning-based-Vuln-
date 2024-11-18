void write_video_frame(AVFormatContext *oc, AVStream *st)

{

    int x, y, i, out_size;

    AVCodecContext *c;



    c = &st->codec;

    

    /* prepare a dummy image */

    /* Y */

    i = frame_count++;

    for(y=0;y<c->height;y++) {

        for(x=0;x<c->width;x++) {

            picture->data[0][y * picture->linesize[0] + x] = x + y + i * 3;

        }

    }

    

    /* Cb and Cr */

    for(y=0;y<c->height/2;y++) {

        for(x=0;x<c->width/2;x++) {

            picture->data[1][y * picture->linesize[1] + x] = 128 + y + i * 2;

            picture->data[2][y * picture->linesize[2] + x] = 64 + x + i * 5;

        }

    }



    /* encode the image */

    out_size = avcodec_encode_video(c, video_outbuf, video_outbuf_size, picture);



    /* write the compressed frame in the media file */

    if (av_write_frame(oc, st->index, video_outbuf, out_size) != 0) {

        fprintf(stderr, "Error while writing video frame\n");

        exit(1);

    }

}
