void video_decode_example(const char *outfilename, const char *filename)

{

    AVCodec *codec;

    AVCodecContext *c= NULL;

    int frame, size, got_picture, len;

    FILE *f;

    AVFrame *picture;

    uint8_t inbuf[INBUF_SIZE + FF_INPUT_BUFFER_PADDING_SIZE], *inbuf_ptr;

    char buf[1024];



    /* set end of buffer to 0 (this ensures that no overreading happens for damaged mpeg streams) */

    memset(inbuf + INBUF_SIZE, 0, FF_INPUT_BUFFER_PADDING_SIZE);



    printf("Video decoding\n");



    /* find the mpeg1 video decoder */

    codec = avcodec_find_decoder(CODEC_ID_MPEG1VIDEO);

    if (!codec) {

        fprintf(stderr, "codec not found\n");

        exit(1);

    }



    c= avcodec_alloc_context();

    picture= avcodec_alloc_frame();



    if(codec->capabilities&CODEC_CAP_TRUNCATED)

        c->flags|= CODEC_FLAG_TRUNCATED; /* we dont send complete frames */



    /* for some codecs, such as msmpeg4 and mpeg4, width and height

       MUST be initialized there because these info are not available

       in the bitstream */



    /* open it */

    if (avcodec_open(c, codec) < 0) {

        fprintf(stderr, "could not open codec\n");

        exit(1);

    }

    

    /* the codec gives us the frame size, in samples */



    f = fopen(filename, "r");

    if (!f) {

        fprintf(stderr, "could not open %s\n", filename);

        exit(1);

    }

    

    frame = 0;

    for(;;) {

        size = fread(inbuf, 1, INBUF_SIZE, f);

        if (size == 0)

            break;



        /* NOTE1: some codecs are stream based (mpegvideo, mpegaudio)

           and this is the only method to use them because you cannot

           know the compressed data size before analysing it. 



           BUT some other codecs (msmpeg4, mpeg4) are inherently frame

           based, so you must call them with all the data for one

           frame exactly. You must also initialize 'width' and

           'height' before initializing them. */



        /* NOTE2: some codecs allow the raw parameters (frame size,

           sample rate) to be changed at any frame. We handle this, so

           you should also take care of it */



        /* here, we use a stream based decoder (mpeg1video), so we

           feed decoder and see if it could decode a frame */

        inbuf_ptr = inbuf;

        while (size > 0) {

            len = avcodec_decode_video(c, picture, &got_picture, 

                                       inbuf_ptr, size);

            if (len < 0) {

                fprintf(stderr, "Error while decoding frame %d\n", frame);

                exit(1);

            }

            if (got_picture) {

                printf("saving frame %3d\n", frame);

                fflush(stdout);



                /* the picture is allocated by the decoder. no need to

                   free it */

                snprintf(buf, sizeof(buf), outfilename, frame);

                pgm_save(picture->data[0], picture->linesize[0], 

                         c->width, c->height, buf);

                frame++;

            }

            size -= len;

            inbuf_ptr += len;

        }

    }



    /* some codecs, such as MPEG, transmit the I and P frame with a

       latency of one frame. You must do the following to have a

       chance to get the last frame of the video */

    len = avcodec_decode_video(c, picture, &got_picture, 

                               NULL, 0);

    if (got_picture) {

        printf("saving last frame %3d\n", frame);

        fflush(stdout);

        

        /* the picture is allocated by the decoder. no need to

           free it */

        snprintf(buf, sizeof(buf), outfilename, frame);

        pgm_save(picture->data[0], picture->linesize[0], 

                 c->width, c->height, buf);

        frame++;

    }

        

    fclose(f);



    avcodec_close(c);

    free(c);

    free(picture);

    printf("\n");

}
