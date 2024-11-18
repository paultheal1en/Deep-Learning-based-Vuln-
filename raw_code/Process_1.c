void Process(void *ctx, AVPicture *picture, enum PixelFormat pix_fmt, int width, int height, int64_t pts)

{

    int err = 0;

    ContextInfo *ci = (ContextInfo *) ctx;

    AVPicture picture1;

    AVPicture picture2;

    AVPicture *pict = picture;

    int out_width;

    int out_height;

    int i;

    uint8_t *ptr = NULL;

    FILE *in = rwpipe_reader( ci->rw );

    FILE *out = rwpipe_writer( ci->rw );



    /* Check that we have a pipe to talk to. */

    if ( in == NULL || out == NULL )

        err = 1;



    /* Convert to RGB24 if necessary */

    if ( !err && pix_fmt != PIX_FMT_RGB24 )

    {

        int size = avpicture_get_size(PIX_FMT_RGB24, width, height);



        if ( size != ci->size1 )

        {

            av_free( ci->buf1 );

            ci->buf1 = av_malloc(size);

            ci->size1 = size;

            err = ci->buf1 == NULL;

        }



        if ( !err )

        {

            avpicture_fill(&picture1, ci->buf1, PIX_FMT_RGB24, width, height);



            // if we already got a SWS context, let's realloc if is not re-useable

            ci->toRGB_convert_ctx = sws_getCachedContext(ci->toRGB_convert_ctx,

                                        width, height, pix_fmt,

                                        width, height, PIX_FMT_RGB24,

                                        sws_flags, NULL, NULL, NULL);

            if (ci->toRGB_convert_ctx == NULL) {

                av_log(NULL, AV_LOG_ERROR,

                       "Cannot initialize the toRGB conversion context\n");

                return;

            }



// img_convert parameters are          2 first destination, then 4 source

// sws_scale   parameters are context, 4 first source,      then 2 destination

            sws_scale(ci->toRGB_convert_ctx,

                     picture->data, picture->linesize, 0, height,

                     picture1.data, picture1.linesize);



            pict = &picture1;

        }

    }



    /* Write out the PPM */

    if ( !err )

    {

        ptr = pict->data[ 0 ];

        fprintf( out, "P6\n%d %d\n255\n", width, height );

        for ( i = 0; !err && i < height; i ++ )

        {

            err = !fwrite( ptr, width * 3, 1, out );

            ptr += pict->linesize[ 0 ];

        }

        if ( !err )

            err = fflush( out );

    }



    /* Read the PPM returned. */

    if ( !err && !rwpipe_read_ppm_header( ci->rw, &out_width, &out_height ) )

    {

        int size = avpicture_get_size(PIX_FMT_RGB24, out_width, out_height);



        if ( size != ci->size2 )

        {

            av_free( ci->buf2 );

            ci->buf2 = av_malloc(size);

            ci->size2 = size;

            err = ci->buf2 == NULL;

        }



        if ( !err )

        {

            avpicture_fill(&picture2, ci->buf2, PIX_FMT_RGB24, out_width, out_height);

            ptr = picture2.data[ 0 ];

            for ( i = 0; !err && i < out_height; i ++ )

            {

                err = !fread( ptr, out_width * 3, 1, in );

                ptr += picture2.linesize[ 0 ];

            }

        }

    }



    /* Convert the returned PPM back to the input format */

    if ( !err )

    {

        /* The out_width/out_height returned from the PPM

         * filter won't necessarily be the same as width and height

         * but it will be scaled anyway to width/height.

         */

        av_log(NULL, AV_LOG_DEBUG,

                  "PPM vhook: Input dimensions: %d x %d Output dimensions: %d x %d\n",

                  width, height, out_width, out_height);

        ci->fromRGB_convert_ctx = sws_getCachedContext(ci->fromRGB_convert_ctx,

                                        out_width, out_height, PIX_FMT_RGB24,

                                        width,     height,     pix_fmt,

                                        sws_flags, NULL, NULL, NULL);

        if (ci->fromRGB_convert_ctx == NULL) {

            av_log(NULL, AV_LOG_ERROR,

                   "Cannot initialize the fromRGB conversion context\n");

            return;

        }



// img_convert parameters are          2 first destination, then 4 source

// sws_scale   parameters are context, 4 first source,      then 2 destination

        sws_scale(ci->fromRGB_convert_ctx,

                 picture2.data, picture2.linesize, 0, out_height,

                 picture->data, picture->linesize);

    }

}
