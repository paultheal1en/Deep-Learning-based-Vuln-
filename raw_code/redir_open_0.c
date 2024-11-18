int redir_open(AVFormatContext **ic_ptr, ByteIOContext *f)

{

    char buf[4096], *q;

    int c;

    AVFormatContext *ic = NULL;



    /* parse each URL and try to open it */

    c = url_fgetc(f);

    while (c != URL_EOF) {

        /* skip spaces */

        for(;;) {

            if (!redir_isspace(c))

                break;

            c = url_fgetc(f);

        }

        if (c == URL_EOF)

            break;

        /* record url */

        q = buf;

        for(;;) {

            if (c == URL_EOF || redir_isspace(c))

                break;

            if ((q - buf) < sizeof(buf) - 1)

                *q++ = c;

            c = url_fgetc(f);

        }

        *q = '\0';

        //printf("URL='%s'\n", buf);

        /* try to open the media file */

        if (av_open_input_file(&ic, buf, NULL, 0, NULL) == 0)

            break;

    }

    *ic_ptr = ic;

    if (!ic)

        return AVERROR(EIO);

    else

        return 0;

}
