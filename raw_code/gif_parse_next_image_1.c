static int gif_parse_next_image(GifState *s)

{

    ByteIOContext *f = s->f;

    int ret, code;



    for (;;) {

        code = url_fgetc(f);

#ifdef DEBUG

        printf("gif: code=%02x '%c'\n", code, code);

#endif

        switch (code) {

        case ',':

            if (gif_read_image(s) < 0)

                return AVERROR(EIO);

            ret = 0;

            goto the_end;

        case ';':

            /* end of image */

            ret = AVERROR(EIO);

            goto the_end;

        case '!':

            if (gif_read_extension(s) < 0)

                return AVERROR(EIO);

            break;

        case EOF:

        default:

            /* error or errneous EOF */

            ret = AVERROR(EIO);

            goto the_end;

        }

    }

  the_end:

    return ret;

}
