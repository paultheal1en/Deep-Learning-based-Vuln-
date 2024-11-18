static enum AVPixelFormat dshow_pixfmt(DWORD biCompression, WORD biBitCount)

{

    switch(biCompression) {

    case BI_BITFIELDS:

    case BI_RGB:

        switch(biBitCount) { /* 1-8 are untested */

            case 1:

                return AV_PIX_FMT_MONOWHITE;

            case 4:

                return AV_PIX_FMT_RGB4;

            case 8:

                return AV_PIX_FMT_RGB8;

            case 16:

                return AV_PIX_FMT_RGB555;

            case 24:

                return AV_PIX_FMT_BGR24;

            case 32:

                return AV_PIX_FMT_0RGB32;

        }

    }

    return avpriv_find_pix_fmt(ff_raw_pix_fmt_tags, biCompression); // all others

}
