static int avg_bits_per_pixel(int pix_fmt)

{

    int bits;

    const PixFmtInfo *pf;



    pf = &pix_fmt_info[pix_fmt];

    if (pf->is_packed) {

        switch(pix_fmt) {

        case PIX_FMT_RGB24:

        case PIX_FMT_BGR24:

            bits = 24;

            break;

        case PIX_FMT_RGBA32:

            bits = 32;

                        break;

        case PIX_FMT_RGB565:

        case PIX_FMT_RGB555:

            bits = 16;

                        break;

        case PIX_FMT_PAL8:

            bits = 8;

            break;

        default:

            bits = 32;

            break;

        }

    } else {

        bits = pf->depth;

        bits += (2 * pf->depth >> 

                 (pf->x_chroma_shift + pf->x_chroma_shift));

    }

    return bits;

}
