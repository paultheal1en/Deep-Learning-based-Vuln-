static void qpeg_decode_intra(const uint8_t *src, uint8_t *dst, int size,

                            int stride, int width, int height)

{

    int i;

    int code;

    int c0, c1;

    int run, copy;

    int filled = 0;

    int rows_to_go;



    rows_to_go = height;

    height--;

    dst = dst + height * stride;



    while((size > 0) && (rows_to_go > 0)) {

        code = *src++;

        size--;

        run = copy = 0;

        if(code == 0xFC) /* end-of-picture code */

            break;

        if(code >= 0xF8) { /* very long run */

            c0 = *src++;

            c1 = *src++;

            size -= 2;

            run = ((code & 0x7) << 16) + (c0 << 8) + c1 + 2;

        } else if (code >= 0xF0) { /* long run */

            c0 = *src++;

            size--;

            run = ((code & 0xF) << 8) + c0 + 2;

        } else if (code >= 0xE0) { /* short run */

            run = (code & 0x1F) + 2;

        } else if (code >= 0xC0) { /* very long copy */

            c0 = *src++;

            c1 = *src++;

            size -= 2;

            copy = ((code & 0x3F) << 16) + (c0 << 8) + c1 + 1;

        } else if (code >= 0x80) { /* long copy */

            c0 = *src++;

            size--;

            copy = ((code & 0x7F) << 8) + c0 + 1;

        } else { /* short copy */

            copy = code + 1;

        }



        /* perform actual run or copy */

        if(run) {

            int p;



            p = *src++;

            size--;

            for(i = 0; i < run; i++) {

                dst[filled++] = p;

                if (filled >= width) {

                    filled = 0;

                    dst -= stride;

                    rows_to_go--;

                    if(rows_to_go <= 0)

                        break;

                }

            }

        } else {

            size -= copy;

            for(i = 0; i < copy; i++) {

                dst[filled++] = *src++;

                if (filled >= width) {

                    filled = 0;

                    dst -= stride;

                    rows_to_go--;

                    if(rows_to_go <= 0)

                        break;

                }

            }

        }

    }

}
