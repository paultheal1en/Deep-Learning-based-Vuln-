static void restore_median_il(uint8_t *src, int step, int stride,

                              int width, int height, int slices, int rmode)

{

    int i, j, slice;

    int A, B, C;

    uint8_t *bsrc;

    int slice_start, slice_height;

    const int cmask   = ~(rmode ? 3 : 1);

    const int stride2 = stride << 1;



    for (slice = 0; slice < slices; slice++) {

        slice_start    = ((slice * height) / slices) & cmask;

        slice_height   = ((((slice + 1) * height) / slices) & cmask) -

                         slice_start;

        slice_height >>= 1;



        bsrc = src + slice_start * stride;



        // first line - left neighbour prediction

        bsrc[0] += 0x80;

        A        = bsrc[0];

        for (i = step; i < width * step; i += step) {

            bsrc[i] += A;

            A        = bsrc[i];

        }

        for (i = 0; i < width * step; i += step) {

            bsrc[stride + i] += A;

            A                 = bsrc[stride + i];

        }

        bsrc += stride2;

        if (slice_height == 1)

            continue;

        // second line - first element has top prediction, the rest uses median

        C        = bsrc[-stride2];

        bsrc[0] += C;

        A        = bsrc[0];

        for (i = step; i < width * step; i += step) {

            B        = bsrc[i - stride2];

            bsrc[i] += mid_pred(A, B, (uint8_t)(A + B - C));

            C        = B;

            A        = bsrc[i];

        }

        for (i = 0; i < width * step; i += step) {

            B                 = bsrc[i - stride];

            bsrc[stride + i] += mid_pred(A, B, (uint8_t)(A + B - C));

            C                 = B;

            A                 = bsrc[stride + i];

        }

        bsrc += stride2;

        // the rest of lines use continuous median prediction

        for (j = 2; j < slice_height; j++) {

            for (i = 0; i < width * step; i += step) {

                B        = bsrc[i - stride2];

                bsrc[i] += mid_pred(A, B, (uint8_t)(A + B - C));

                C        = B;

                A        = bsrc[i];

            }

            for (i = 0; i < width * step; i += step) {

                B                 = bsrc[i - stride];

                bsrc[i + stride] += mid_pred(A, B, (uint8_t)(A + B - C));

                C                 = B;

                A                 = bsrc[i + stride];

            }

            bsrc += stride2;

        }

    }

}
