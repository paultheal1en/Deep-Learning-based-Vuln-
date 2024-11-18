static void FUNC(dequant)(int16_t *coeffs, int16_t log2_size)

{

    int shift  = 15 - BIT_DEPTH - log2_size;

    int x, y;

    int size = 1 << log2_size;



    if (shift > 0) {

        int offset = 1 << (shift - 1);

        for (y = 0; y < size; y++) {

            for (x = 0; x < size; x++) {

                *coeffs = (*coeffs + offset) >> shift;

                coeffs++;

            }

        }

    } else {

        for (y = 0; y < size; y++) {

            for (x = 0; x < size; x++) {

                *coeffs = *coeffs << -shift;

                coeffs++;

            }

        }

    }

}
