static void ssim_4x4x2_core(const uint8_t *main, int main_stride,

                            const uint8_t *ref, int ref_stride,

                            int sums[2][4])

{

    int x, y, z;



    for (z = 0; z < 2; z++) {

        uint32_t s1 = 0, s2 = 0, ss = 0, s12 = 0;



        for (y = 0; y < 4; y++) {

            for (x = 0; x < 4; x++) {

                int a = main[x + y * main_stride];

                int b = ref[x + y * ref_stride];



                s1  += a;

                s2  += b;

                ss  += a*a;

                ss  += b*b;

                s12 += a*b;

            }

        }



        sums[z][0] = s1;

        sums[z][1] = s2;

        sums[z][2] = ss;

        sums[z][3] = s12;

        main += 4;

        ref += 4;

    }

}
