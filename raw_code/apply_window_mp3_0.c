static void apply_window_mp3(float *in, float *win, int *unused, float *out,

                             int incr)

{

    LOCAL_ALIGNED_16(float, suma, [17]);

    LOCAL_ALIGNED_16(float, sumb, [17]);

    LOCAL_ALIGNED_16(float, sumc, [17]);

    LOCAL_ALIGNED_16(float, sumd, [17]);



    float sum;



    /* copy to avoid wrap */

    memcpy(in + 512, in, 32 * sizeof(*in));



    apply_window(in + 16, win     , win + 512, suma, sumc, 16);

    apply_window(in + 32, win + 48, win + 640, sumb, sumd, 16);



    SUM8(MACS, suma[0], win + 32, in + 48);



    sumc[ 0] = 0;

    sumb[16] = 0;

    sumd[16] = 0;



#define SUMS(suma, sumb, sumc, sumd, out1, out2)               \

            "movups " #sumd "(%4),       %%xmm0          \n\t" \

            "shufps         $0x1b,       %%xmm0, %%xmm0  \n\t" \

            "subps  " #suma "(%1),       %%xmm0          \n\t" \

            "movaps        %%xmm0," #out1 "(%0)          \n\t" \

\

            "movups " #sumc "(%3),       %%xmm0          \n\t" \

            "shufps         $0x1b,       %%xmm0, %%xmm0  \n\t" \

            "addps  " #sumb "(%2),       %%xmm0          \n\t" \

            "movaps        %%xmm0," #out2 "(%0)          \n\t"



    if (incr == 1) {

        __asm__ volatile(

            SUMS( 0, 48,  4, 52,  0, 112)

            SUMS(16, 32, 20, 36, 16,  96)

            SUMS(32, 16, 36, 20, 32,  80)

            SUMS(48,  0, 52,  4, 48,  64)



            :"+&r"(out)

            :"r"(&suma[0]), "r"(&sumb[0]), "r"(&sumc[0]), "r"(&sumd[0])

            :"memory"

            );

        out += 16*incr;

    } else {

        int j;

        float *out2 = out + 32 * incr;

        out[0  ]  = -suma[   0];

        out += incr;

        out2 -= incr;

        for(j=1;j<16;j++) {

            *out  = -suma[   j] + sumd[16-j];

            *out2 =  sumb[16-j] + sumc[   j];

            out  += incr;

            out2 -= incr;

        }

    }



    sum = 0;

    SUM8(MLSS, sum, win + 16 + 32, in + 32);

    *out = sum;

}
