static void vector_fmul_window_mips(float *dst, const float *src0,

        const float *src1, const float *win, int len)

{

    int i, j;

    /*

     * variables used in inline assembler

     */

    float * dst_i, * dst_j, * dst_i2, * dst_j2;

    float temp, temp1, temp2, temp3, temp4, temp5, temp6, temp7;



    dst  += len;

    win  += len;

    src0 += len;



    for (i = -len, j = len - 1; i < 0; i += 8, j -= 8) {



        dst_i = dst + i;

        dst_j = dst + j;



        dst_i2 = dst + i + 4;

        dst_j2 = dst + j - 4;



        __asm__ volatile (

            "mul.s   %[temp],   %[s1],       %[wi]            \n\t"

            "mul.s   %[temp1],  %[s1],       %[wj]            \n\t"

            "mul.s   %[temp2],  %[s11],      %[wi1]           \n\t"

            "mul.s   %[temp3],  %[s11],      %[wj1]           \n\t"



            "msub.s  %[temp],   %[temp],     %[s0],  %[wj]    \n\t"

            "madd.s  %[temp1],  %[temp1],    %[s0],  %[wi]    \n\t"

            "msub.s  %[temp2],  %[temp2],    %[s01], %[wj1]   \n\t"

            "madd.s  %[temp3],  %[temp3],    %[s01], %[wi1]   \n\t"



            "swc1    %[temp],   0(%[dst_i])                   \n\t" /* dst[i] = s0*wj - s1*wi; */

            "swc1    %[temp1],  0(%[dst_j])                   \n\t" /* dst[j] = s0*wi + s1*wj; */

            "swc1    %[temp2],  4(%[dst_i])                   \n\t" /* dst[i+1] = s01*wj1 - s11*wi1; */

            "swc1    %[temp3], -4(%[dst_j])                   \n\t" /* dst[j-1] = s01*wi1 + s11*wj1; */



            "mul.s   %[temp4],  %[s12],      %[wi2]           \n\t"

            "mul.s   %[temp5],  %[s12],      %[wj2]           \n\t"

            "mul.s   %[temp6],  %[s13],      %[wi3]           \n\t"

            "mul.s   %[temp7],  %[s13],      %[wj3]           \n\t"



            "msub.s  %[temp4],  %[temp4],    %[s02], %[wj2]   \n\t"

            "madd.s  %[temp5],  %[temp5],    %[s02], %[wi2]   \n\t"

            "msub.s  %[temp6],  %[temp6],    %[s03], %[wj3]   \n\t"

            "madd.s  %[temp7],  %[temp7],    %[s03], %[wi3]   \n\t"



            "swc1    %[temp4],  8(%[dst_i])                   \n\t" /* dst[i+2] = s02*wj2 - s12*wi2; */

            "swc1    %[temp5], -8(%[dst_j])                   \n\t" /* dst[j-2] = s02*wi2 + s12*wj2; */

            "swc1    %[temp6],  12(%[dst_i])                  \n\t" /* dst[i+2] = s03*wj3 - s13*wi3; */

            "swc1    %[temp7], -12(%[dst_j])                  \n\t" /* dst[j-3] = s03*wi3 + s13*wj3; */

            : [temp]"=&f"(temp),  [temp1]"=&f"(temp1), [temp2]"=&f"(temp2),

              [temp3]"=&f"(temp3), [temp4]"=&f"(temp4), [temp5]"=&f"(temp5),

              [temp6]"=&f"(temp6), [temp7]"=&f"(temp7)

            : [dst_j]"r"(dst_j),     [dst_i]"r" (dst_i),

              [s0] "f"(src0[i]),     [wj] "f"(win[j]),     [s1] "f"(src1[j]),

              [wi] "f"(win[i]),      [s01]"f"(src0[i + 1]),[wj1]"f"(win[j - 1]),

              [s11]"f"(src1[j - 1]), [wi1]"f"(win[i + 1]), [s02]"f"(src0[i + 2]),

              [wj2]"f"(win[j - 2]),  [s12]"f"(src1[j - 2]),[wi2]"f"(win[i + 2]),

              [s03]"f"(src0[i + 3]), [wj3]"f"(win[j - 3]), [s13]"f"(src1[j - 3]),

              [wi3]"f"(win[i + 3])

            : "memory"

        );



        __asm__ volatile (

            "mul.s  %[temp],   %[s1],       %[wi]            \n\t"

            "mul.s  %[temp1],  %[s1],       %[wj]            \n\t"

            "mul.s  %[temp2],  %[s11],      %[wi1]           \n\t"

            "mul.s  %[temp3],  %[s11],      %[wj1]           \n\t"



            "msub.s %[temp],   %[temp],     %[s0],  %[wj]    \n\t"

            "madd.s %[temp1],  %[temp1],    %[s0],  %[wi]    \n\t"

            "msub.s %[temp2],  %[temp2],    %[s01], %[wj1]   \n\t"

            "madd.s %[temp3],  %[temp3],    %[s01], %[wi1]   \n\t"



            "swc1   %[temp],   0(%[dst_i2])                  \n\t" /* dst[i] = s0*wj - s1*wi; */

            "swc1   %[temp1],  0(%[dst_j2])                  \n\t" /* dst[j] = s0*wi + s1*wj; */

            "swc1   %[temp2],  4(%[dst_i2])                  \n\t" /* dst[i+1] = s01*wj1 - s11*wi1; */

            "swc1   %[temp3], -4(%[dst_j2])                  \n\t" /* dst[j-1] = s01*wi1 + s11*wj1; */



            "mul.s  %[temp4],  %[s12],      %[wi2]           \n\t"

            "mul.s  %[temp5],  %[s12],      %[wj2]           \n\t"

            "mul.s  %[temp6],  %[s13],      %[wi3]           \n\t"

            "mul.s  %[temp7],  %[s13],      %[wj3]           \n\t"



            "msub.s %[temp4],  %[temp4],    %[s02], %[wj2]   \n\t"

            "madd.s %[temp5],  %[temp5],    %[s02], %[wi2]   \n\t"

            "msub.s %[temp6],  %[temp6],    %[s03], %[wj3]   \n\t"

            "madd.s %[temp7],  %[temp7],    %[s03], %[wi3]   \n\t"



            "swc1   %[temp4],  8(%[dst_i2])                  \n\t" /* dst[i+2] = s02*wj2 - s12*wi2; */

            "swc1   %[temp5], -8(%[dst_j2])                  \n\t" /* dst[j-2] = s02*wi2 + s12*wj2; */

            "swc1   %[temp6],  12(%[dst_i2])                 \n\t" /* dst[i+2] = s03*wj3 - s13*wi3; */

            "swc1   %[temp7], -12(%[dst_j2])                 \n\t" /* dst[j-3] = s03*wi3 + s13*wj3; */

            : [temp]"=&f"(temp),

              [temp1]"=&f"(temp1), [temp2]"=&f"(temp2), [temp3]"=&f"(temp3),

              [temp4]"=&f"(temp4), [temp5]"=&f"(temp5), [temp6]"=&f"(temp6),

              [temp7]  "=&f" (temp7)

            : [dst_j2]"r"(dst_j2),   [dst_i2]"r"(dst_i2),

              [s0] "f"(src0[i + 4]), [wj] "f"(win[j - 4]), [s1] "f"(src1[j - 4]),

              [wi] "f"(win[i + 4]),  [s01]"f"(src0[i + 5]),[wj1]"f"(win[j - 5]),

              [s11]"f"(src1[j - 5]), [wi1]"f"(win[i + 5]), [s02]"f"(src0[i + 6]),

              [wj2]"f"(win[j - 6]),  [s12]"f"(src1[j - 6]),[wi2]"f"(win[i + 6]),

              [s03]"f"(src0[i + 7]), [wj3]"f"(win[j - 7]), [s13]"f"(src1[j - 7]),

              [wi3]"f"(win[i + 7])

            : "memory"

        );

    }

}
