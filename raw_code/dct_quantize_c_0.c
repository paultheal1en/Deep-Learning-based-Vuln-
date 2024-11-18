static int dct_quantize_c(MpegEncContext *s, 

                        DCTELEM *block, int n,

                        int qscale, int *overflow)

{

    int i, j, level, last_non_zero, q;

    const int *qmat;

    int bias;

    int max=0;

    unsigned int threshold1, threshold2;



    av_fdct (block);



    /* we need this permutation so that we correct the IDCT

       permutation. will be moved into DCT code */

    block_permute(block);



    if (s->mb_intra) {

        if (!s->h263_aic) {

            if (n < 4)

                q = s->y_dc_scale;

            else

                q = s->c_dc_scale;

            q = q << 3;

        } else

            /* For AIC we skip quant/dequant of INTRADC */

            q = 1 << 3;

            

        /* note: block[0] is assumed to be positive */

        block[0] = (block[0] + (q >> 1)) / q;

        i = 1;

        last_non_zero = 0;

        qmat = s->q_intra_matrix[qscale];

        bias= s->intra_quant_bias<<(QMAT_SHIFT - 3 - QUANT_BIAS_SHIFT);

    } else {

        i = 0;

        last_non_zero = -1;

        qmat = s->q_inter_matrix[qscale];

        bias= s->inter_quant_bias<<(QMAT_SHIFT - 3 - QUANT_BIAS_SHIFT);

    }

    threshold1= (1<<(QMAT_SHIFT - 3)) - bias - 1;

    threshold2= threshold1<<1;



    for(;i<64;i++) {

        j = zigzag_direct[i];

        level = block[j];

        level = level * qmat[j];



//        if(   bias+level >= (1<<(QMAT_SHIFT - 3))

//           || bias-level >= (1<<(QMAT_SHIFT - 3))){

        if(((unsigned)(level+threshold1))>threshold2){

            if(level>0){

                level= (bias + level)>>(QMAT_SHIFT - 3);

                block[j]= level;

            }else{

                level= (bias - level)>>(QMAT_SHIFT - 3);

                block[j]= -level;

            }

            max |=level;

            last_non_zero = i;

        }else{

            block[j]=0;

        }

    }

    *overflow= s->max_qcoeff < max; //overflow might have happend

    

    return last_non_zero;

}
