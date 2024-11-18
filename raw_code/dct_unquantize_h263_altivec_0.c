void dct_unquantize_h263_altivec(MpegEncContext *s, 

                                 DCTELEM *block, int n, int qscale)

{

POWERPC_TBL_DECLARE(altivec_dct_unquantize_h263_num, 1);

    int i, level, qmul, qadd;

    int nCoeffs;

    

    assert(s->block_last_index[n]>=0);



POWERPC_TBL_START_COUNT(altivec_dct_unquantize_h263_num, 1);

    

    qadd = (qscale - 1) | 1;

    qmul = qscale << 1;

    

    if (s->mb_intra) {

        if (!s->h263_aic) {

            if (n < 4) 

                block[0] = block[0] * s->y_dc_scale;

            else

                block[0] = block[0] * s->c_dc_scale;

        }else

            qadd = 0;

        i = 1;

        nCoeffs= 63; //does not allways use zigzag table 

    } else {

        i = 0;

        nCoeffs= s->intra_scantable.raster_end[ s->block_last_index[n] ];

    }



#ifdef ALTIVEC_USE_REFERENCE_C_CODE

    for(;i<=nCoeffs;i++) {

        level = block[i];

        if (level) {

            if (level < 0) {

                level = level * qmul - qadd;

            } else {

                level = level * qmul + qadd;

            }

            block[i] = level;

        }

    }

#else /* ALTIVEC_USE_REFERENCE_C_CODE */

    {

      register const vector short vczero = (const vector short)vec_splat_s16(0);

      short __attribute__ ((aligned(16))) qmul8[] =

          {

            qmul, qmul, qmul, qmul,

            qmul, qmul, qmul, qmul

          };

      short __attribute__ ((aligned(16))) qadd8[] =

          {

            qadd, qadd, qadd, qadd,

            qadd, qadd, qadd, qadd

          };

      short __attribute__ ((aligned(16))) nqadd8[] =

          {

            -qadd, -qadd, -qadd, -qadd,

            -qadd, -qadd, -qadd, -qadd

          };

      register vector short blockv, qmulv, qaddv, nqaddv, temp1;

      register vector bool short blockv_null, blockv_neg;

      register short backup_0 = block[0];

      register int j = 0;

      

      qmulv = vec_ld(0, qmul8);

      qaddv = vec_ld(0, qadd8);

      nqaddv = vec_ld(0, nqadd8);



#if 0 // block *is* 16 bytes-aligned, it seems.

      // first make sure block[j] is 16 bytes-aligned

      for(j = 0; (j <= nCoeffs) && ((((unsigned long)block) + (j << 1)) & 0x0000000F) ; j++) {

        level = block[j];

        if (level) {

          if (level < 0) {

                level = level * qmul - qadd;

            } else {

                level = level * qmul + qadd;

            }

            block[j] = level;

        }

      }

#endif

      

      // vectorize all the 16 bytes-aligned blocks

      // of 8 elements

      for(; (j + 7) <= nCoeffs ; j+=8)

      {

        blockv = vec_ld(j << 1, block);

        blockv_neg = vec_cmplt(blockv, vczero);

        blockv_null = vec_cmpeq(blockv, vczero);

        // choose between +qadd or -qadd as the third operand

        temp1 = vec_sel(qaddv, nqaddv, blockv_neg);

        // multiply & add (block{i,i+7} * qmul [+-] qadd)

        temp1 = vec_mladd(blockv, qmulv, temp1);

        // put 0 where block[{i,i+7} used to have 0

        blockv = vec_sel(temp1, blockv, blockv_null);

        vec_st(blockv, j << 1, block);

      }



      // if nCoeffs isn't a multiple of 8, finish the job

      // using good old scalar units.

      // (we could do it using a truncated vector,

      // but I'm not sure it's worth the hassle)

      for(; j <= nCoeffs ; j++) {

        level = block[j];

        if (level) {

          if (level < 0) {

                level = level * qmul - qadd;

            } else {

                level = level * qmul + qadd;

            }

            block[j] = level;

        }

      }

      

      if (i == 1)

      { // cheat. this avoid special-casing the first iteration

        block[0] = backup_0;

      }

    }

#endif /* ALTIVEC_USE_REFERENCE_C_CODE */



POWERPC_TBL_STOP_COUNT(altivec_dct_unquantize_h263_num, nCoeffs == 63);

}
