static inline void idct_col (int16_t * col, int offset)

{

#define T1 13036

#define T2 27146

#define T3 43790

#define C4 23170



    static const short _T1[] ATTR_ALIGN(8) = {T1,T1,T1,T1};

    static const short _T2[] ATTR_ALIGN(8) = {T2,T2,T2,T2};

    static const short _T3[] ATTR_ALIGN(8) = {T3,T3,T3,T3};

    static const short _C4[] ATTR_ALIGN(8) = {C4,C4,C4,C4};



    /* column code adapted from Peter Gubanov */

    /* http://www.elecard.com/peter/idct.shtml */



    movq_m2r (*_T1, mm0);               // mm0 = T1



    movq_m2r (*(col+offset+1*8), mm1);  // mm1 = x1

    movq_r2r (mm0, mm2);                // mm2 = T1



    movq_m2r (*(col+offset+7*8), mm4);  // mm4 = x7

    pmulhw_r2r (mm1, mm0);              // mm0 = T1*x1



    movq_m2r (*_T3, mm5);               // mm5 = T3

    pmulhw_r2r (mm4, mm2);              // mm2 = T1*x7



    movq_m2r (*(col+offset+5*8), mm6);  // mm6 = x5

    movq_r2r (mm5, mm7);                // mm7 = T3-1



    movq_m2r (*(col+offset+3*8), mm3);  // mm3 = x3

    psubsw_r2r (mm4, mm0);              // mm0 = v17



    movq_m2r (*_T2, mm4);               // mm4 = T2

    pmulhw_r2r (mm3, mm5);              // mm5 = (T3-1)*x3



    paddsw_r2r (mm2, mm1);              // mm1 = u17

    pmulhw_r2r (mm6, mm7);              // mm7 = (T3-1)*x5



    /* slot */



    movq_r2r (mm4, mm2);                // mm2 = T2

    paddsw_r2r (mm3, mm5);              // mm5 = T3*x3



    pmulhw_m2r (*(col+offset+2*8), mm4);// mm4 = T2*x2

    paddsw_r2r (mm6, mm7);              // mm7 = T3*x5



    psubsw_r2r (mm6, mm5);              // mm5 = v35

    paddsw_r2r (mm3, mm7);              // mm7 = u35



    movq_m2r (*(col+offset+6*8), mm3);  // mm3 = x6

    movq_r2r (mm0, mm6);                // mm6 = v17



    pmulhw_r2r (mm3, mm2);              // mm2 = T2*x6

    psubsw_r2r (mm5, mm0);              // mm0 = b3



    psubsw_r2r (mm3, mm4);              // mm4 = v26

    paddsw_r2r (mm6, mm5);              // mm5 = v12



    movq_r2m (mm0, *(col+offset+3*8));  // save b3 in scratch0

    movq_r2r (mm1, mm6);                // mm6 = u17



    paddsw_m2r (*(col+offset+2*8), mm2);// mm2 = u26

    paddsw_r2r (mm7, mm6);              // mm6 = b0



    psubsw_r2r (mm7, mm1);              // mm1 = u12

    movq_r2r (mm1, mm7);                // mm7 = u12



    movq_m2r (*(col+offset+0*8), mm3);  // mm3 = x0

    paddsw_r2r (mm5, mm1);              // mm1 = u12+v12



    movq_m2r (*_C4, mm0);               // mm0 = C4/2

    psubsw_r2r (mm5, mm7);              // mm7 = u12-v12



    movq_r2m (mm6, *(col+offset+5*8));  // save b0 in scratch1

    pmulhw_r2r (mm0, mm1);              // mm1 = b1/2



    movq_r2r (mm4, mm6);                // mm6 = v26

    pmulhw_r2r (mm0, mm7);              // mm7 = b2/2



    movq_m2r (*(col+offset+4*8), mm5);  // mm5 = x4

    movq_r2r (mm3, mm0);                // mm0 = x0



    psubsw_r2r (mm5, mm3);              // mm3 = v04

    paddsw_r2r (mm5, mm0);              // mm0 = u04



    paddsw_r2r (mm3, mm4);              // mm4 = a1

    movq_r2r (mm0, mm5);                // mm5 = u04



    psubsw_r2r (mm6, mm3);              // mm3 = a2

    paddsw_r2r (mm2, mm5);              // mm5 = a0



    paddsw_r2r (mm1, mm1);              // mm1 = b1

    psubsw_r2r (mm2, mm0);              // mm0 = a3



    paddsw_r2r (mm7, mm7);              // mm7 = b2

    movq_r2r (mm3, mm2);                // mm2 = a2



    movq_r2r (mm4, mm6);                // mm6 = a1

    paddsw_r2r (mm7, mm3);              // mm3 = a2+b2



    psraw_i2r (COL_SHIFT, mm3);         // mm3 = y2

    paddsw_r2r (mm1, mm4);              // mm4 = a1+b1



    psraw_i2r (COL_SHIFT, mm4);         // mm4 = y1

    psubsw_r2r (mm1, mm6);              // mm6 = a1-b1



    movq_m2r (*(col+offset+5*8), mm1);  // mm1 = b0

    psubsw_r2r (mm7, mm2);              // mm2 = a2-b2



    psraw_i2r (COL_SHIFT, mm6);         // mm6 = y6

    movq_r2r (mm5, mm7);                // mm7 = a0



    movq_r2m (mm4, *(col+offset+1*8));  // save y1

    psraw_i2r (COL_SHIFT, mm2);         // mm2 = y5



    movq_r2m (mm3, *(col+offset+2*8));  // save y2

    paddsw_r2r (mm1, mm5);              // mm5 = a0+b0



    movq_m2r (*(col+offset+3*8), mm4);  // mm4 = b3

    psubsw_r2r (mm1, mm7);              // mm7 = a0-b0



    psraw_i2r (COL_SHIFT, mm5);         // mm5 = y0

    movq_r2r (mm0, mm3);                // mm3 = a3



    movq_r2m (mm2, *(col+offset+5*8));  // save y5

    psubsw_r2r (mm4, mm3);              // mm3 = a3-b3



    psraw_i2r (COL_SHIFT, mm7);         // mm7 = y7

    paddsw_r2r (mm0, mm4);              // mm4 = a3+b3



    movq_r2m (mm5, *(col+offset+0*8));  // save y0

    psraw_i2r (COL_SHIFT, mm3);         // mm3 = y4



    movq_r2m (mm6, *(col+offset+6*8));  // save y6

    psraw_i2r (COL_SHIFT, mm4);         // mm4 = y3



    movq_r2m (mm7, *(col+offset+7*8));  // save y7



    movq_r2m (mm3, *(col+offset+4*8));  // save y4



    movq_r2m (mm4, *(col+offset+3*8));  // save y3



#undef T1

#undef T2

#undef T3

#undef C4

}
