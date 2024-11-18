static inline void tcg_out_st(TCGContext *s, TCGType type, TCGReg arg,

                              TCGReg arg1, intptr_t arg2)

{

    int opi, opx;



    assert(TCG_TARGET_REG_BITS == 64 || type == TCG_TYPE_I32);

    if (type == TCG_TYPE_I32) {

        opi = STW, opx = STWX;

    } else {

        opi = STD, opx = STDX;

    }

    tcg_out_mem_long(s, opi, opx, arg, arg1, arg2);

}
