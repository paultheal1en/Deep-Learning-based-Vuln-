static inline void tcg_out_ld(TCGContext *s, TCGType type, TCGReg ret,

                              TCGReg arg1, intptr_t arg2)

{

    int opi, opx;



    assert(TCG_TARGET_REG_BITS == 64 || type == TCG_TYPE_I32);

    if (type == TCG_TYPE_I32) {

        opi = LWZ, opx = LWZX;

    } else {

        opi = LD, opx = LDX;

    }

    tcg_out_mem_long(s, opi, opx, ret, arg1, arg2);

}
