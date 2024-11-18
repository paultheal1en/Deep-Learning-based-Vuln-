static void tcg_out_ld (TCGContext *s, TCGType type, int ret, int arg1,

                        tcg_target_long arg2)

{

    if (type == TCG_TYPE_I32)

        tcg_out_ldst (s, ret, arg1, arg2, LWZ, LWZX);

    else

        tcg_out_ldst (s, ret, arg1, arg2, LD, LDX);

}
