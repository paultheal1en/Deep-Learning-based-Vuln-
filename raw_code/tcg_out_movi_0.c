static void tcg_out_movi(TCGContext *s, TCGType type,

                         TCGReg t0, tcg_target_long arg)

{

    uint8_t *old_code_ptr = s->code_ptr;

    uint32_t arg32 = arg;

    if (type == TCG_TYPE_I32 || arg == arg32) {

        tcg_out_op_t(s, INDEX_op_movi_i32);

        tcg_out_r(s, t0);

        tcg_out32(s, arg32);

    } else {

        assert(type == TCG_TYPE_I64);

#if TCG_TARGET_REG_BITS == 64

        tcg_out_op_t(s, INDEX_op_movi_i64);

        tcg_out_r(s, t0);

        tcg_out64(s, arg);

#else

        TODO();

#endif

    }

    old_code_ptr[1] = s->code_ptr - old_code_ptr;

}
