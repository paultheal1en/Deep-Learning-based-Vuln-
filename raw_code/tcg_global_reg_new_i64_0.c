TCGv_i64 tcg_global_reg_new_i64(int reg, const char *name)

{

    int idx;



    idx = tcg_global_reg_new_internal(TCG_TYPE_I64, reg, name);

    return MAKE_TCGV_I64(idx);

}
