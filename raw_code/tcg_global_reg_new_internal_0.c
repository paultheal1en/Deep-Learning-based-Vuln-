static inline int tcg_global_reg_new_internal(TCGType type, int reg,

                                              const char *name)

{

    TCGContext *s = &tcg_ctx;

    TCGTemp *ts;

    int idx;



#if TCG_TARGET_REG_BITS == 32

    if (type != TCG_TYPE_I32)

        tcg_abort();

#endif

    if (tcg_regset_test_reg(s->reserved_regs, reg))

        tcg_abort();

    idx = s->nb_globals;

    tcg_temp_alloc(s, s->nb_globals + 1);

    ts = &s->temps[s->nb_globals];

    ts->base_type = type;

    ts->type = type;

    ts->fixed_reg = 1;

    ts->reg = reg;

    ts->name = name;

    s->nb_globals++;

    tcg_regset_set_reg(s->reserved_regs, reg);

    return idx;

}
