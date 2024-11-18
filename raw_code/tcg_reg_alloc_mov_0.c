static void tcg_reg_alloc_mov(TCGContext *s, const TCGOpDef *def,

                              const TCGArg *args, uint16_t dead_args,

                              uint8_t sync_args)

{

    TCGRegSet allocated_regs;

    TCGTemp *ts, *ots;

    TCGType otype, itype;



    tcg_regset_set(allocated_regs, s->reserved_regs);

    ots = &s->temps[args[0]];

    ts = &s->temps[args[1]];



    /* Note that otype != itype for no-op truncation.  */

    otype = ots->type;

    itype = ts->type;



    /* If the source value is not in a register, and we're going to be

       forced to have it in a register in order to perform the copy,

       then copy the SOURCE value into its own register first.  That way

       we don't have to reload SOURCE the next time it is used. */

    if (((NEED_SYNC_ARG(0) || ots->fixed_reg) && ts->val_type != TEMP_VAL_REG)

        || ts->val_type == TEMP_VAL_MEM) {

        temp_load(s, ts, tcg_target_available_regs[itype], allocated_regs);

    }



    if (IS_DEAD_ARG(0) && !ots->fixed_reg) {

        /* mov to a non-saved dead register makes no sense (even with

           liveness analysis disabled). */

        assert(NEED_SYNC_ARG(0));

        /* The code above should have moved the temp to a register. */

        assert(ts->val_type == TEMP_VAL_REG);

        if (!ots->mem_allocated) {

            temp_allocate_frame(s, args[0]);

        }

        if (ots->indirect_reg) {

            tcg_regset_set_reg(allocated_regs, ts->reg);

            temp_load(s, ots->mem_base,

                      tcg_target_available_regs[TCG_TYPE_PTR],

                      allocated_regs);

        }

        tcg_out_st(s, otype, ts->reg, ots->mem_base->reg, ots->mem_offset);

        if (IS_DEAD_ARG(1)) {

            temp_dead(s, ts);

        }

        temp_dead(s, ots);

    } else if (ts->val_type == TEMP_VAL_CONST) {

        /* propagate constant */

        if (ots->val_type == TEMP_VAL_REG) {

            s->reg_to_temp[ots->reg] = NULL;

        }

        ots->val_type = TEMP_VAL_CONST;

        ots->val = ts->val;

        if (IS_DEAD_ARG(1)) {

            temp_dead(s, ts);

        }

    } else {

        /* The code in the first if block should have moved the

           temp to a register. */

        assert(ts->val_type == TEMP_VAL_REG);

        if (IS_DEAD_ARG(1) && !ts->fixed_reg && !ots->fixed_reg) {

            /* the mov can be suppressed */

            if (ots->val_type == TEMP_VAL_REG) {

                s->reg_to_temp[ots->reg] = NULL;

            }

            ots->reg = ts->reg;

            temp_dead(s, ts);

        } else {

            if (ots->val_type != TEMP_VAL_REG) {

                /* When allocating a new register, make sure to not spill the

                   input one. */

                tcg_regset_set_reg(allocated_regs, ts->reg);

                ots->reg = tcg_reg_alloc(s, tcg_target_available_regs[otype],

                                         allocated_regs, ots->indirect_base);

            }

            tcg_out_mov(s, otype, ots->reg, ts->reg);

        }

        ots->val_type = TEMP_VAL_REG;

        ots->mem_coherent = 0;

        s->reg_to_temp[ots->reg] = ots;

        if (NEED_SYNC_ARG(0)) {

            tcg_reg_sync(s, ots->reg, allocated_regs);

        }

    }

}
