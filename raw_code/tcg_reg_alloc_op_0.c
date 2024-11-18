static void tcg_reg_alloc_op(TCGContext *s, 

                             const TCGOpDef *def, TCGOpcode opc,

                             const TCGArg *args, uint16_t dead_args,

                             uint8_t sync_args)

{

    TCGRegSet allocated_regs;

    int i, k, nb_iargs, nb_oargs, reg;

    TCGArg arg;

    const TCGArgConstraint *arg_ct;

    TCGTemp *ts;

    TCGArg new_args[TCG_MAX_OP_ARGS];

    int const_args[TCG_MAX_OP_ARGS];



    nb_oargs = def->nb_oargs;

    nb_iargs = def->nb_iargs;



    /* copy constants */

    memcpy(new_args + nb_oargs + nb_iargs, 

           args + nb_oargs + nb_iargs, 

           sizeof(TCGArg) * def->nb_cargs);



    /* satisfy input constraints */ 

    tcg_regset_set(allocated_regs, s->reserved_regs);

    for(k = 0; k < nb_iargs; k++) {

        i = def->sorted_args[nb_oargs + k];

        arg = args[i];

        arg_ct = &def->args_ct[i];

        ts = &s->temps[arg];

        if (ts->val_type == TEMP_VAL_MEM) {

            reg = tcg_reg_alloc(s, arg_ct->u.regs, allocated_regs);

            tcg_out_ld(s, ts->type, reg, ts->mem_reg, ts->mem_offset);

            ts->val_type = TEMP_VAL_REG;

            ts->reg = reg;

            ts->mem_coherent = 1;

            s->reg_to_temp[reg] = arg;

        } else if (ts->val_type == TEMP_VAL_CONST) {

            if (tcg_target_const_match(ts->val, ts->type, arg_ct)) {

                /* constant is OK for instruction */

                const_args[i] = 1;

                new_args[i] = ts->val;

                goto iarg_end;

            } else {

                /* need to move to a register */

                reg = tcg_reg_alloc(s, arg_ct->u.regs, allocated_regs);

                tcg_out_movi(s, ts->type, reg, ts->val);

                ts->val_type = TEMP_VAL_REG;

                ts->reg = reg;

                ts->mem_coherent = 0;

                s->reg_to_temp[reg] = arg;

            }

        }

        assert(ts->val_type == TEMP_VAL_REG);

        if (arg_ct->ct & TCG_CT_IALIAS) {

            if (ts->fixed_reg) {

                /* if fixed register, we must allocate a new register

                   if the alias is not the same register */

                if (arg != args[arg_ct->alias_index])

                    goto allocate_in_reg;

            } else {

                /* if the input is aliased to an output and if it is

                   not dead after the instruction, we must allocate

                   a new register and move it */

                if (!IS_DEAD_ARG(i)) {

                    goto allocate_in_reg;

                }

                /* check if the current register has already been allocated

                   for another input aliased to an output */

                int k2, i2;

                for (k2 = 0 ; k2 < k ; k2++) {

                    i2 = def->sorted_args[nb_oargs + k2];

                    if ((def->args_ct[i2].ct & TCG_CT_IALIAS) &&

                        (new_args[i2] == ts->reg)) {

                        goto allocate_in_reg;

                    }

                }

            }

        }

        reg = ts->reg;

        if (tcg_regset_test_reg(arg_ct->u.regs, reg)) {

            /* nothing to do : the constraint is satisfied */

        } else {

        allocate_in_reg:

            /* allocate a new register matching the constraint 

               and move the temporary register into it */

            reg = tcg_reg_alloc(s, arg_ct->u.regs, allocated_regs);

            tcg_out_mov(s, ts->type, reg, ts->reg);

        }

        new_args[i] = reg;

        const_args[i] = 0;

        tcg_regset_set_reg(allocated_regs, reg);

    iarg_end: ;

    }

    

    /* mark dead temporaries and free the associated registers */

    for (i = nb_oargs; i < nb_oargs + nb_iargs; i++) {

        if (IS_DEAD_ARG(i)) {

            temp_dead(s, args[i]);

        }

    }



    if (def->flags & TCG_OPF_BB_END) {

        tcg_reg_alloc_bb_end(s, allocated_regs);

    } else {

        if (def->flags & TCG_OPF_CALL_CLOBBER) {

            /* XXX: permit generic clobber register list ? */ 

            for(reg = 0; reg < TCG_TARGET_NB_REGS; reg++) {

                if (tcg_regset_test_reg(tcg_target_call_clobber_regs, reg)) {

                    tcg_reg_free(s, reg);

                }

            }

        }

        if (def->flags & TCG_OPF_SIDE_EFFECTS) {

            /* sync globals if the op has side effects and might trigger

               an exception. */

            sync_globals(s, allocated_regs);

        }

        

        /* satisfy the output constraints */

        tcg_regset_set(allocated_regs, s->reserved_regs);

        for(k = 0; k < nb_oargs; k++) {

            i = def->sorted_args[k];

            arg = args[i];

            arg_ct = &def->args_ct[i];

            ts = &s->temps[arg];

            if (arg_ct->ct & TCG_CT_ALIAS) {

                reg = new_args[arg_ct->alias_index];

            } else {

                /* if fixed register, we try to use it */

                reg = ts->reg;

                if (ts->fixed_reg &&

                    tcg_regset_test_reg(arg_ct->u.regs, reg)) {

                    goto oarg_end;

                }

                reg = tcg_reg_alloc(s, arg_ct->u.regs, allocated_regs);

            }

            tcg_regset_set_reg(allocated_regs, reg);

            /* if a fixed register is used, then a move will be done afterwards */

            if (!ts->fixed_reg) {

                if (ts->val_type == TEMP_VAL_REG) {

                    s->reg_to_temp[ts->reg] = -1;

                }

                ts->val_type = TEMP_VAL_REG;

                ts->reg = reg;

                /* temp value is modified, so the value kept in memory is

                   potentially not the same */

                ts->mem_coherent = 0;

                s->reg_to_temp[reg] = arg;

            }

        oarg_end:

            new_args[i] = reg;

        }

    }



    /* emit instruction */

    tcg_out_op(s, opc, new_args, const_args);

    

    /* move the outputs in the correct register if needed */

    for(i = 0; i < nb_oargs; i++) {

        ts = &s->temps[args[i]];

        reg = new_args[i];

        if (ts->fixed_reg && ts->reg != reg) {

            tcg_out_mov(s, ts->type, ts->reg, reg);

        }

        if (NEED_SYNC_ARG(i)) {

            tcg_reg_sync(s, reg);

        }

        if (IS_DEAD_ARG(i)) {

            temp_dead(s, args[i]);

        }

    }

}
