static void gen_branch(DisasContext *ctx, int insn_bytes)

{

    if (ctx->hflags & MIPS_HFLAG_BMASK) {

        int proc_hflags = ctx->hflags & MIPS_HFLAG_BMASK;

        /* Branches completion */

        ctx->hflags &= ~MIPS_HFLAG_BMASK;

        ctx->bstate = BS_BRANCH;

        save_cpu_state(ctx, 0);

        /* FIXME: Need to clear can_do_io.  */

        switch (proc_hflags & MIPS_HFLAG_BMASK_BASE) {

        case MIPS_HFLAG_FBNSLOT:

            MIPS_DEBUG("forbidden slot");

            gen_goto_tb(ctx, 0, ctx->pc + insn_bytes);

            break;

        case MIPS_HFLAG_B:

            /* unconditional branch */

            MIPS_DEBUG("unconditional branch");

            if (proc_hflags & MIPS_HFLAG_BX) {

                tcg_gen_xori_i32(hflags, hflags, MIPS_HFLAG_M16);

            }

            gen_goto_tb(ctx, 0, ctx->btarget);

            break;

        case MIPS_HFLAG_BL:

            /* blikely taken case */

            MIPS_DEBUG("blikely branch taken");

            gen_goto_tb(ctx, 0, ctx->btarget);

            break;

        case MIPS_HFLAG_BC:

            /* Conditional branch */

            MIPS_DEBUG("conditional branch");

            {

                TCGLabel *l1 = gen_new_label();



                tcg_gen_brcondi_tl(TCG_COND_NE, bcond, 0, l1);

                gen_goto_tb(ctx, 1, ctx->pc + insn_bytes);

                gen_set_label(l1);

                gen_goto_tb(ctx, 0, ctx->btarget);

            }

            break;

        case MIPS_HFLAG_BR:

            /* unconditional branch to register */

            MIPS_DEBUG("branch to register");

            if (ctx->insn_flags & (ASE_MIPS16 | ASE_MICROMIPS)) {

                TCGv t0 = tcg_temp_new();

                TCGv_i32 t1 = tcg_temp_new_i32();



                tcg_gen_andi_tl(t0, btarget, 0x1);

                tcg_gen_trunc_tl_i32(t1, t0);

                tcg_temp_free(t0);

                tcg_gen_andi_i32(hflags, hflags, ~(uint32_t)MIPS_HFLAG_M16);

                tcg_gen_shli_i32(t1, t1, MIPS_HFLAG_M16_SHIFT);

                tcg_gen_or_i32(hflags, hflags, t1);

                tcg_temp_free_i32(t1);



                tcg_gen_andi_tl(cpu_PC, btarget, ~(target_ulong)0x1);

            } else {

                tcg_gen_mov_tl(cpu_PC, btarget);

            }

            if (ctx->singlestep_enabled) {

                save_cpu_state(ctx, 0);

                gen_helper_0e0i(raise_exception, EXCP_DEBUG);

            }

            tcg_gen_exit_tb(0);

            break;

        default:

            MIPS_DEBUG("unknown branch");

            break;

        }

    }

}
