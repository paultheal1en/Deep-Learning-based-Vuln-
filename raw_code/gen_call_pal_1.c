static ExitStatus gen_call_pal(DisasContext *ctx, int palcode)

{

    /* We're emulating OSF/1 PALcode.  Many of these are trivial access

       to internal cpu registers.  */



    /* Unprivileged PAL call */

    if (palcode >= 0x80 && palcode < 0xC0) {

        switch (palcode) {

        case 0x86:

            /* IMB */

            /* No-op inside QEMU.  */

            break;

        case 0x9E:

            /* RDUNIQUE */

            tcg_gen_mov_i64(cpu_ir[IR_V0], cpu_unique);

            break;

        case 0x9F:

            /* WRUNIQUE */

            tcg_gen_mov_i64(cpu_unique, cpu_ir[IR_A0]);

            break;

        default:

            palcode &= 0xbf;

            goto do_call_pal;

        }

        return NO_EXIT;

    }



#ifndef CONFIG_USER_ONLY

    /* Privileged PAL code */

    if (palcode < 0x40 && (ctx->tb->flags & TB_FLAGS_USER_MODE) == 0) {

        switch (palcode) {

        case 0x01:

            /* CFLUSH */

            /* No-op inside QEMU.  */

            break;

        case 0x02:

            /* DRAINA */

            /* No-op inside QEMU.  */

            break;

        case 0x2D:

            /* WRVPTPTR */

            tcg_gen_st_i64(cpu_ir[IR_A0], cpu_env, offsetof(CPUAlphaState, vptptr));

            break;

        case 0x31:

            /* WRVAL */

            tcg_gen_mov_i64(cpu_sysval, cpu_ir[IR_A0]);

            break;

        case 0x32:

            /* RDVAL */

            tcg_gen_mov_i64(cpu_ir[IR_V0], cpu_sysval);

            break;



        case 0x35: {

            /* SWPIPL */

            TCGv tmp;



            /* Note that we already know we're in kernel mode, so we know

               that PS only contains the 3 IPL bits.  */

            tcg_gen_ld8u_i64(cpu_ir[IR_V0], cpu_env, offsetof(CPUAlphaState, ps));



            /* But make sure and store only the 3 IPL bits from the user.  */

            tmp = tcg_temp_new();

            tcg_gen_andi_i64(tmp, cpu_ir[IR_A0], PS_INT_MASK);

            tcg_gen_st8_i64(tmp, cpu_env, offsetof(CPUAlphaState, ps));

            tcg_temp_free(tmp);

            break;

        }



        case 0x36:

            /* RDPS */

            tcg_gen_ld8u_i64(cpu_ir[IR_V0], cpu_env, offsetof(CPUAlphaState, ps));

            break;

        case 0x38:

            /* WRUSP */

            tcg_gen_mov_i64(cpu_usp, cpu_ir[IR_A0]);

            break;

        case 0x3A:

            /* RDUSP */

            tcg_gen_mov_i64(cpu_ir[IR_V0], cpu_usp);

            break;

        case 0x3C:

            /* WHAMI */

            tcg_gen_ld32s_i64(cpu_ir[IR_V0], cpu_env,

                -offsetof(AlphaCPU, env) + offsetof(CPUState, cpu_index));

            break;



        default:

            palcode &= 0x3f;

            goto do_call_pal;

        }

        return NO_EXIT;

    }

#endif

    return gen_invalid(ctx);



 do_call_pal:

#ifdef CONFIG_USER_ONLY

    return gen_excp(ctx, EXCP_CALL_PAL, palcode);

#else

    {

        TCGv pc = tcg_const_i64(ctx->pc);

        TCGv entry = tcg_const_i64(palcode & 0x80

                                   ? 0x2000 + (palcode - 0x80) * 64

                                   : 0x1000 + palcode * 64);



        gen_helper_call_pal(cpu_env, pc, entry);



        tcg_temp_free(entry);

        tcg_temp_free(pc);



        /* Since the destination is running in PALmode, we don't really

           need the page permissions check.  We'll see the existence of

           the page when we create the TB, and we'll flush all TBs if

           we change the PAL base register.  */

        if (!ctx->singlestep_enabled && !(ctx->tb->cflags & CF_LAST_IO)) {

            tcg_gen_goto_tb(0);

            tcg_gen_exit_tb((uintptr_t)ctx->tb);

            return EXIT_GOTO_TB;

        }



        return EXIT_PC_UPDATED;

    }

#endif

}
