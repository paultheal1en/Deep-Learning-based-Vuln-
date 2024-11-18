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

            return gen_excp(ctx, EXCP_CALL_PAL, palcode & 0xbf);

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

            return gen_excp(ctx, EXCP_CALL_PAL, palcode & 0x3f);

        }

        return NO_EXIT;

    }

#endif



    return gen_invalid(ctx);

}
