static void gen_or(DisasContext *ctx)

{

    int rs, ra, rb;



    rs = rS(ctx->opcode);

    ra = rA(ctx->opcode);

    rb = rB(ctx->opcode);

    /* Optimisation for mr. ri case */

    if (rs != ra || rs != rb) {

        if (rs != rb)

            tcg_gen_or_tl(cpu_gpr[ra], cpu_gpr[rs], cpu_gpr[rb]);

        else

            tcg_gen_mov_tl(cpu_gpr[ra], cpu_gpr[rs]);

        if (unlikely(Rc(ctx->opcode) != 0))

            gen_set_Rc0(ctx, cpu_gpr[ra]);

    } else if (unlikely(Rc(ctx->opcode) != 0)) {

        gen_set_Rc0(ctx, cpu_gpr[rs]);

#if defined(TARGET_PPC64)

    } else {

        int prio = 0;



        switch (rs) {

        case 1:

            /* Set process priority to low */

            prio = 2;

            break;

        case 6:

            /* Set process priority to medium-low */

            prio = 3;

            break;

        case 2:

            /* Set process priority to normal */

            prio = 4;

            break;

#if !defined(CONFIG_USER_ONLY)

        case 31:

            if (!ctx->pr) {

                /* Set process priority to very low */

                prio = 1;

            }

            break;

        case 5:

            if (!ctx->pr) {

                /* Set process priority to medium-hight */

                prio = 5;

            }

            break;

        case 3:

            if (!ctx->pr) {

                /* Set process priority to high */

                prio = 6;

            }

            break;

        case 7:

            if (ctx->hv) {

                /* Set process priority to very high */

                prio = 7;

            }

            break;

#endif

        default:

            /* nop */

            break;

        }

        if (prio) {

            TCGv t0 = tcg_temp_new();

            gen_load_spr(t0, SPR_PPR);

            tcg_gen_andi_tl(t0, t0, ~0x001C000000000000ULL);

            tcg_gen_ori_tl(t0, t0, ((uint64_t)prio) << 50);

            gen_store_spr(SPR_PPR, t0);

            tcg_temp_free(t0);

        }

#endif

    }

}
