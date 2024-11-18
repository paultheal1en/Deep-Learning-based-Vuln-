static inline void gen_bcond(DisasContext *ctx, int type)

{

    uint32_t bo = BO(ctx->opcode);

    int l1;

    TCGv target;



    ctx->exception = POWERPC_EXCP_BRANCH;

    if (type == BCOND_LR || type == BCOND_CTR || type == BCOND_TAR) {

        target = tcg_temp_local_new();

        if (type == BCOND_CTR)

            tcg_gen_mov_tl(target, cpu_ctr);

        else if (type == BCOND_TAR)

            gen_load_spr(target, SPR_TAR);

        else

            tcg_gen_mov_tl(target, cpu_lr);

    } else {

        TCGV_UNUSED(target);

    }

    if (LK(ctx->opcode))

        gen_setlr(ctx, ctx->nip);

    l1 = gen_new_label();

    if ((bo & 0x4) == 0) {

        /* Decrement and test CTR */

        TCGv temp = tcg_temp_new();

        if (unlikely(type == BCOND_CTR)) {

            gen_inval_exception(ctx, POWERPC_EXCP_INVAL_INVAL);

            return;

        }

        tcg_gen_subi_tl(cpu_ctr, cpu_ctr, 1);

        if (NARROW_MODE(ctx)) {

            tcg_gen_ext32u_tl(temp, cpu_ctr);

        } else {

            tcg_gen_mov_tl(temp, cpu_ctr);

        }

        if (bo & 0x2) {

            tcg_gen_brcondi_tl(TCG_COND_NE, temp, 0, l1);

        } else {

            tcg_gen_brcondi_tl(TCG_COND_EQ, temp, 0, l1);

        }

        tcg_temp_free(temp);

    }

    if ((bo & 0x10) == 0) {

        /* Test CR */

        uint32_t bi = BI(ctx->opcode);

        uint32_t mask = 1 << (3 - (bi & 0x03));

        TCGv_i32 temp = tcg_temp_new_i32();



        if (bo & 0x8) {

            tcg_gen_andi_i32(temp, cpu_crf[bi >> 2], mask);

            tcg_gen_brcondi_i32(TCG_COND_EQ, temp, 0, l1);

        } else {

            tcg_gen_andi_i32(temp, cpu_crf[bi >> 2], mask);

            tcg_gen_brcondi_i32(TCG_COND_NE, temp, 0, l1);

        }

        tcg_temp_free_i32(temp);

    }

    gen_update_cfar(ctx, ctx->nip);

    if (type == BCOND_IM) {

        target_ulong li = (target_long)((int16_t)(BD(ctx->opcode)));

        if (likely(AA(ctx->opcode) == 0)) {

            gen_goto_tb(ctx, 0, ctx->nip + li - 4);

        } else {

            gen_goto_tb(ctx, 0, li);

        }

        gen_set_label(l1);

        gen_goto_tb(ctx, 1, ctx->nip);

    } else {

        if (NARROW_MODE(ctx)) {

            tcg_gen_andi_tl(cpu_nip, target, (uint32_t)~3);

        } else {

            tcg_gen_andi_tl(cpu_nip, target, ~3);

        }

        tcg_gen_exit_tb(0);

        gen_set_label(l1);

        gen_update_nip(ctx, ctx->nip);

        tcg_gen_exit_tb(0);

    }

    if (type == BCOND_LR || type == BCOND_CTR) {

        tcg_temp_free(target);

    }

}
