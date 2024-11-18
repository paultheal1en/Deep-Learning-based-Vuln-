static inline void gen_op_mfspr(DisasContext *ctx)

{

    void (*read_cb)(void *opaque, int gprn, int sprn);

    uint32_t sprn = SPR(ctx->opcode);



#if !defined(CONFIG_USER_ONLY)

    if (ctx->mem_idx == 2)

        read_cb = ctx->spr_cb[sprn].hea_read;

    else if (ctx->mem_idx)

        read_cb = ctx->spr_cb[sprn].oea_read;

    else

#endif

        read_cb = ctx->spr_cb[sprn].uea_read;

    if (likely(read_cb != NULL)) {

        if (likely(read_cb != SPR_NOACCESS)) {

            (*read_cb)(ctx, rD(ctx->opcode), sprn);

        } else {

            /* Privilege exception */

            /* This is a hack to avoid warnings when running Linux:

             * this OS breaks the PowerPC virtualisation model,

             * allowing userland application to read the PVR

             */

            if (sprn != SPR_PVR) {

                qemu_log("Trying to read privileged spr %d %03x at "

                         TARGET_FMT_lx "\n", sprn, sprn, ctx->nip);

                printf("Trying to read privileged spr %d %03x at "

                       TARGET_FMT_lx "\n", sprn, sprn, ctx->nip);

            }

            gen_inval_exception(ctx, POWERPC_EXCP_PRIV_REG);

        }

    } else {

        /* Not defined */

        qemu_log("Trying to read invalid spr %d %03x at "

                    TARGET_FMT_lx "\n", sprn, sprn, ctx->nip);

        printf("Trying to read invalid spr %d %03x at " TARGET_FMT_lx "\n",

               sprn, sprn, ctx->nip);

        gen_inval_exception(ctx, POWERPC_EXCP_INVAL_SPR);

    }

}
