static inline void gen_op_mfspr (DisasContext *ctx)

{

    void (*read_cb)(void *opaque, int sprn);

    uint32_t sprn = SPR(ctx->opcode);



#if !defined(CONFIG_USER_ONLY)

    if (ctx->supervisor)

        read_cb = ctx->spr_cb[sprn].oea_read;

    else

#endif

        read_cb = ctx->spr_cb[sprn].uea_read;

    if (likely(read_cb != NULL)) {

        if (likely(read_cb != SPR_NOACCESS)) {

            (*read_cb)(ctx, sprn);

            gen_op_store_T0_gpr(rD(ctx->opcode));

        } else {

            /* Privilege exception */

            if (loglevel != 0) {

                fprintf(logfile, "Trying to read privileged spr %d %03x\n",

                        sprn, sprn);

            }

            printf("Trying to read privileged spr %d %03x\n", sprn, sprn);

            RET_PRIVREG(ctx);

        }

    } else {

        /* Not defined */

        if (loglevel != 0) {

            fprintf(logfile, "Trying to read invalid spr %d %03x\n",

                    sprn, sprn);

        }

        printf("Trying to read invalid spr %d %03x\n", sprn, sprn);

        RET_EXCP(ctx, EXCP_PROGRAM, EXCP_INVAL | EXCP_INVAL_SPR);

    }

}
