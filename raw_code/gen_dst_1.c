static void gen_dst(DisasContext *ctx)

{

    if (rA(ctx->opcode) == 0) {

        gen_inval_exception(ctx, POWERPC_EXCP_INVAL_LSWX);

    } else {

        /* interpreted as no-op */

    }

}
