static void spr_read_tbl (DisasContext *ctx, int gprn, int sprn)

{

    if (use_icount) {

        gen_io_start();

    }

    gen_helper_load_tbl(cpu_gpr[gprn], cpu_env);

    if (use_icount) {

        gen_io_end();

        gen_stop_exception(ctx);

    }

}
