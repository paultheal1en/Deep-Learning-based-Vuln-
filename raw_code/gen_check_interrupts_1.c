static void gen_check_interrupts(DisasContext *dc)

{

    if (dc->tb->cflags & CF_USE_ICOUNT) {

        gen_io_start();

    }

    gen_helper_check_interrupts(cpu_env);

    if (dc->tb->cflags & CF_USE_ICOUNT) {

        gen_io_end();

    }

}
