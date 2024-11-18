static int gen_set_psr(DisasContext *s, uint32_t mask, int spsr, TCGv t0)

{

    TCGv tmp;

    if (spsr) {

        /* ??? This is also undefined in system mode.  */

        if (IS_USER(s))

            return 1;



        tmp = load_cpu_field(spsr);

        tcg_gen_andi_i32(tmp, tmp, ~mask);

        tcg_gen_andi_i32(t0, t0, mask);

        tcg_gen_or_i32(tmp, tmp, t0);

        store_cpu_field(tmp, spsr);

    } else {

        gen_set_cpsr(t0, mask);

    }

    dead_tmp(t0);

    gen_lookup_tb(s);

    return 0;

}
