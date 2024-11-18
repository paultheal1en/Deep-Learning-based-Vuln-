void HELPER(wsr_lend)(uint32_t v)

{

    if (env->sregs[LEND] != v) {

        tb_invalidate_phys_page_range(

                env->sregs[LEND] - 1, env->sregs[LEND], 0);

        env->sregs[LEND] = v;

        tb_invalidate_phys_page_range(

                env->sregs[LEND] - 1, env->sregs[LEND], 0);

    }

}
