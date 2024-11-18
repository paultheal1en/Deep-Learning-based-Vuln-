void HELPER(wsr_lbeg)(uint32_t v)

{

    if (env->sregs[LBEG] != v) {

        tb_invalidate_phys_page_range(

                env->sregs[LEND] - 1, env->sregs[LEND], 0);

        env->sregs[LBEG] = v;

    }

}
