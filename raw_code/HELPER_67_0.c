void HELPER(wsr_ibreakenable)(uint32_t v)

{

    uint32_t change = v ^ env->sregs[IBREAKENABLE];

    unsigned i;



    for (i = 0; i < env->config->nibreak; ++i) {

        if (change & (1 << i)) {

            tb_invalidate_phys_page_range(

                    env->sregs[IBREAKA + i], env->sregs[IBREAKA + i] + 1, 0);

        }

    }

    env->sregs[IBREAKENABLE] = v & ((1 << env->config->nibreak) - 1);

}
