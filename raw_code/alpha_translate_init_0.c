static void alpha_translate_init(void)

{

    int i;

    char *p;

    static int done_init = 0;



    if (done_init)

        return;



    cpu_env = tcg_global_reg_new(TCG_TYPE_PTR, TCG_AREG0, "env");



    p = cpu_reg_names;

    for (i = 0; i < 31; i++) {

        sprintf(p, "ir%d", i);

        cpu_ir[i] = tcg_global_mem_new(TCG_TYPE_I64, TCG_AREG0,

                                       offsetof(CPUState, ir[i]), p);

        p += (i < 10) ? 4 : 5;



        sprintf(p, "fir%d", i);

        cpu_fir[i] = tcg_global_mem_new(TCG_TYPE_I64, TCG_AREG0,

                                        offsetof(CPUState, fir[i]), p);

        p += (i < 10) ? 5 : 6;

    }



    cpu_pc = tcg_global_mem_new(TCG_TYPE_I64, TCG_AREG0,

                                offsetof(CPUState, pc), "pc");



    cpu_lock = tcg_global_mem_new(TCG_TYPE_I64, TCG_AREG0,

                                  offsetof(CPUState, lock), "lock");



    /* register helpers */

#undef DEF_HELPER

#define DEF_HELPER(ret, name, params) tcg_register_helper(name, #name);

#include "helper.h"



    done_init = 1;

}
