static void kvm_mce_broadcast_rest(CPUState *env)

{

    CPUState *cenv;

    int family, model, cpuver = env->cpuid_version;



    family = (cpuver >> 8) & 0xf;

    model = ((cpuver >> 12) & 0xf0) + ((cpuver >> 4) & 0xf);



    /* Broadcast MCA signal for processor version 06H_EH and above */

    if ((family == 6 && model >= 14) || family > 6) {

        for (cenv = first_cpu; cenv != NULL; cenv = cenv->next_cpu) {

            if (cenv == env) {

                continue;

            }

            kvm_inject_x86_mce(cenv, 1, MCI_STATUS_VAL | MCI_STATUS_UC,

                               MCG_STATUS_MCIP | MCG_STATUS_RIPV, 0, 0, 1);

        }

    }

}
