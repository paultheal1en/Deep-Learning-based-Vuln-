uint32_t kvm_arch_get_supported_cpuid(CPUState *env, uint32_t function,

                                      uint32_t index, int reg)

{

    struct kvm_cpuid2 *cpuid;

    int i, max;

    uint32_t ret = 0;

    uint32_t cpuid_1_edx;



    if (!kvm_check_extension(env->kvm_state, KVM_CAP_EXT_CPUID)) {

        return -1U;

    }



    max = 1;

    while ((cpuid = try_get_cpuid(env->kvm_state, max)) == NULL) {

        max *= 2;

    }



    for (i = 0; i < cpuid->nent; ++i) {

        if (cpuid->entries[i].function == function &&

            cpuid->entries[i].index == index) {

            switch (reg) {

            case R_EAX:

                ret = cpuid->entries[i].eax;

                break;

            case R_EBX:

                ret = cpuid->entries[i].ebx;

                break;

            case R_ECX:

                ret = cpuid->entries[i].ecx;

                break;

            case R_EDX:

                ret = cpuid->entries[i].edx;

                switch (function) {

                case 1:

                    /* KVM before 2.6.30 misreports the following features */

                    ret |= CPUID_MTRR | CPUID_PAT | CPUID_MCE | CPUID_MCA;

                    break;

                case 0x80000001:

                    /* On Intel, kvm returns cpuid according to the Intel spec,

                     * so add missing bits according to the AMD spec:

                     */

                    cpuid_1_edx = kvm_arch_get_supported_cpuid(env, 1, 0, R_EDX);

                    ret |= cpuid_1_edx & 0x183f7ff;

                    break;

                }

                break;

            }

        }

    }



    qemu_free(cpuid);



    return ret;

}
