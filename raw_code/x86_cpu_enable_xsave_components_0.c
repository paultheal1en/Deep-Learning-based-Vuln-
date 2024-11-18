static void x86_cpu_enable_xsave_components(X86CPU *cpu)

{

    CPUX86State *env = &cpu->env;

    int i;



    if (!(env->features[FEAT_1_ECX] & CPUID_EXT_XSAVE)) {

        return;

    }



    env->xsave_components = (XSTATE_FP_MASK | XSTATE_SSE_MASK);

    for (i = 2; i < ARRAY_SIZE(x86_ext_save_areas); i++) {

        const ExtSaveArea *esa = &x86_ext_save_areas[i];

        if (env->features[esa->feature] & esa->bits) {

            env->xsave_components |= (1ULL << i);

        }

    }



    if (kvm_enabled()) {

        KVMState *s = kvm_state;

        uint64_t kvm_mask = kvm_arch_get_supported_cpuid(s, 0xd, 0, R_EDX);

        kvm_mask <<= 32;

        kvm_mask |= kvm_arch_get_supported_cpuid(s, 0xd, 0, R_EAX);

        env->xsave_components &= kvm_mask;

    }

}
