static void kvm_cpu_fill_host(x86_def_t *x86_cpu_def)

{

#ifdef CONFIG_KVM

    KVMState *s = kvm_state;

    uint32_t eax = 0, ebx = 0, ecx = 0, edx = 0;



    assert(kvm_enabled());



    x86_cpu_def->name = "host";


    host_cpuid(0x0, 0, &eax, &ebx, &ecx, &edx);

    x86_cpu_vendor_words2str(x86_cpu_def->vendor, ebx, edx, ecx);



    host_cpuid(0x1, 0, &eax, &ebx, &ecx, &edx);

    x86_cpu_def->family = ((eax >> 8) & 0x0F) + ((eax >> 20) & 0xFF);

    x86_cpu_def->model = ((eax >> 4) & 0x0F) | ((eax & 0xF0000) >> 12);

    x86_cpu_def->stepping = eax & 0x0F;



    x86_cpu_def->level = kvm_arch_get_supported_cpuid(s, 0x0, 0, R_EAX);

    x86_cpu_def->features[FEAT_1_EDX] =

        kvm_arch_get_supported_cpuid(s, 0x1, 0, R_EDX);

    x86_cpu_def->features[FEAT_1_ECX] =

        kvm_arch_get_supported_cpuid(s, 0x1, 0, R_ECX);



    if (x86_cpu_def->level >= 7) {

        x86_cpu_def->features[FEAT_7_0_EBX] =

                    kvm_arch_get_supported_cpuid(s, 0x7, 0, R_EBX);

    } else {

        x86_cpu_def->features[FEAT_7_0_EBX] = 0;

    }



    x86_cpu_def->xlevel = kvm_arch_get_supported_cpuid(s, 0x80000000, 0, R_EAX);

    x86_cpu_def->features[FEAT_8000_0001_EDX] =

                kvm_arch_get_supported_cpuid(s, 0x80000001, 0, R_EDX);

    x86_cpu_def->features[FEAT_8000_0001_ECX] =

                kvm_arch_get_supported_cpuid(s, 0x80000001, 0, R_ECX);



    cpu_x86_fill_model_id(x86_cpu_def->model_id);



    /* Call Centaur's CPUID instruction. */

    if (!strcmp(x86_cpu_def->vendor, CPUID_VENDOR_VIA)) {

        host_cpuid(0xC0000000, 0, &eax, &ebx, &ecx, &edx);

        eax = kvm_arch_get_supported_cpuid(s, 0xC0000000, 0, R_EAX);

        if (eax >= 0xC0000001) {

            /* Support VIA max extended level */

            x86_cpu_def->xlevel2 = eax;

            host_cpuid(0xC0000001, 0, &eax, &ebx, &ecx, &edx);

            x86_cpu_def->features[FEAT_C000_0001_EDX] =

                    kvm_arch_get_supported_cpuid(s, 0xC0000001, 0, R_EDX);

        }

    }



    /* Other KVM-specific feature fields: */

    x86_cpu_def->features[FEAT_SVM] =

        kvm_arch_get_supported_cpuid(s, 0x8000000A, 0, R_EDX);

    x86_cpu_def->features[FEAT_KVM] =

        kvm_arch_get_supported_cpuid(s, KVM_CPUID_FEATURES, 0, R_EAX);



#endif /* CONFIG_KVM */

}