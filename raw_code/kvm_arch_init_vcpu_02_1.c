int kvm_arch_init_vcpu(CPUState *cs)

{

    struct {

        struct kvm_cpuid2 cpuid;

        struct kvm_cpuid_entry2 entries[KVM_MAX_CPUID_ENTRIES];

    } QEMU_PACKED cpuid_data;

    X86CPU *cpu = X86_CPU(cs);

    CPUX86State *env = &cpu->env;

    uint32_t limit, i, j, cpuid_i;

    uint32_t unused;

    struct kvm_cpuid_entry2 *c;

    uint32_t signature[3];

    int r;



    cpuid_i = 0;



    /* Paravirtualization CPUIDs */

    c = &cpuid_data.entries[cpuid_i++];

    memset(c, 0, sizeof(*c));

    c->function = KVM_CPUID_SIGNATURE;

    if (!hyperv_enabled(cpu)) {

        memcpy(signature, "KVMKVMKVM\0\0\0", 12);

        c->eax = 0;

    } else {

        memcpy(signature, "Microsoft Hv", 12);

        c->eax = HYPERV_CPUID_MIN;

    }

    c->ebx = signature[0];

    c->ecx = signature[1];

    c->edx = signature[2];



    c = &cpuid_data.entries[cpuid_i++];

    memset(c, 0, sizeof(*c));

    c->function = KVM_CPUID_FEATURES;

    c->eax = env->features[FEAT_KVM];



    if (hyperv_enabled(cpu)) {

        memcpy(signature, "Hv#1\0\0\0\0\0\0\0\0", 12);

        c->eax = signature[0];



        c = &cpuid_data.entries[cpuid_i++];

        memset(c, 0, sizeof(*c));

        c->function = HYPERV_CPUID_VERSION;

        c->eax = 0x00001bbc;

        c->ebx = 0x00060001;



        c = &cpuid_data.entries[cpuid_i++];

        memset(c, 0, sizeof(*c));

        c->function = HYPERV_CPUID_FEATURES;

        if (cpu->hyperv_relaxed_timing) {

            c->eax |= HV_X64_MSR_HYPERCALL_AVAILABLE;

        }

        if (cpu->hyperv_vapic) {

            c->eax |= HV_X64_MSR_HYPERCALL_AVAILABLE;

            c->eax |= HV_X64_MSR_APIC_ACCESS_AVAILABLE;

        }



        c = &cpuid_data.entries[cpuid_i++];

        memset(c, 0, sizeof(*c));

        c->function = HYPERV_CPUID_ENLIGHTMENT_INFO;

        if (cpu->hyperv_relaxed_timing) {

            c->eax |= HV_X64_RELAXED_TIMING_RECOMMENDED;

        }

        if (cpu->hyperv_vapic) {

            c->eax |= HV_X64_APIC_ACCESS_RECOMMENDED;

        }

        c->ebx = cpu->hyperv_spinlock_attempts;



        c = &cpuid_data.entries[cpuid_i++];

        memset(c, 0, sizeof(*c));

        c->function = HYPERV_CPUID_IMPLEMENT_LIMITS;

        c->eax = 0x40;

        c->ebx = 0x40;



        c = &cpuid_data.entries[cpuid_i++];

        memset(c, 0, sizeof(*c));

        c->function = KVM_CPUID_SIGNATURE_NEXT;

        memcpy(signature, "KVMKVMKVM\0\0\0", 12);

        c->eax = 0;

        c->ebx = signature[0];

        c->ecx = signature[1];

        c->edx = signature[2];

    }



    has_msr_async_pf_en = c->eax & (1 << KVM_FEATURE_ASYNC_PF);



    has_msr_pv_eoi_en = c->eax & (1 << KVM_FEATURE_PV_EOI);



    has_msr_kvm_steal_time = c->eax & (1 << KVM_FEATURE_STEAL_TIME);



    cpu_x86_cpuid(env, 0, 0, &limit, &unused, &unused, &unused);



    for (i = 0; i <= limit; i++) {

        if (cpuid_i == KVM_MAX_CPUID_ENTRIES) {

            fprintf(stderr, "unsupported level value: 0x%x\n", limit);

            abort();

        }

        c = &cpuid_data.entries[cpuid_i++];



        switch (i) {

        case 2: {

            /* Keep reading function 2 till all the input is received */

            int times;



            c->function = i;

            c->flags = KVM_CPUID_FLAG_STATEFUL_FUNC |

                       KVM_CPUID_FLAG_STATE_READ_NEXT;

            cpu_x86_cpuid(env, i, 0, &c->eax, &c->ebx, &c->ecx, &c->edx);

            times = c->eax & 0xff;



            for (j = 1; j < times; ++j) {

                if (cpuid_i == KVM_MAX_CPUID_ENTRIES) {

                    fprintf(stderr, "cpuid_data is full, no space for "

                            "cpuid(eax:2):eax & 0xf = 0x%x\n", times);

                    abort();

                }

                c = &cpuid_data.entries[cpuid_i++];

                c->function = i;

                c->flags = KVM_CPUID_FLAG_STATEFUL_FUNC;

                cpu_x86_cpuid(env, i, 0, &c->eax, &c->ebx, &c->ecx, &c->edx);

            }

            break;

        }

        case 4:

        case 0xb:

        case 0xd:

            for (j = 0; ; j++) {

                if (i == 0xd && j == 64) {

                    break;

                }

                c->function = i;

                c->flags = KVM_CPUID_FLAG_SIGNIFCANT_INDEX;

                c->index = j;

                cpu_x86_cpuid(env, i, j, &c->eax, &c->ebx, &c->ecx, &c->edx);



                if (i == 4 && c->eax == 0) {

                    break;

                }

                if (i == 0xb && !(c->ecx & 0xff00)) {

                    break;

                }

                if (i == 0xd && c->eax == 0) {

                    continue;

                }

                if (cpuid_i == KVM_MAX_CPUID_ENTRIES) {

                    fprintf(stderr, "cpuid_data is full, no space for "

                            "cpuid(eax:0x%x,ecx:0x%x)\n", i, j);

                    abort();

                }

                c = &cpuid_data.entries[cpuid_i++];

            }

            break;

        default:

            c->function = i;

            c->flags = 0;

            cpu_x86_cpuid(env, i, 0, &c->eax, &c->ebx, &c->ecx, &c->edx);

            break;

        }

    }



    if (limit >= 0x0a) {

        uint32_t ver;



        cpu_x86_cpuid(env, 0x0a, 0, &ver, &unused, &unused, &unused);

        if ((ver & 0xff) > 0) {

            has_msr_architectural_pmu = true;

            num_architectural_pmu_counters = (ver & 0xff00) >> 8;



            /* Shouldn't be more than 32, since that's the number of bits

             * available in EBX to tell us _which_ counters are available.

             * Play it safe.

             */

            if (num_architectural_pmu_counters > MAX_GP_COUNTERS) {

                num_architectural_pmu_counters = MAX_GP_COUNTERS;

            }

        }

    }



    cpu_x86_cpuid(env, 0x80000000, 0, &limit, &unused, &unused, &unused);



    for (i = 0x80000000; i <= limit; i++) {

        if (cpuid_i == KVM_MAX_CPUID_ENTRIES) {

            fprintf(stderr, "unsupported xlevel value: 0x%x\n", limit);

            abort();

        }

        c = &cpuid_data.entries[cpuid_i++];



        c->function = i;

        c->flags = 0;

        cpu_x86_cpuid(env, i, 0, &c->eax, &c->ebx, &c->ecx, &c->edx);

    }



    /* Call Centaur's CPUID instructions they are supported. */

    if (env->cpuid_xlevel2 > 0) {

        cpu_x86_cpuid(env, 0xC0000000, 0, &limit, &unused, &unused, &unused);



        for (i = 0xC0000000; i <= limit; i++) {

            if (cpuid_i == KVM_MAX_CPUID_ENTRIES) {

                fprintf(stderr, "unsupported xlevel2 value: 0x%x\n", limit);

                abort();

            }

            c = &cpuid_data.entries[cpuid_i++];



            c->function = i;

            c->flags = 0;

            cpu_x86_cpuid(env, i, 0, &c->eax, &c->ebx, &c->ecx, &c->edx);

        }

    }



    cpuid_data.cpuid.nent = cpuid_i;



    if (((env->cpuid_version >> 8)&0xF) >= 6

        && (env->features[FEAT_1_EDX] & (CPUID_MCE | CPUID_MCA)) ==

           (CPUID_MCE | CPUID_MCA)

        && kvm_check_extension(cs->kvm_state, KVM_CAP_MCE) > 0) {

        uint64_t mcg_cap;

        int banks;

        int ret;



        ret = kvm_get_mce_cap_supported(cs->kvm_state, &mcg_cap, &banks);

        if (ret < 0) {

            fprintf(stderr, "kvm_get_mce_cap_supported: %s", strerror(-ret));

            return ret;

        }



        if (banks > MCE_BANKS_DEF) {

            banks = MCE_BANKS_DEF;

        }

        mcg_cap &= MCE_CAP_DEF;

        mcg_cap |= banks;

        ret = kvm_vcpu_ioctl(cs, KVM_X86_SETUP_MCE, &mcg_cap);

        if (ret < 0) {

            fprintf(stderr, "KVM_X86_SETUP_MCE: %s", strerror(-ret));

            return ret;

        }



        env->mcg_cap = mcg_cap;

    }



    qemu_add_vm_change_state_handler(cpu_update_state, env);



    c = cpuid_find_entry(&cpuid_data.cpuid, 1, 0);

    if (c) {

        has_msr_feature_control = !!(c->ecx & CPUID_EXT_VMX) ||

                                  !!(c->ecx & CPUID_EXT_SMX);

    }



    cpuid_data.cpuid.padding = 0;

    r = kvm_vcpu_ioctl(cs, KVM_SET_CPUID2, &cpuid_data);

    if (r) {

        return r;

    }



    r = kvm_check_extension(cs->kvm_state, KVM_CAP_TSC_CONTROL);

    if (r && env->tsc_khz) {

        r = kvm_vcpu_ioctl(cs, KVM_SET_TSC_KHZ, env->tsc_khz);

        if (r < 0) {

            fprintf(stderr, "KVM_SET_TSC_KHZ failed\n");

            return r;

        }

    }



    if (kvm_has_xsave()) {

        env->kvm_xsave_buf = qemu_memalign(4096, sizeof(struct kvm_xsave));

    }



    return 0;

}
