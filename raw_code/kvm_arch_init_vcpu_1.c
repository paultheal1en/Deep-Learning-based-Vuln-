int kvm_arch_init_vcpu(CPUState *env)

{

    struct {

        struct kvm_cpuid2 cpuid;

        struct kvm_cpuid_entry2 entries[100];

    } __attribute__((packed)) cpuid_data;

    uint32_t limit, i, j, cpuid_i;

    uint32_t eax, ebx, ecx, edx;



    cpuid_i = 0;



    cpu_x86_cpuid(env, 0, 0, &eax, &ebx, &ecx, &edx);

    limit = eax;



    for (i = 0; i <= limit; i++) {

        struct kvm_cpuid_entry2 *c = &cpuid_data.entries[cpuid_i++];



        switch (i) {

        case 2: {

            /* Keep reading function 2 till all the input is received */

            int times;



            cpu_x86_cpuid(env, i, 0, &eax, &ebx, &ecx, &edx);

            times = eax & 0xff;



            c->function = i;

            c->flags |= KVM_CPUID_FLAG_STATEFUL_FUNC;

            c->flags |= KVM_CPUID_FLAG_STATE_READ_NEXT;

            c->eax = eax;

            c->ebx = ebx;

            c->ecx = ecx;

            c->edx = edx;



            for (j = 1; j < times; ++j) {

                cpu_x86_cpuid(env, i, 0, &eax, &ebx, &ecx, &edx);

                c->function = i;

                c->flags |= KVM_CPUID_FLAG_STATEFUL_FUNC;

                c->eax = eax;

                c->ebx = ebx;

                c->ecx = ecx;

                c->edx = edx;

                c = &cpuid_data.entries[++cpuid_i];

            }

            break;

        }

        case 4:

        case 0xb:

        case 0xd:

            for (j = 0; ; j++) {

                cpu_x86_cpuid(env, i, j, &eax, &ebx, &ecx, &edx);

                c->function = i;

                c->flags = KVM_CPUID_FLAG_SIGNIFCANT_INDEX;

                c->index = j;

                c->eax = eax;

                c->ebx = ebx;

                c->ecx = ecx;

                c->edx = edx;

                c = &cpuid_data.entries[++cpuid_i];



                if (i == 4 && eax == 0)

                    break;

                if (i == 0xb && !(ecx & 0xff00))

                    break;

                if (i == 0xd && eax == 0)

                    break;

            }

            break;

        default:

            cpu_x86_cpuid(env, i, 0, &eax, &ebx, &ecx, &edx);

            c->function = i;

            c->eax = eax;

            c->ebx = ebx;

            c->ecx = ecx;

            c->edx = edx;

            break;

        }

    }

    cpu_x86_cpuid(env, 0x80000000, 0, &eax, &ebx, &ecx, &edx);

    limit = eax;



    for (i = 0x80000000; i <= limit; i++) {

        struct kvm_cpuid_entry2 *c = &cpuid_data.entries[cpuid_i++];



        cpu_x86_cpuid(env, i, 0, &eax, &ebx, &ecx, &edx);

        c->function = i;

        c->eax = eax;

        c->ebx = ebx;

        c->ecx = ecx;

        c->edx = edx;

    }



    cpuid_data.cpuid.nent = cpuid_i;



    return kvm_vcpu_ioctl(env, KVM_SET_CPUID2, &cpuid_data);

}
