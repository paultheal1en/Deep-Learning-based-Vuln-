void cpu_x86_cpuid(CPUX86State *env, uint32_t index,

                   uint32_t *eax, uint32_t *ebx,

                   uint32_t *ecx, uint32_t *edx)

{

    /* test if maximum index reached */

    if (index & 0x80000000) {

        if (index > env->cpuid_xlevel)

            index = env->cpuid_level;

    } else {

        if (index > env->cpuid_level)

            index = env->cpuid_level;

    }



    switch(index) {

    case 0:

        *eax = env->cpuid_level;

        *ebx = env->cpuid_vendor1;

        *edx = env->cpuid_vendor2;

        *ecx = env->cpuid_vendor3;



        /* sysenter isn't supported on compatibility mode on AMD.  and syscall

         * isn't supported in compatibility mode on Intel.  so advertise the

         * actuall cpu, and say goodbye to migration between different vendors

         * is you use compatibility mode. */

        if (kvm_enabled())

            host_cpuid(0, NULL, ebx, ecx, edx);

        break;

    case 1:

        *eax = env->cpuid_version;

        *ebx = (env->cpuid_apic_id << 24) | 8 << 8; /* CLFLUSH size in quad words, Linux wants it. */

        *ecx = env->cpuid_ext_features;

        *edx = env->cpuid_features;



        /* "Hypervisor present" bit required for Microsoft SVVP */

        if (kvm_enabled())

            *ecx |= (1 << 31);

        break;

    case 2:

        /* cache info: needed for Pentium Pro compatibility */

        *eax = 1;

        *ebx = 0;

        *ecx = 0;

        *edx = 0x2c307d;

        break;

    case 4:

        /* cache info: needed for Core compatibility */

        switch (*ecx) {

            case 0: /* L1 dcache info */

                *eax = 0x0000121;

                *ebx = 0x1c0003f;

                *ecx = 0x000003f;

                *edx = 0x0000001;

                break;

            case 1: /* L1 icache info */

                *eax = 0x0000122;

                *ebx = 0x1c0003f;

                *ecx = 0x000003f;

                *edx = 0x0000001;

                break;

            case 2: /* L2 cache info */

                *eax = 0x0000143;

                *ebx = 0x3c0003f;

                *ecx = 0x0000fff;

                *edx = 0x0000001;

                break;

            default: /* end of info */

                *eax = 0;

                *ebx = 0;

                *ecx = 0;

                *edx = 0;

                break;

        }



        break;

    case 5:

        /* mwait info: needed for Core compatibility */

        *eax = 0; /* Smallest monitor-line size in bytes */

        *ebx = 0; /* Largest monitor-line size in bytes */

        *ecx = CPUID_MWAIT_EMX | CPUID_MWAIT_IBE;

        *edx = 0;

        break;

    case 6:

        /* Thermal and Power Leaf */

        *eax = 0;

        *ebx = 0;

        *ecx = 0;

        *edx = 0;

        break;

    case 9:

        /* Direct Cache Access Information Leaf */

        *eax = 0; /* Bits 0-31 in DCA_CAP MSR */

        *ebx = 0;

        *ecx = 0;

        *edx = 0;

        break;

    case 0xA:

        /* Architectural Performance Monitoring Leaf */

        *eax = 0;

        *ebx = 0;

        *ecx = 0;

        *edx = 0;

        break;

    case 0x80000000:

        *eax = env->cpuid_xlevel;

        *ebx = env->cpuid_vendor1;

        *edx = env->cpuid_vendor2;

        *ecx = env->cpuid_vendor3;

        break;

    case 0x80000001:

        *eax = env->cpuid_features;

        *ebx = 0;

        *ecx = env->cpuid_ext3_features;

        *edx = env->cpuid_ext2_features;



        if (kvm_enabled()) {

            uint32_t h_eax, h_edx;



            host_cpuid(0x80000001, &h_eax, NULL, NULL, &h_edx);



            /* disable CPU features that the host does not support */



            /* long mode */

            if ((h_edx & 0x20000000) == 0 /* || !lm_capable_kernel */)

                *edx &= ~0x20000000;

            /* syscall */

            if ((h_edx & 0x00000800) == 0)

                *edx &= ~0x00000800;

            /* nx */

            if ((h_edx & 0x00100000) == 0)

                *edx &= ~0x00100000;



            /* disable CPU features that KVM cannot support */



            /* svm */

            *ecx &= ~4UL;

            /* 3dnow */

            *edx = ~0xc0000000;

        }

        break;

    case 0x80000002:

    case 0x80000003:

    case 0x80000004:

        *eax = env->cpuid_model[(index - 0x80000002) * 4 + 0];

        *ebx = env->cpuid_model[(index - 0x80000002) * 4 + 1];

        *ecx = env->cpuid_model[(index - 0x80000002) * 4 + 2];

        *edx = env->cpuid_model[(index - 0x80000002) * 4 + 3];

        break;

    case 0x80000005:

        /* cache info (L1 cache) */

        *eax = 0x01ff01ff;

        *ebx = 0x01ff01ff;

        *ecx = 0x40020140;

        *edx = 0x40020140;

        break;

    case 0x80000006:

        /* cache info (L2 cache) */

        *eax = 0;

        *ebx = 0x42004200;

        *ecx = 0x02008140;

        *edx = 0;

        break;

    case 0x80000008:

        /* virtual & phys address size in low 2 bytes. */

/* XXX: This value must match the one used in the MMU code. */ 

        if (env->cpuid_ext2_features & CPUID_EXT2_LM) {

            /* 64 bit processor */

#if defined(USE_KQEMU)

            *eax = 0x00003020;	/* 48 bits virtual, 32 bits physical */

#else

/* XXX: The physical address space is limited to 42 bits in exec.c. */

            *eax = 0x00003028;	/* 48 bits virtual, 40 bits physical */

#endif

        } else {

#if defined(USE_KQEMU)

            *eax = 0x00000020;	/* 32 bits physical */

#else

            if (env->cpuid_features & CPUID_PSE36)

                *eax = 0x00000024; /* 36 bits physical */

            else

                *eax = 0x00000020; /* 32 bits physical */

#endif

        }

        *ebx = 0;

        *ecx = 0;

        *edx = 0;

        break;

    case 0x8000000A:

        *eax = 0x00000001; /* SVM Revision */

        *ebx = 0x00000010; /* nr of ASIDs */

        *ecx = 0;

        *edx = 0; /* optional features */

        break;

    default:

        /* reserved values: zero */

        *eax = 0;

        *ebx = 0;

        *ecx = 0;

        *edx = 0;

        break;

    }

}
