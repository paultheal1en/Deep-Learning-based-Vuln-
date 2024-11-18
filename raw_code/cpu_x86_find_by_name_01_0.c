static int cpu_x86_find_by_name(X86CPU *cpu, x86_def_t *x86_cpu_def,

                                const char *name)

{

    x86_def_t *def;

    int i;



    if (name == NULL) {

        return -1;

    }

    if (kvm_enabled() && strcmp(name, "host") == 0) {

        kvm_cpu_fill_host(x86_cpu_def);

        object_property_set_bool(OBJECT(cpu), true, "pmu", &error_abort);

        return 0;

    }



    for (i = 0; i < ARRAY_SIZE(builtin_x86_defs); i++) {

        def = &builtin_x86_defs[i];

        if (strcmp(name, def->name) == 0) {

            memcpy(x86_cpu_def, def, sizeof(*def));

            /* sysenter isn't supported in compatibility mode on AMD,

             * syscall isn't supported in compatibility mode on Intel.

             * Normally we advertise the actual CPU vendor, but you can

             * override this using the 'vendor' property if you want to use

             * KVM's sysenter/syscall emulation in compatibility mode and

             * when doing cross vendor migration

             */

            if (kvm_enabled()) {

                uint32_t  ebx = 0, ecx = 0, edx = 0;

                host_cpuid(0, 0, NULL, &ebx, &ecx, &edx);

                x86_cpu_vendor_words2str(x86_cpu_def->vendor, ebx, edx, ecx);

            }

            return 0;

        }

    }



    return -1;

}
