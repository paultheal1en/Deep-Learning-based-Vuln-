void sparc_cpu_list(FILE *f, int (*cpu_fprintf)(FILE *f, const char *fmt, ...))

{

    unsigned int i;



    for (i = 0; i < ARRAY_SIZE(sparc_defs); i++) {

        (*cpu_fprintf)(f, "Sparc %16s IU " TARGET_FMT_lx " FPU %08x MMU %08x NWINS %d ",

                       sparc_defs[i].name,

                       sparc_defs[i].iu_version,

                       sparc_defs[i].fpu_version,

                       sparc_defs[i].mmu_version,

                       sparc_defs[i].nwindows);

        print_features(f, cpu_fprintf, CPU_DEFAULT_FEATURES &

                       ~sparc_defs[i].features, "-");

        print_features(f, cpu_fprintf, ~CPU_DEFAULT_FEATURES &

                       sparc_defs[i].features, "+");

        (*cpu_fprintf)(f, "\n");

    }

    (*cpu_fprintf)(f, "Default CPU feature flags (use '-' to remove): ");

    print_features(f, cpu_fprintf, CPU_DEFAULT_FEATURES, NULL);

    (*cpu_fprintf)(f, "\n");

    (*cpu_fprintf)(f, "Available CPU feature flags (use '+' to add): ");

    print_features(f, cpu_fprintf, ~CPU_DEFAULT_FEATURES, NULL);

    (*cpu_fprintf)(f, "\n");

    (*cpu_fprintf)(f, "Numerical features (use '=' to set): iu_version "

                   "fpu_version mmu_version nwindows\n");

}
