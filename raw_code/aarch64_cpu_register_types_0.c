static void aarch64_cpu_register_types(void)

{

    int i;



    type_register_static(&aarch64_cpu_type_info);

    for (i = 0; i < ARRAY_SIZE(aarch64_cpus); i++) {

        aarch64_cpu_register(&aarch64_cpus[i]);

    }

}
