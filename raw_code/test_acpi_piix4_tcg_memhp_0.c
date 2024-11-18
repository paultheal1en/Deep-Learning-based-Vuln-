static void test_acpi_piix4_tcg_memhp(void)

{

    test_data data;



    memset(&data, 0, sizeof(data));

    data.machine = MACHINE_PC;

    data.variant = ".memhp";

    test_acpi_one(" -m 128,slots=3,maxmem=1G -numa node", &data);

    free_test_data(&data);

}
