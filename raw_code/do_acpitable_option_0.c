static void do_acpitable_option(const char *optarg)

{

    if (acpi_table_add(optarg) < 0) {

        fprintf(stderr, "Wrong acpi table provided\n");

        exit(1);

    }

}
