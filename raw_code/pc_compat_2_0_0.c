static void pc_compat_2_0(MachineState *machine)

{

    pc_compat_2_1(machine);

    smbios_legacy_mode = true;

    has_reserved_memory = false;

    pc_set_legacy_acpi_data_size();

}
