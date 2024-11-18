static void pc_compat_1_7(MachineState *machine)

{

    pc_compat_2_0(machine);

    smbios_defaults = false;

    gigabyte_align = false;

    option_rom_has_mr = true;

    x86_cpu_change_kvm_default("x2apic", NULL);

}
