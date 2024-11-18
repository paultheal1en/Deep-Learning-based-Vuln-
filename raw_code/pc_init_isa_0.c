static void pc_init_isa(MachineState *machine)

{

    pci_enabled = false;

    has_acpi_build = false;

    smbios_defaults = false;

    gigabyte_align = false;

    smbios_legacy_mode = true;

    has_reserved_memory = false;

    option_rom_has_mr = true;

    rom_file_has_mr = false;

    if (!machine->cpu_model) {

        machine->cpu_model = "486";

    }

    x86_cpu_change_kvm_default("kvm-pv-eoi", NULL);

    enable_compat_apic_id_mode();

    pc_init1(machine, TYPE_I440FX_PCI_HOST_BRIDGE, TYPE_I440FX_PCI_DEVICE);

}
