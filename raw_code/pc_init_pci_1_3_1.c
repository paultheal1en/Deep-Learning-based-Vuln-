static void pc_init_pci_1_3(QEMUMachineInitArgs *args)

{

    enable_compat_apic_id_mode();

    pc_sysfw_flash_vs_rom_bug_compatible = true;

    has_pvpanic = false;

    pc_init_pci(args);

}
