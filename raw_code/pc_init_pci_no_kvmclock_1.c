static void pc_init_pci_no_kvmclock(MachineState *machine)

{

    has_pci_info = false;

    has_acpi_build = false;

    smbios_defaults = false;






    x86_cpu_compat_disable_kvm_features(FEAT_KVM, KVM_FEATURE_PV_EOI);

    enable_compat_apic_id_mode();

    pc_init1(machine, 1, 0);

}