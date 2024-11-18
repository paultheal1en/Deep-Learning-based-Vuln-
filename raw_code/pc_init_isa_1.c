static void pc_init_isa(MachineState *machine)

{

    has_pci_info = false;

    has_acpi_build = false;

    smbios_defaults = false;






    if (!machine->cpu_model) {

        machine->cpu_model = "486";

    }

    x86_cpu_compat_disable_kvm_features(FEAT_KVM, KVM_FEATURE_PV_EOI);

    enable_compat_apic_id_mode();

    pc_init1(machine, 0, 1);

}