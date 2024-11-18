static void pc_compat_2_1(MachineState *machine)

{

    PCMachineState *pcms = PC_MACHINE(machine);



    pc_compat_2_2(machine);

    pcms->enforce_aligned_dimm = false;

    smbios_uuid_encoded = false;

    x86_cpu_change_kvm_default("svm", NULL);

}
