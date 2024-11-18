static bool acpi_has_nvdimm(void)

{

    PCMachineState *pcms = PC_MACHINE(qdev_get_machine());



    return pcms->nvdimm;

}
