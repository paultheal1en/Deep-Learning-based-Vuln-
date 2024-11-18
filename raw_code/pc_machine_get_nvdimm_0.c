static bool pc_machine_get_nvdimm(Object *obj, Error **errp)

{

    PCMachineState *pcms = PC_MACHINE(obj);



    return pcms->nvdimm;

}
