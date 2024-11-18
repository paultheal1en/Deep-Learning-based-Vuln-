static bool pc_machine_get_aligned_dimm(Object *obj, Error **errp)

{

    PCMachineState *pcms = PC_MACHINE(obj);



    return pcms->enforce_aligned_dimm;

}
