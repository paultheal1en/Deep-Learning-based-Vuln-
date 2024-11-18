hwaddr get_pteg_offset32(PowerPCCPU *cpu, hwaddr hash)

{

    CPUPPCState *env = &cpu->env;



    return (hash * HASH_PTEG_SIZE_32) & env->htab_mask;

}
