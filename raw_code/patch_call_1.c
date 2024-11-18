static void patch_call(VAPICROMState *s, X86CPU *cpu, target_ulong ip,

                       uint32_t target)

{

    uint32_t offset;



    offset = cpu_to_le32(target - ip - 5);

    patch_byte(cpu, ip, 0xe8); /* call near */

    cpu_memory_rw_debug(CPU(cpu), ip + 1, (void *)&offset, sizeof(offset), 1);

}
