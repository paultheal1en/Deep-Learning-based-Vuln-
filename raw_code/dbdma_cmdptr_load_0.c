static void dbdma_cmdptr_load(DBDMA_channel *ch)

{

    DBDMA_DPRINTF("dbdma_cmdptr_load 0x%08x\n",

                  be32_to_cpu(ch->regs[DBDMA_CMDPTR_LO]));

    cpu_physical_memory_read(be32_to_cpu(ch->regs[DBDMA_CMDPTR_LO]),

                             (uint8_t*)&ch->current, sizeof(dbdma_cmd));

}
