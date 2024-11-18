static void nop(DBDMA_channel *ch)

{

    dbdma_cmd *current = &ch->current;



    if (conditional_wait(ch))

        goto wait;



    current->xfer_status = cpu_to_le16(be32_to_cpu(ch->regs[DBDMA_STATUS]));

    dbdma_cmdptr_save(ch);



    conditional_interrupt(ch);

    conditional_branch(ch);



wait:

    qemu_bh_schedule(dbdma_bh);

}
