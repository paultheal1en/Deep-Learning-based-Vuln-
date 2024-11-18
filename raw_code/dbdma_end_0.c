static void dbdma_end(DBDMA_io *io)

{

    DBDMA_channel *ch = io->channel;

    dbdma_cmd *current = &ch->current;



    if (conditional_wait(ch))

        goto wait;



    current->xfer_status = cpu_to_le16(be32_to_cpu(ch->regs[DBDMA_STATUS]));

    current->res_count = cpu_to_le16(be32_to_cpu(io->len));

    dbdma_cmdptr_save(ch);

    if (io->is_last)

        ch->regs[DBDMA_STATUS] &= cpu_to_be32(~FLUSH);



    conditional_interrupt(ch);

    conditional_branch(ch);



wait:

    ch->processing = 0;

    if ((ch->regs[DBDMA_STATUS] & cpu_to_be32(RUN)) &&

        (ch->regs[DBDMA_STATUS] & cpu_to_be32(ACTIVE)))

        channel_run(ch);

}
