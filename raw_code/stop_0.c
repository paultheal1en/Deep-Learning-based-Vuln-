static void stop(DBDMA_channel *ch)

{

    ch->regs[DBDMA_STATUS] &= cpu_to_be32(~(ACTIVE|DEAD|FLUSH));



    /* the stop command does not increment command pointer */

}
